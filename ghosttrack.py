"""
GhostTrack: robust 2D indoor positioning from a few fixed anchors.

Pipeline (matches the deck):
  1. Geometric contradiction engine  - robust multilateration; the anchor whose
     distance disagrees with the others (the "liar") is downweighted, not deleted.
  2. CNN-LSTM tracker (PyTorch)      - Conv1D smooths local noise, LSTM carries
     motion/memory (this is what keeps predicting through dropouts and blackouts).
     Trained with Huber loss. A second head predicts an error radius (confidence).
  3. Evaluation                      - vs. plain least-squares multilateration,
     split by NLOS-heavy windows and by number of visible anchors.

Usage:
    python generate_dataset.py          # creates dataset/*.csv
    python ghosttrack.py                # trains and evaluates
    python ghosttrack.py --geometry-only   # no PyTorch needed; tests step 1 only
"""
import argparse
import numpy as np
import pandas as pd

MAX_A = 6
SCALE = 25.0      # normaliser (rooms are <= 25 m)
WINDOW = 40       # timesteps the tracker looks at (4 s at 10 Hz)


# ----------------------------------------------------------------------------
# 1. Geometric contradiction engine (NumPy only, no labels needed)
# ----------------------------------------------------------------------------
def multilaterate(anchors, d, w=None, iters=25, p0=None):
    """Damped (Levenberg-Marquardt) weighted least squares. anchors:(n,2) d:(n,) -> (2,).
    The estimate is clipped to the anchors' bounding box (+1 m) so near-collinear
    anchor geometry can never send it off to infinity."""
    if w is None:
        w = np.ones(len(d))
    lo, hi = anchors.min(0) - 1.0, anchors.max(0) + 1.0
    p = np.clip(anchors.mean(0) if p0 is None else p0.copy(), lo, hi)
    lam = 1e-2
    cost = np.sum(w * (np.linalg.norm(p - anchors, axis=1) - d) ** 2)
    for _ in range(iters):
        diff = p - anchors
        rng = np.linalg.norm(diff, axis=1) + 1e-9
        J = diff / rng[:, None]
        r = rng - d
        H = J.T @ (J * w[:, None])
        g = J.T @ (w * r)
        step = np.linalg.solve(H + lam * np.eye(2) * (np.trace(H) + 1e-6), g)
        cand = np.clip(p - step, lo, hi)
        c2 = np.sum(w * (np.linalg.norm(cand - anchors, axis=1) - d) ** 2)
        if c2 < cost:
            p, cost, lam = cand, c2, max(lam * 0.5, 1e-6)
            if np.linalg.norm(step) < 1e-4:
                break
        else:
            lam *= 4
    return p


def geometric_engine(anchors, d, tau=0.6):
    """
    Robust multilateration = the "liar finder".
      1. Try the full anchor set and every leave-one-out subset (cheap: <= 7 fits).
      2. Score each candidate position by a truncated residual cost over ALL anchors,
         so a candidate that ignores a contradicting anchor wins when that anchor lies.
      3. Refine with a few reweighting rounds; weights become reliability scores.
    Returns (position or None if < 3 anchors, per-anchor reliability in (0,1]).
    Readings are kept with a reliability score, never thrown away.
    """
    n = len(d)
    if n < 3:
        return None, np.ones(n)
    cands = [np.ones(n)]
    if n >= 4:
        for k in range(n):
            m = np.ones(n); m[k] = 0.0
            cands.append(m)
    best_p, best_score = None, 1e18
    for m in cands:
        keep = m > 0
        p = multilaterate(anchors[keep], d[keep])
        res = np.linalg.norm(p - anchors, axis=1) - d      # negative = reading is LONGER than geometry
        e = np.where(res < 0, -res, res / 3.0)             # long readings are suspicious, short ones much less
        score = np.sum(np.minimum(e ** 2, 4.0)) + 0.5 * (n - keep.sum())  # penalise dropping an anchor
        if score < best_score:
            best_p, best_score = p, score
    p = best_p
    def weights(p):
        res = np.linalg.norm(p - anchors, axis=1) - d
        e = np.where(res < 0, -res, res / 3.0)             # NLOS only lengthens: penalise long readings
        return 1.0 / (1.0 + (e / tau) ** 2) ** 2
    for _ in range(4):
        p = multilaterate(anchors, d, weights(p), p0=p)
    return p, weights(p)


def run_geometry(df):
    """Adds columns: ls_x, ls_y (plain least squares), geo_x, geo_y (robust), w0..w5."""
    n = len(df)
    ls = np.full((n, 2), np.nan)
    geo = np.full((n, 2), np.nan)
    W = np.ones((n, MAX_A))
    D = df[[f"d{a}" for a in range(MAX_A)]].to_numpy()
    AX = df[[f"a{a}_x" for a in range(MAX_A)]].to_numpy()
    AY = df[[f"a{a}_y" for a in range(MAX_A)]].to_numpy()
    for i in range(n):
        ok = ~np.isnan(D[i])
        if ok.sum() < 3:
            continue
        A = np.stack([AX[i][ok], AY[i][ok]], 1)
        ls[i] = multilaterate(A, D[i][ok])
        p, w = geometric_engine(A, D[i][ok])
        geo[i] = p
        W[i][ok] = w
    df["ls_x"], df["ls_y"] = ls[:, 0], ls[:, 1]
    df["geo_x"], df["geo_y"] = geo[:, 0], geo[:, 1]
    for a in range(MAX_A):
        df[f"w{a}"] = W[:, a]
    return df


# ----------------------------------------------------------------------------
# 2. Features / windows
# ----------------------------------------------------------------------------
def build_windows(df):
    """Returns X:(N,WINDOW,F), Y:(N,2) true pos / SCALE, meta DataFrame for the last step."""
    Xs, Ys, meta = [], [], []
    for _, g in df.groupby("session_id", sort=False):
        g = g.reset_index(drop=True)
        steps = len(g)
        feats = []
        for a in range(MAX_A):
            vis = g[f"vis{a}"].to_numpy()
            d = np.nan_to_num(g[f"d{a}"].to_numpy()) / SCALE
            ax = np.nan_to_num(g[f"a{a}_x"].to_numpy()) / SCALE
            ay = np.nan_to_num(g[f"a{a}_y"].to_numpy()) / SCALE
            wt = g[f"w{a}"].to_numpy() * vis
            feats += [d * vis, vis, ax, ay, wt]
        geo = np.nan_to_num(g[["geo_x", "geo_y"]].to_numpy()) / SCALE
        has_geo = (~np.isnan(g["geo_x"].to_numpy())).astype(float)
        imu = g[["acc_x", "acc_y", "gyro_z"]].to_numpy() / np.array([3.0, 3.0, 2.0])
        F = np.column_stack(feats + [geo, has_geo, imu]).astype(np.float32)
        Y = g[["x", "y"]].to_numpy() / SCALE
        for t in range(WINDOW - 1, steps):
            Xs.append(F[t - WINDOW + 1: t + 1])
            Ys.append(Y[t])
            meta.append(g.loc[t, ["session_id", "t", "ls_x", "ls_y", "geo_x", "geo_y",
                                  "x", "y", "n_anchors"]
                              + [f"vis{a}" for a in range(MAX_A)]
                              + [f"nlos{a}" for a in range(MAX_A)]])
    return np.array(Xs, np.float32), np.array(Ys, np.float32), pd.DataFrame(meta)


# ----------------------------------------------------------------------------
# 3. CNN-LSTM tracker (PyTorch)
# ----------------------------------------------------------------------------
def build_model(n_feat, hidden=96):
    import torch.nn as nn

    class CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            # Conv1D: slides over TIME, features are channels -> smooths local noise
            self.conv = nn.Sequential(
                nn.Conv1d(n_feat, 64, kernel_size=5, padding=2), nn.ReLU(),
                nn.Conv1d(64, 64, kernel_size=3, padding=1), nn.ReLU())
            # LSTM: momentum / memory over the window (carries through dropouts)
            self.lstm = nn.LSTM(64, hidden, num_layers=1, batch_first=True)
            self.pos = nn.Linear(hidden, 2)                                   # x, y
            self.rad = nn.Sequential(nn.Linear(hidden, 1), nn.Softplus())     # error radius

        def forward(self, x):                      # x: (B, T, F)
            h = self.conv(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.lstm(h)
            last = out[:, -1]
            return self.pos(last), self.rad(last).squeeze(-1)

    return CNNLSTM()


def train_model(Xtr, Ytr, Xva, Yva, epochs=12, lr=2e-3, batch=256, huber_delta=0.04):
    import torch
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(Xtr.shape[2]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    huber = torch.nn.HuberLoss(delta=huber_delta)   # ~1 m in normalised units; resists outliers
    Xtr, Ytr = torch.tensor(Xtr), torch.tensor(Ytr)
    Xva_t, Yva_t = torch.tensor(Xva).to(dev), torch.tensor(Yva).to(dev)
    best, best_state = 1e9, None
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), batch):
            idx = perm[i:i + batch]
            xb, yb = Xtr[idx].to(dev), Ytr[idx].to(dev)
            pos, rad = model(xb)
            err = (pos - yb).norm(dim=1)
            loss = huber(pos, yb) + 0.5 * huber(rad, err.detach())  # radius learns to match error
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv, _ = model(Xva_t)
            verr = ((pv - Yva_t).norm(dim=1) * SCALE).mean().item()
        sched.step(verr)
        print(f"epoch {ep + 1:2d}/{epochs}  val mean error = {verr:.2f} m")
        if verr < best:
            best, best_state = verr, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, dev


def predict(model, dev, X):
    import torch
    model.eval()
    P, R = [], []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            p, r = model(torch.tensor(X[i:i + 1024]).to(dev))
            P.append(p.cpu().numpy()); R.append(r.cpu().numpy())
    return np.concatenate(P) * SCALE, np.concatenate(R) * SCALE


# ----------------------------------------------------------------------------
# 4. Evaluation
# ----------------------------------------------------------------------------
def stats(err):
    err = err[~np.isnan(err)]
    if len(err) == 0:
        return "n/a"
    return (f"RMSE {np.sqrt((err ** 2).mean()):5.2f} m | median {np.median(err):5.2f} m | "
            f"90th pct {np.percentile(err, 90):5.2f} m | n={len(err)}")


def report(meta, model_pos=None, model_rad=None):
    truth = meta[["x", "y"]].to_numpy()
    vis = meta[[f"vis{a}" for a in range(MAX_A)]].to_numpy()
    nl = meta[[f"nlos{a}" for a in range(MAX_A)]].to_numpy()
    n_vis = vis.sum(1)
    n_nlos_vis = np.nansum(nl * vis, 1)
    e_ls = np.linalg.norm(meta[["ls_x", "ls_y"]].to_numpy() - truth, axis=1)
    e_geo = np.linalg.norm(meta[["geo_x", "geo_y"]].to_numpy() - truth, axis=1)
    e_ml = np.linalg.norm(model_pos - truth, axis=1) if model_pos is not None else None
    groups = {
        "ALL timesteps": np.ones(len(meta), bool),
        ">=3 anchors visible, no NLOS": (n_vis >= 3) & (n_nlos_vis == 0),
        ">=3 anchors visible, 1+ NLOS": (n_vis >= 3) & (n_nlos_vis >= 1),
        "fewer than 3 anchors (blind spot / blackout)": n_vis < 3,
    }
    for name, m in groups.items():
        print(f"\n[{name}]  ({m.sum()} steps)")
        print("  plain least-squares :", stats(e_ls[m]))
        print("  geometric engine    :", stats(e_geo[m]))
        if e_ml is not None:
            print("  CNN-LSTM (ours)     :", stats(e_ml[m]))
    if e_ml is not None and model_rad is not None:
        cov = (e_ml <= model_rad).mean()
        cov2 = (e_ml <= 2 * model_rad).mean()
        print(f"\nConfidence check: error <= predicted radius in {cov:.0%} of steps; "
              f"<= 2x radius in {cov2:.0%}")
    print("\nNote: least-squares / geometric are 'n/a' when < 3 anchors are visible;"
          " the CNN-LSTM still outputs a position there.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geometry-only", action="store_true")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--max-sessions", type=int, default=0, help="limit sessions per split (quick test)")
    args = ap.parse_args()

    splits = {}
    for name in ("train", "val", "test"):
        df = pd.read_csv(f"dataset/{name}.csv")
        if args.max_sessions:
            keep = df.session_id.unique()[: args.max_sessions]
            df = df[df.session_id.isin(keep)].copy()
        print(f"running geometric engine on {name} ({len(df)} rows)...")
        splits[name] = run_geometry(df)

    if args.geometry_only:
        _, _, meta = build_windows(splits["test"])
        report(meta)
        return

    data = {k: build_windows(v) for k, v in splits.items()}
    model, dev = train_model(data["train"][0], data["train"][1], data["val"][0], data["val"][1],
                             epochs=args.epochs)
    pos, rad = predict(model, dev, data["test"][0])
    print("\n===== TEST SET (rooms never seen in training) =====")
    report(data["test"][2], pos, rad)
    import torch
    torch.save(model.state_dict(), "ghosttrack_model.pt")
    print("\nsaved ghosttrack_model.pt")


if __name__ == "__main__":
    main()
