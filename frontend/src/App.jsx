import React, { useEffect, useRef, useState } from 'react';

const WS_URL = import.meta.env.VITE_BACKEND_WS_URL || 'ws://127.0.0.1:8000/ws';
const HTTP_URL = import.meta.env.VITE_BACKEND_HTTP_URL || 'http://127.0.0.1:8000';

const DEFAULT_LAYOUT = {
  room_id: 'room_demo',
  width: 12.0,
  height: 10.0,
  anchors: [
    { id: 'A1', x: 1.0, y: 1.0, online: true },
    { id: 'A2', x: 11.0, y: 1.0, online: true },
    { id: 'A3', x: 11.0, y: 9.0, online: true },
    { id: 'A4', x: 1.0, y: 9.0, online: true },
  ],
  walls: [
    { x1: 5.0, y1: 0.0, x2: 5.0, y2: 5.5 },
    { x1: 5.0, y1: 5.5, x2: 9.0, y2: 5.5 },
  ],
};

export default function App() {
  const [connected, setConnected] = useState(false);
  const [frame, setFrame] = useState(null);
  const [layout, setLayout] = useState(DEFAULT_LAYOUT);
  const [noiseSigma, setNoiseSigma] = useState(0.15);
  const [nlosBias, setNlosBias] = useState(2.5);
  const [showCompare, setShowCompare] = useState(true);
  const [showRadii, setShowRadii] = useState(true);
  const [drawMode, setDrawMode] = useState('none'); // 'none' | 'wall' | 'anchor'
  const [dragStart, setDragStart] = useState(null);
  const [dragCurrent, setDragCurrent] = useState(null);

  const wsRef = useRef(null);
  const canvasRef = useRef(null);
  const gtTrailRef = useRef([]);
  const estTrailRef = useRef([]);
  const naiveTrailRef = useRef([]);

  // Fetch initial layout config
  useEffect(() => {
    fetch(`${HTTP_URL}/config`)
      .then((r) => r.json())
      .then((data) => {
        if (data.layout) setLayout(data.layout);
        if (typeof data.noise_sigma === 'number') setNoiseSigma(data.noise_sigma);
        if (typeof data.nlos_bias === 'number') setNlosBias(data.nlos_bias);
      })
      .catch(() => {});
  }, []);

  // Connect WebSocket with auto-reconnect
  useEffect(() => {
    let active = true;
    let reconnectTimer = null;

    function connect() {
      if (!active) return;
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        if (active) setConnected(true);
      };

      ws.onmessage = (evt) => {
        if (!active) return;
        try {
          const msg = JSON.parse(evt.data);
          setFrame(msg);

          if (msg.ground_truth) {
            gtTrailRef.current = [
              ...gtTrailRef.current.slice(-55),
              { x: msg.ground_truth.x, y: msg.ground_truth.y },
            ];
          }
          estTrailRef.current = [
            ...estTrailRef.current.slice(-55),
            { x: msg.x, y: msg.y, state: msg.state },
          ];
          const naive = (msg.competing_trackers || []).find((t) =>
            t.name.includes('Naive')
          );
          if (naive && typeof naive.x === 'number') {
            naiveTrailRef.current = [
              ...naiveTrailRef.current.slice(-40),
              { x: naive.x, y: naive.y },
            ];
          }
        } catch (err) {
          // ignore malformed frame
        }
      };

      ws.onclose = () => {
        if (!active) return;
        setConnected(false);
        reconnectTimer = setTimeout(connect, 1000);
      };
    }

    connect();
    return () => {
      active = false;
      if (reconnectTimer) clearTimeout(reconnectTimer);
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const sendCommand = (cmdObj) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmdObj));
    }
  };

  // Canvas Rendering: Floor plan, Belief Heatmap, Links, Trails, Target & Callout
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const roomW = layout.width || 12.0;
    const roomH = layout.height || 10.0;

    const toPxX = (mx) => (mx / roomW) * W;
    const toPxY = (my) => H - (my / roomH) * H; // Y=0 at bottom

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#050912';
    ctx.fillRect(0, 0, W, H);

    // 1. Render 2D Belief Heatmap
    if (frame && frame.belief_grid && frame.belief_grid.length > 0) {
      const rows = frame.belief_grid.length;
      const cols = frame.belief_grid[0].length;
      const cellW = W / cols;
      const cellH = H / rows;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const val = frame.belief_grid[r][c];
          if (val < 0.04) continue;
          const py = H - (r + 1) * cellH;
          const px = c * cellW;

          let hue = '16, 185, 129'; // Emerald for Tracked
          if (frame.state === 'Ambiguous') hue = '245, 158, 11';
          else if (frame.state === 'Ghost') hue = '168, 85, 247';
          else if (frame.state === 'Lost') hue = '239, 68, 68';

          ctx.fillStyle = `rgba(${hue}, ${Math.min(0.68, val * 0.65)})`;
          ctx.fillRect(px, py, cellW + 1, cellH + 1);
        }
      }
    }

    // 2. Render Subtle Meter Grid
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.09)';
    ctx.lineWidth = 1;
    for (let gx = 1; gx < roomW; gx++) {
      ctx.beginPath();
      ctx.moveTo(toPxX(gx), 0);
      ctx.lineTo(toPxX(gx), H);
      ctx.stroke();
    }
    for (let gy = 1; gy < roomH; gy++) {
      ctx.beginPath();
      ctx.moveTo(0, toPxY(gy));
      ctx.lineTo(W, toPxY(gy));
      ctx.stroke();
    }

    // 3. Render Interior Walls (USP #07 Map-Aware)
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    (layout.walls || []).forEach((w) => {
      ctx.beginPath();
      ctx.moveTo(toPxX(w.x1), toPxY(w.y1));
      ctx.lineTo(toPxX(w.x2), toPxY(w.y2));
      ctx.stroke();
    });

    // Preview wall while dragging
    if (drawMode === 'wall' && dragStart && dragCurrent) {
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 4;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(toPxX(dragStart.x), toPxY(dragStart.y));
      ctx.lineTo(toPxX(dragCurrent.x), toPxY(dragCurrent.y));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // 4. Render Anchor Range Circles & Diagnostic Links (USP #04)
    if (frame && frame.scored_readings) {
      frame.scored_readings.forEach((sr) => {
        const anchor = (layout.anchors || []).find((a) => a.id === sr.anchor_id);
        if (!anchor || !sr.online) return;

        const ax = toPxX(anchor.x);
        const ay = toPxY(anchor.y);
        const tx = toPxX(frame.x);
        const ty = toPxY(frame.y);

        if (showRadii) {
          ctx.beginPath();
          ctx.arc(ax, ay, (sr.distance / roomW) * W, 0, Math.PI * 2);
          ctx.strokeStyle = sr.is_liar
            ? 'rgba(239, 68, 68, 0.32)'
            : 'rgba(6, 182, 212, 0.16)';
          ctx.lineWidth = sr.is_liar ? 2 : 1;
          ctx.stroke();
        }

        // Link ray from anchor to target
        ctx.beginPath();
        ctx.moveTo(ax, ay);
        ctx.lineTo(tx, ty);
        if (sr.is_liar) {
          ctx.strokeStyle = 'rgba(239, 68, 68, 0.85)';
          ctx.lineWidth = 2.2;
          ctx.setLineDash([6, 4]);
        } else {
          ctx.strokeStyle = 'rgba(16, 185, 129, 0.35)';
          ctx.lineWidth = 1.3;
          ctx.setLineDash([]);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        // Midpoint diagnostic label
        const mx = (ax + tx) / 2;
        const my = (ay + ty) / 2;
        ctx.font = '600 11px sans-serif';
        ctx.fillStyle = sr.is_liar ? '#fca5a5' : '#94a3b8';
        const label = sr.is_liar
          ? `${sr.distance.toFixed(1)}m (NLOS +${sr.estimated_bias.toFixed(1)}m)`
          : `${sr.distance.toFixed(1)}m`;
        ctx.fillText(label, mx + 4, my - 4);
      });
    }

    // 5. Render Plausible Bimodal Locations when Ambiguous (USP #01)
    if (frame && frame.plausible_locations && frame.plausible_locations.length > 1) {
      frame.plausible_locations.forEach((pt, idx) => {
        const px = toPxX(pt.x);
        const py = toPxY(pt.y);
        ctx.beginPath();
        ctx.arc(px, py, 14, 0, Math.PI * 2);
        ctx.strokeStyle = '#fbbf24';
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#fde68a';
        ctx.font = '700 11px sans-serif';
        ctx.fillText(`Mode ${idx + 1} (${Math.round(pt.weight * 100)}%)`, px + 16, py + 4);
      });
    }

    // 6. Render Trajectories (Ground Truth, Naive Trilateration, GhostTrack)
    if (gtTrailRef.current.length > 1) {
      ctx.beginPath();
      gtTrailRef.current.forEach((p, i) => {
        const px = toPxX(p.x);
        const py = toPxY(p.y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.35)';
      ctx.lineWidth = 2;
      ctx.stroke();
    }

    if (showCompare && naiveTrailRef.current.length > 1) {
      ctx.beginPath();
      naiveTrailRef.current.forEach((p, i) => {
        const px = toPxX(p.x);
        const py = toPxY(p.y);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.strokeStyle = 'rgba(239, 68, 68, 0.65)';
      ctx.lineWidth = 1.8;
      ctx.setLineDash([3, 3]);
      ctx.stroke();
      ctx.setLineDash([]);

      const lastN = naiveTrailRef.current[naiveTrailRef.current.length - 1];
      ctx.beginPath();
      ctx.arc(toPxX(lastN.x), toPxY(lastN.y), 6, 0, Math.PI * 2);
      ctx.fillStyle = '#ef4444';
      ctx.fill();
    }

    if (estTrailRef.current.length > 1) {
      for (let i = 1; i < estTrailRef.current.length; i++) {
        const p0 = estTrailRef.current[i - 1];
        const p1 = estTrailRef.current[i];
        ctx.beginPath();
        ctx.moveTo(toPxX(p0.x), toPxY(p0.y));
        ctx.lineTo(toPxX(p1.x), toPxY(p1.y));
        if (p1.state === 'Ghost' || p1.state === 'Lost') {
          ctx.strokeStyle = '#c084fc';
          ctx.setLineDash([5, 4]);
        } else if (p1.state === 'Ambiguous') {
          ctx.strokeStyle = '#fbbf24';
          ctx.setLineDash([]);
        } else {
          ctx.strokeStyle = '#10b981';
          ctx.setLineDash([]);
        }
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      ctx.setLineDash([]);
    }

    // 7. Render Anchors
    (layout.anchors || []).forEach((a) => {
      const sr =
        frame &&
        (frame.scored_readings || []).find((r) => r.anchor_id === a.id);
      const isOnline = sr ? sr.online : a.online;
      const isLiar = sr ? sr.is_liar : false;

      const ax = toPxX(a.x);
      const ay = toPxY(a.y);

      ctx.beginPath();
      ctx.arc(ax, ay, 11, 0, Math.PI * 2);
      if (!isOnline) ctx.fillStyle = '#334155';
      else if (isLiar) ctx.fillStyle = '#ef4444';
      else ctx.fillStyle = '#06b6d4';
      ctx.fill();
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#0f172a';
      ctx.stroke();

      ctx.fillStyle = '#ffffff';
      ctx.font = '800 11px sans-serif';
      ctx.fillText(
        `${a.id}${!isOnline ? ' (OFF)' : isLiar ? ' (LIAR)' : ''}`,
        ax - 12,
        ay - 15
      );
    });

    // 8. Render Ground Truth Marker & Fixed-Size Target Marker with Defined Confidence Callout (USP #03)
    if (frame && frame.ground_truth) {
      const gx = toPxX(frame.ground_truth.x);
      const gy = toPxY(frame.ground_truth.y);
      ctx.beginPath();
      ctx.arc(gx, gy, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#ffffff';
      ctx.fill();
    }

    if (frame) {
      const tx = toPxX(frame.x);
      const ty = toPxY(frame.y);

      // Fixed-size target marker (PPT Slide 6 specification)
      ctx.beginPath();
      ctx.arc(tx, ty, 8, 0, Math.PI * 2);
      ctx.fillStyle =
        frame.state === 'Ghost'
          ? '#c084fc'
          : frame.state === 'Lost'
          ? '#ef4444'
          : frame.state === 'Ambiguous'
          ? '#fbbf24'
          : '#10b981';
      ctx.fill();
      ctx.lineWidth = 2.5;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();

      // Callout label beside marker: "88% prob within 2m"
      const confPct = Math.round((frame.confidence_within_2m || 0.85) * 100);
      const calloutText = `${confPct}% within 2m (σ=${frame.sigma.toFixed(2)}m)`;
      ctx.font = '700 12px sans-serif';
      const textW = ctx.measureText(calloutText).width;
      const boxX = Math.min(W - textW - 18, tx + 14);
      const boxY = Math.max(24, ty - 12);

      ctx.fillStyle = 'rgba(15, 23, 42, 0.88)';
      ctx.fillRect(boxX - 5, boxY - 14, textW + 10, 20);
      ctx.strokeStyle = '#38bdf8';
      ctx.lineWidth = 1;
      ctx.strokeRect(boxX - 5, boxY - 14, textW + 10, 20);

      ctx.fillStyle = '#f8fafc';
      ctx.fillText(calloutText, boxX, boxY);
    }
  }, [frame, layout, showCompare, showRadii, drawMode, dragStart, dragCurrent]);

  // Interactive Canvas Handlers (Draw Wall or Add Anchor)
  const getCanvasRoomCoords = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const mx = +((px / rect.width) * layout.width).toFixed(2);
    const my = +(((rect.height - py) / rect.height) * layout.height).toFixed(2);
    return { x: mx, y: my };
  };

  const handleCanvasMouseDown = (e) => {
    const pt = getCanvasRoomCoords(e);
    if (drawMode === 'wall') {
      setDragStart(pt);
      setDragCurrent(pt);
    } else if (drawMode === 'anchor') {
      const nextId = `A${layout.anchors.length + 1}`;
      const updated = {
        ...layout,
        anchors: [...layout.anchors, { id: nextId, x: pt.x, y: pt.y, online: true }],
      };
      setLayout(updated);
      sendCommand({ command: 'load_layout', layout: updated });
      setDrawMode('none');
    }
  };

  const handleCanvasMouseMove = (e) => {
    if (drawMode === 'wall' && dragStart) {
      setDragCurrent(getCanvasRoomCoords(e));
    }
  };

  const handleCanvasMouseUp = () => {
    if (drawMode === 'wall' && dragStart && dragCurrent) {
      const dist = Math.hypot(dragCurrent.x - dragStart.x, dragCurrent.y - dragStart.y);
      if (dist > 0.4) {
        const updated = {
          ...layout,
          walls: [
            ...layout.walls,
            { x1: dragStart.x, y1: dragStart.y, x2: dragCurrent.x, y2: dragCurrent.y },
          ],
        };
        setLayout(updated);
        sendCommand({ command: 'load_layout', layout: updated });
      }
      setDragStart(null);
      setDragCurrent(null);
    }
  };

  const currentState = frame ? frame.state : 'Tracked';
  const confPct = frame ? Math.round(frame.confidence_within_2m * 100) : 95;
  const ageMs = frame ? frame.measurement_age_ms : 0;

  return (
    <div className="app-shell">
      {/* Top Header */}
      <header className="top-header">
        <div className="brand-block">
          <h1>
            InSight AI · GhostTrack
            <span className="team-pill">Team Kajukatli · Diptanshu Jamar</span>
          </h1>
          <div className="brand-subtitle">
            Ambiguity-aware, explainable indoor positioning via geometric outlier rejection &amp; predictive dead-reckoning
          </div>
        </div>

        <div className="header-badges">
          <div className={`state-badge state-${currentState}`}>
            STATE: {currentState}
            {frame && !frame.recovery_verified ? ' (VERIFYING FIX)' : ''}
          </div>
          <div className="metric-chip">
            Anchors Online:{' '}
            <strong>
              {frame ? `${frame.anchors_online} / ${frame.anchors_total}` : '4 / 4'}
            </strong>
          </div>
          <div className="metric-chip">
            Within 2m Confidence: <strong>{confPct}%</strong>
          </div>
          <div className="metric-chip">
            Packet Age: <strong>{ageMs.toFixed(0)} ms</strong>
          </div>
          <div className="metric-chip">
            WS: <strong style={{ color: connected ? '#34d399' : '#f87171' }}>{connected ? 'LIVE 10Hz' : 'RECONNECTING'}</strong>
          </div>
        </div>
      </header>

      {/* 5-Stage Pipeline Status Bar */}
      <section className="pipeline-bar">
        <div className="stage-step active">
          <span className="stage-num">01 Sensor Stream</span>
          <span className="stage-title">MQTT Telemetry (10 Hz)</span>
          <span className="stage-desc">UWB/BLE Ranging + 6-DoF IMU</span>
        </div>
        <div
          className={`stage-step ${
            frame && frame.rejected_anchors?.length > 0 ? 'warn' : 'active'
          }`}
        >
          <span className="stage-num">02 Geometric Filter</span>
          <span className="stage-title">Zero-Label Liar Check</span>
          <span className="stage-desc">
            {frame && frame.rejected_anchors?.length > 0
              ? `Downweighted: ${frame.rejected_anchors.join(', ')}`
              : 'All intersecting radii consistent'}
          </span>
        </div>
        <div className="stage-step active">
          <span className="stage-num">03 Self-Trained AI + CNN-LSTM</span>
          <span className="stage-title">Scorer + Huber Momentum</span>
          <span className="stage-desc">Range bias correction &amp; Conv1D smoothing</span>
        </div>
        <div
          className={`stage-step ${
            currentState === 'Ghost' || currentState === 'Lost' ? 'ghost' : 'active'
          }`}
        >
          <span className="stage-num">04 Ghost-Track &amp; Adviser</span>
          <span className="stage-title">
            {currentState === 'Ghost'
              ? 'ACTIVE: IMU + LSTM Fusion'
              : currentState === 'Lost'
              ? 'OUTAGE LIMIT: Target Lost'
              : 'Standby (Coverage Verified)'}
          </span>
          <span className="stage-desc">Controlled recovery &amp; anchor ranking</span>
        </div>
        <div className="stage-step active">
          <span className="stage-num">05 Live Belief Heatmap</span>
          <span className="stage-title">
            {currentState === 'Ambiguous' ? 'Bimodal Corridor Split' : `σ = ${frame ? frame.sigma.toFixed(2) : '0.32'} m`}
          </span>
          <span className="stage-desc">{confPct}% probability within 2.0 m</span>
        </div>
      </section>

      {/* Main Two-Column Content */}
      <div className="dashboard-grid">
        {/* Left Column: Live Map-Aware Belief Heatmap + Sensor Diagnostics */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                Live Map-Aware Belief-Region Heatmap &amp; Floor Plan ({layout.width}m × {layout.height}m)
              </div>
              <div className="btn-row">
                <button
                  className={showCompare ? 'active-mode' : ''}
                  onClick={() => setShowCompare(!showCompare)}
                >
                  {showCompare ? '✔ Naive Trilateration Shown' : 'Show Naive Trilateration'}
                </button>
                <button
                  className={showRadii ? 'active-mode' : ''}
                  onClick={() => setShowRadii(!showRadii)}
                >
                  {showRadii ? '✔ Range Circles Shown' : 'Show Range Circles'}
                </button>
                <button
                  className={drawMode === 'wall' ? 'btn-amber' : ''}
                  onClick={() => setDrawMode(drawMode === 'wall' ? 'none' : 'wall')}
                >
                  {drawMode === 'wall' ? '✎ Dragging Wall on Map...' : '+ Draw Wall on Map'}
                </button>
                <button
                  className={drawMode === 'anchor' ? 'btn-amber' : ''}
                  onClick={() => setDrawMode(drawMode === 'anchor' ? 'none' : 'anchor')}
                >
                  {drawMode === 'anchor' ? '📍 Click Map to Place...' : '+ Place Anchor'}
                </button>
              </div>
            </div>

            <div className="canvas-wrapper">
              <canvas
                ref={canvasRef}
                width={840}
                height={520}
                className="floorplan-canvas"
                onMouseDown={handleCanvasMouseDown}
                onMouseMove={handleCanvasMouseMove}
                onMouseUp={handleCanvasMouseUp}
              />
            </div>

            <div className="legend-row">
              <span className="legend-item">
                <span className="dot" style={{ background: '#ffffff' }} /> Ground Truth
              </span>
              <span className="legend-item">
                <span className="dot" style={{ background: '#10b981' }} /> GhostTrack (Tracked)
              </span>
              <span className="legend-item">
                <span className="dot" style={{ background: '#fbbf24' }} /> Ambiguous (Bimodal)
              </span>
              <span className="legend-item">
                <span className="dot" style={{ background: '#c084fc' }} /> Ghost Mode (Dead-Reckoning)
              </span>
              <span className="legend-item">
                <span className="dot" style={{ background: '#ef4444' }} /> Naive Trilateration / NLOS Liar
              </span>
            </div>
          </div>

          {/* Sensor Diagnostics Table (USP #04) */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                USP #04 · Per-Anchor Sensor Diagnostics, Reliability Weights &amp; AI Bias Estimates
              </div>
            </div>
            <table className="diag-table">
              <thead>
                <tr>
                  <th>Anchor</th>
                  <th>Status</th>
                  <th>Observed Dist</th>
                  <th>AI Est. Bias</th>
                  <th>Corrected Dist</th>
                  <th>Geo Residual</th>
                  <th>Weight (w)</th>
                  <th>Diagnostic Explanation</th>
                </tr>
              </thead>
              <tbody>
                {(frame?.scored_readings || []).map((sr) => (
                  <tr key={sr.anchor_id}>
                    <td>
                      <strong>{sr.anchor_id}</strong>
                    </td>
                    <td>
                      {!sr.online ? (
                        <span style={{ color: '#94a3b8' }}>OFFLINE</span>
                      ) : sr.is_liar ? (
                        <span className="badge-liar">LIAR DOWNWEIGHTED</span>
                      ) : (
                        <span className="badge-clean">CLEAN LOS</span>
                      )}
                    </td>
                    <td>{sr.online ? `${sr.distance.toFixed(2)} m` : '—'}</td>
                    <td>
                      {sr.online && sr.estimated_bias > 0.05
                        ? `+${sr.estimated_bias.toFixed(2)} m`
                        : '0.00 m'}
                    </td>
                    <td>{sr.online ? `${sr.corrected_distance.toFixed(2)} m` : '—'}</td>
                    <td>{sr.online ? `${sr.residual.toFixed(2)} m` : '—'}</td>
                    <td>
                      <strong>{sr.reliability_score.toFixed(2)}</strong>
                    </td>
                    <td style={{ color: sr.is_liar ? '#fca5a5' : '#94a3b8' }}>
                      {sr.reason}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right Column: Stress-Test Chaos Console, Recovery Adviser, Competing Trackers & Audit Log */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '14px' }}>
          {/* Stress-Test Console (USP #08) */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                USP #08 · Stress-Test Console ("Judges Control the Chaos")
              </div>
              <button
                className="btn-primary"
                onClick={() => sendCommand({ command: 'start_replay', scenario: 'scripted_chaos' })}
              >
                ▶ Run Scripted Chaos Scenario
              </button>
            </div>

            <div className="control-group">
              <div className="control-label">
                <span>NLOS Wall-Bounce Bias (Signals Lie)</span>
                <span className="val">+{nlosBias.toFixed(2)} m</span>
              </div>
              <input
                type="range"
                min="0"
                max="6.0"
                step="0.1"
                value={nlosBias}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  setNlosBias(v);
                  sendCommand({ command: 'set_nlos_bias', bias: v });
                }}
              />
            </div>

            <div className="control-group">
              <div className="control-label">
                <span>Gaussian Multipath Noise (σ)</span>
                <span className="val">{noiseSigma.toFixed(2)} m</span>
              </div>
              <input
                type="range"
                min="0"
                max="1.5"
                step="0.05"
                value={noiseSigma}
                onChange={(e) => {
                  const v = parseFloat(e.target.value);
                  setNoiseSigma(v);
                  sendCommand({ command: 'set_noise', sigma: v });
                }}
              />
            </div>

            <div className="control-label" style={{ marginTop: '4px' }}>
              <span>Click Anchors to Toggle Online / Trigger Blind Spots:</span>
            </div>
            <div className="anchor-grid">
              {(layout.anchors || []).map((a) => {
                const sr = (frame?.scored_readings || []).find(
                  (r) => r.anchor_id === a.id
                );
                const isOnline = sr ? sr.online : a.online;
                const isLiar = sr ? sr.is_liar : false;
                return (
                  <button
                    key={a.id}
                    className={`anchor-btn ${
                      !isOnline ? 'offline' : isLiar ? 'liar' : 'online'
                    }`}
                    onClick={() => {
                      const nextState = !isOnline;
                      setLayout((prev) => ({
                        ...prev,
                        anchors: prev.anchors.map((item) =>
                          item.id === a.id ? { ...item, online: nextState } : item
                        ),
                      }));
                      sendCommand({
                        command: 'toggle_anchor',
                        anchor_id: a.id,
                        online: nextState,
                      });
                    }}
                  >
                    <strong>{a.id}</strong>
                    <span style={{ fontSize: '0.7rem' }}>
                      {!isOnline ? 'OFFLINE' : isLiar ? 'LIAR (NLOS)' : 'ONLINE'}
                    </span>
                  </button>
                );
              })}
            </div>

            <div className="btn-row" style={{ marginTop: '4px' }}>
              <button
                className="btn-danger"
                onClick={() =>
                  sendCommand({ command: 'trigger_blackout', blackout: true })
                }
              >
                ⚡ Trigger Full Blackout (Ghost Mode)
              </button>
              <button
                onClick={() =>
                  sendCommand({ command: 'trigger_blackout', blackout: false })
                }
              >
                ↺ Restore All Anchors
              </button>
              <button
                onClick={() => {
                  const cleared = { ...layout, walls: [] };
                  setLayout(cleared);
                  sendCommand({ command: 'load_layout', layout: cleared });
                }}
              >
                Clear Walls
              </button>
              <button
                onClick={() => {
                  setLayout(DEFAULT_LAYOUT);
                  setNoiseSigma(0.15);
                  setNlosBias(2.5);
                  gtTrailRef.current = [];
                  estTrailRef.current = [];
                  naiveTrailRef.current = [];
                  sendCommand({ command: 'load_layout', layout: DEFAULT_LAYOUT });
                  sendCommand({ command: 'reset', seed: 42 });
                }}
              >
                Reset Default
              </button>
            </div>
          </div>

          {/* Explainable Anchor Recovery Adviser (USP #02) */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                USP #02 · Explainable Anchor Recovery Adviser
              </div>
            </div>
            {frame && frame.recovery_advice && frame.recovery_advice.length > 0 ? (
              frame.recovery_advice.slice(0, 3).map((adv) => (
                <div key={adv.anchor_id} className="advice-item">
                  <div>
                    <strong>
                      #{adv.priority} Priority · {adv.anchor_id} ({adv.status.toUpperCase()})
                    </strong>
                    <div style={{ color: '#cbd5e1', marginTop: '2px' }}>
                      {adv.reason}
                    </div>
                  </div>
                  {adv.status === 'offline' && (
                    <button
                      className="btn-primary"
                      onClick={() =>
                        sendCommand({
                          command: 'toggle_anchor',
                          anchor_id: adv.anchor_id,
                          online: true,
                        })
                      }
                    >
                      Restore {adv.anchor_id}
                    </button>
                  )}
                </div>
              ))
            ) : (
              <div style={{ fontSize: '0.8rem', color: '#34d399' }}>
                ✔ All anchors online and geometrically consistent. Optimal GDOP achieved.
              </div>
            )}
          </div>

          {/* Competing Trackers Benchmark & Failure Replay (USP #09) */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                USP #09 · Live Competing Trackers Benchmark ( vs. Ground Truth )
              </div>
            </div>
            <table className="diag-table">
              <thead>
                <tr>
                  <th>Tracker Pipeline</th>
                  <th>Position (x, y)</th>
                  <th>Instant Error</th>
                  <th>Rolling RMSE</th>
                  <th>Behavior</th>
                </tr>
              </thead>
              <tbody>
                {(frame?.competing_trackers || []).map((ct) => (
                  <tr key={ct.name}>
                    <td>
                      <strong>{ct.name}</strong>
                    </td>
                    <td>
                      {typeof ct.x === 'number'
                        ? `(${ct.x.toFixed(2)}, ${ct.y.toFixed(2)})`
                        : '—'}
                    </td>
                    <td>
                      {typeof ct.error_m === 'number'
                        ? `${ct.error_m.toFixed(2)} m`
                        : '—'}
                    </td>
                    <td>
                      <strong
                        style={{
                          color: ct.name.includes('GhostTrack')
                            ? '#34d399'
                            : ct.name.includes('Naive')
                            ? '#f87171'
                            : '#38bdf8',
                        }}
                      >
                        {ct.rmse_m.toFixed(2)} m
                      </strong>
                    </td>
                    <td>{ct.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Auditable Decision Log (USP #11) */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                USP #11 · Auditable Decisions Log (States, Weights &amp; Recovery)
              </div>
            </div>
            <div className="audit-feed">
              {(frame?.audit_events || []).length === 0 ? (
                <div className="audit-row info">
                  [SYSTEM] Pipeline initialized. Monitoring geometric consistency...
                </div>
              ) : (
                frame.audit_events.map((ev, idx) => (
                  <div key={idx} className={`audit-row ${ev.severity}`}>
                    <strong>[{ev.category}]</strong> {ev.message}
                  </div>
                ))
              )}
            </div>
          </div>
        </div>
      </div>

      {/* 12 USPs Live Verification Matrix */}
      <section className="card">
        <div className="card-header">
          <div className="card-title">
            All 12 PPT USPs Active in This Build
          </div>
        </div>
        <div className="usp-grid">
          <div className="usp-card">
            <strong>01 Ambiguity-Aware Tracking</strong>
            Preserves bimodal plausible regions when 2 anchors intersect.
          </div>
          <div className="usp-card">
            <strong>02 Explainable Anchor Recovery</strong>
            Ranks offline/NLOS anchors by GDOP &amp; symmetry-breaking gain.
          </div>
          <div className="usp-card">
            <strong>03 Measured Reliability</strong>
            Live calibrated "{confPct}% within 2m" probability &amp; unseen-room RMSE.
          </div>
          <div className="usp-card">
            <strong>04 Sensor Diagnostics</strong>
            Downweights liars with reliability scores &amp; highlights red links.
          </div>
          <div className="usp-card">
            <strong>05 Ghost Mode, Visible Limits</strong>
            LSTM + IMU dead-reckoning; transitions to Lost after prolonged outage.
          </div>
          <div className="usp-card">
            <strong>06 Controlled Recovery</strong>
            Verifies returning anchor fixes across frames before trusting.
          </div>
          <div className="usp-card">
            <strong>07 Map-Aware Tracking</strong>
            Floor plan walls block impossible moves &amp; mask the belief heatmap.
          </div>
          <div className="usp-card">
            <strong>08 Judge-Controlled Testing</strong>
            Live wall drawing, NLOS/noise sliders, anchor toggles &amp; blackout.
          </div>
          <div className="usp-card">
            <strong>09 Failure Replay</strong>
            Simultaneous comparison across Naive, Geo Filter &amp; GhostTrack.
          </div>
          <div className="usp-card">
            <strong>10 Freshness-Aware Tracking</strong>
            Displays measurement age ({ageMs.toFixed(0)} ms) &amp; penalizes stale packets.
          </div>
          <div className="usp-card">
            <strong>11 Auditable Decisions</strong>
            Real-time audit log of state changes, weights, and adviser ranks.
          </div>
          <div className="usp-card">
            <strong>12 Local, No-Cloud Operation</strong>
            100% local execution (Mosquitto/Embedded MQTT + FastAPI + React).
          </div>
        </div>
      </section>
    </div>
  );
}
