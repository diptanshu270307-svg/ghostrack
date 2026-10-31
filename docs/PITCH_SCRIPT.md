# GhostTrack — 3-Minute Presentation Pitch & Demo Guide

**Team:** Kajukatli (Diptanshu Jamar)  
**Project:** InSight AI · GhostTrack (Resilient Indoor Positioning)  

---

## 🎙️ Spoken Script (Word-for-Word, ~3 Minutes)

### [0:00 - 0:40] The Hook & The Problem
> "Hello everyone, I’m Diptanshu Jamar presenting **GhostTrack** by Team Kajukatli.
>
> Indoor positioning systems—using BLE or Ultra-Wideband in warehouses, hospitals, and smart factories—fail constantly in the real world. Why? Because **radio waves bounce off concrete walls and metal obstacles** (NLOS/multipath). Distance readings appear longer than they actually are, making standard trilateration jump erratically across rooms.
>
> Worse, when a person or robot walks into a blind spot or loses signal, standard tracking systems freeze, glitch, or hallucinate impossible movements. We set out to solve this without requiring expensive new infrastructure."

---

### [0:40 - 1:30] The Core Innovation & Architecture
*(Point to the Dashboard / Architecture flow)*
> "GhostTrack is an **ambiguity-aware, explainable positioning engine** that fuses geometric consensus with neural dead-reckoning.
>
> Instead of blindly trusting corrupted sensor readings, GhostTrack runs a 4-stage pipeline:
> 1. **Geometric Contradiction Filter:** Intersects anchor radii and mathematically identifies which anchors are 'lying' due to wall reflections, downweighting them continuously.
> 2. **Hybrid CNN-LSTM:** A 36-feature neural model trained to estimate non-line-of-sight bias and maintain trajectory momentum.
> 3. **Physics & Map Awareness:** Interior walls physically restrict impossible movements during dead-reckoning."

---

### [1:30 - 2:20] The 4 Explicit States (The "Ghost" in GhostTrack)
*(Point to the 4 State badges: Tracked, Ambiguous, Ghost, Lost)*
> "The true strength of GhostTrack is our **4-State Reliability Lifecycle**:
> * When at least 3 clean anchors are visible, the system is **`Tracked`** with pinpoint accuracy.
> * When only 2 anchors are visible, conventional systems guess a random 50/50 point. GhostTrack enters **`Ambiguous`** and renders a bimodal probability heatmap, showing both plausible physical locations.
> * During a signal blackout, the system enters **`Ghost Mode`**. It fuses our LSTM momentum with device IMU acceleration to dead-reckon their path, while visually expanding an uncertainty radius so operators know how confident the estimate is.
> * If a blackout lasts longer than 5 seconds, it refuses to fabricate data and transitions cleanly to **`Lost`** until recovery verification passes."

---

### [2:20 - 3:00] Live Chaos Console, Multi-Target, & Conclusion
*(Point to the Interactive Stress-Test Console & Multi-Target switcher)*
> "On the dashboard, we also feature a **Live Chaos Console** where we can toggle anchors offline, inject radio noise, and draw custom walls in real-time. We can track multiple targets simultaneously—Person Alpha, Robot Bravo, and Staff Charlie—each with independent state machines and audit logs.
>
> The entire stack is containerized with FastAPI, PyTorch CPU, WebSockets, and React, capable of running 100% locally or in the cloud.
>
> **In short: GhostTrack doesn’t just guess where you are—it knows when signals are lying, refuses to invent fake paths, and gives operators transparent, explainable reliability.** Thank you!"

---

## 🎯 Live Demo Action Steps (30-Second Walkthrough)

1. **Show Multi-Person Tracking:** Switch focus between **Person Alpha**, **Robot Bravo**, and **Staff Charlie** on the top tabs to show independent state tracking.
2. **Inject NLOS / Chaos:** Open the **Stress-Test Console** on the right, increase the NLOS Bias slider or toggle an Anchor off. Show how the geometric filter identifies the offending anchor in red.
3. **Trigger Ghost Mode:** Toggle all anchors off (Blackout). Show the target enter **Ghost** state with an expanding uncertainty ring ($\sigma$).
4. **Trigger Recovery:** Restore the anchors. Point to the **Audit Log** showing the consecutive verification frames restoring the state back to **Tracked**.

---

## 💡 Quick Q&A Defense

| Question | Answer |
|---|---|
| **"What happens during NLOS wall reflections?"** | Radio reflections always make measured distance *greater* than true Euclidean distance. Our leave-one-out geometric filter detects that radii don't intersect consistently and penalizes the offending anchor. |
| **"Why not just use an end-to-end deep learning model?"** | Pure ML models overfit to specific room layouts and hallucinate dangerous paths when sensors disconnect. We use hybrid AI: geometry enforces physical constraints, while the CNN-LSTM handles momentum and residual bias. |
| **"How do you handle recovery after a blackout?"** | We use **Controlled Recovery**: when signals return, they must pass consecutive geometric consensus checks before the system switches back to `Tracked`, preventing sudden snap-back jumps. |
