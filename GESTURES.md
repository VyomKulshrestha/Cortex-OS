# 🤚 Heliox OS — Gesture System v3 (30+ Gestures)

Heliox OS includes a state-of-the-art webcam-based hand gesture recognition engine powered by [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands.html). It supports **30+ gestures** including both static poses and real-time motion tracking.

## Architecture

```
Webcam Feed
    │
    ▼
MediaPipe Hands (21 landmarks per hand)
    │
    ├──► classifyGesture() ──► Static Pose Detection
    │                              (finger extension patterns,
    │                               tip distances, orientation)
    │
    ├──► detectCircularMotion() ──► Circular Gesture Detection
    │                                  (cross product analysis on
    │                                   12-point index finger buffer)
    │
    └──► detectPushPull() ──► Z-Axis Depth Detection
                                (8-point wrist Z-history)
    │
    ▼
Frame Stabilizer (5 consecutive identical frames required)
    │
    ▼
Cooldown Gate (1200ms between triggers)
    │
    ▼
executeGestureAction() ──► session.sendCommand / session.confirm
    │
    ▼
UI Feedback (emoji badge, particle burst, gesture history)
```

## Debouncing & Stability

The gesture engine uses a **3-layer anti-jitter system** to prevent false triggers:

| Layer | Mechanism | Purpose |
|-------|-----------|---------|
| **Frame Stabilizer** | Gesture must be detected for 5 consecutive frames | Eliminates transition noise |
| **Cooldown Gate** | 1200ms lockout after each trigger | Prevents double-fires |
| **Buffer Clearing** | Motion buffers reset after circular/push gestures | Prevents re-triggering |

---

## Static Pose Gestures (21)

These are recognized by analyzing which fingers are extended, curled, or touching.

| # | Gesture | Emoji | How To Do It | System Action |
|---|---------|-------|--------------|---------------|
| 1 | **Open Palm** | ✋ | All fingers and thumb extended | Cancel / Stop current task |
| 2 | **Thumbs Up** | 👍 | Only thumb up, fist closed | Confirm AI plan |
| 3 | **Thumbs Down** | 👎 | Only thumb down, fist closed | Deny / Reject AI plan |
| 4 | **Peace Sign** | ✌️ | Index + middle finger up | Toggle voice mode |
| 5 | **Fist** | 👊 | All fingers curled tight | Execute last command |
| 6 | **Point Up** | 👆 | Only index finger raised | Scroll up |
| 7 | **Rock Sign** | 🤟 | Index + pinky extended | Show system info |
| 8 | **OK Sign** | 👌 | Thumb tip touches index tip, others up | Acknowledge / Accept |
| 9 | **Call Me** | 🤙 | Thumb + pinky extended, others curled | Open settings panel |
| 10 | **Finger Gun** | 🔫 | Thumb + index extended horizontally | Take a screenshot |
| 11 | **Pinch** | 🤏 | Thumb + index tips touching, others curled | Grab / Select element |
| 12 | **Middle Finger** | 🖕 | Only middle finger extended | **Emergency Stop** all tasks |
| 13 | **Pinky Up** | 🌸 | Only pinky finger extended | Fancy mode trigger |
| 14 | **Vulcan Salute** | 🖖 | All 4 fingers up with gap between middle and ring | Run full system diagnostics |
| 15 | **Crossed Fingers** | 🤞 | Index + middle up, tips touching | Surprise me random action |
| 16 | **Snap Ready** | 🫰 | Thumb touching middle finger, others curled | Quick Launch most-used app |
| 17 | **Devil Horns** | 🤘 | Index + pinky spread wide, no thumb | Open music player / Play music |
| 18 | **Palm Down** | 🫳 | All fingers extended, tips pointing down | **Mute** volume to 0 |
| 19 | **Palm Up** | 🫴 | All fingers extended, tips pointing up high | **Unmute** volume to 50% |
| 20 | **Three Up** | 🔆 | Index + middle + ring up (no pinky, no thumb) | Brightness up 20% |
| 21 | **Four Up** | 🔅 | All 4 fingers up (no thumb) | Brightness down 20% |

---

## Motion-Based Gestures (10)

These detect **hand movement over time** using position history buffers.

| # | Gesture | Emoji | How To Do It | System Action |
|---|---------|-------|--------------|---------------|
| 22 | **Swipe Left** | 👈 | Open palm, move hand left quickly | Previous tab |
| 23 | **Swipe Right** | 👉 | Open palm, move hand right quickly | Next tab |
| 24 | **Swipe Up** | ⬆️ | Open palm, move hand up quickly | Scroll up fast |
| 25 | **Swipe Down** | ⬇️ | Open palm, move hand down quickly | Scroll down fast |
| 26 | **Circular Clockwise** | 🔄 | Draw a circle with index finger (CW) | **Volume Up** (+15%) |
| 27 | **Circular Counter-CW** | 🔃 | Draw a circle with index finger (CCW) | **Volume Down** (-15%) |
| 28 | **Palm Push** | 🫸 | Open palm, push hand toward screen | **Confirm** AI action |
| 29 | **Palm Pull** | 🫷 | Open palm, pull hand away from screen | **Cancel** AI action |
| 30 | **Two-Finger Swipe Left** | ⏪ | Peace sign, swipe left | Switch workspace left |
| 31 | **Two-Finger Swipe Right** | ⏩ | Peace sign, swipe right | Switch workspace right |

---

## Detection Algorithms

### Static Pose Detection

Each of the 21 hand landmarks from MediaPipe is analyzed:
- **Finger extension**: `tip.y < pip.y` means the finger is extended (up)
- **Thumb extension**: `thumb_tip.x < thumb_ip.x` (lateral movement)
- **Tip distance**: `dist(thumb_tip, index_tip)` for OK/Pinch gestures
- **Orientation**: Comparing average fingertip Y vs wrist Y for palm up/down

Gestures are checked **most-specific-first** to prevent ambiguity:
1. Orientation-dependent (Palm Down/Up)
2. Multi-touch (OK, Pinch, Snap Ready, Crossed)
3. Single-finger (Middle, Pinky)
4. Compound (Devil Horns, Finger Gun, Call Me)
5. Simple (Fist, Palm, Peace, Point)

### Circular Motion Detection

Uses a **12-point index finger position buffer**:

```
1. Compute centroid: cx = mean(x), cy = mean(y)
2. Compute radius of each point from centroid
3. Check circularity: stddev(radii) < 50% of avgRadius
4. Determine direction via cross product sum:
   crossSum > 0 → Clockwise  → Volume Up
   crossSum < 0 → Counter-CW → Volume Down
```

### Palm Push/Pull (Z-Axis Depth)

Uses an **8-point wrist history with Z-coordinate** from MediaPipe:

```
1. Compare Z-depth: newest.z - oldest.z
2. Requires elapsed time: 100ms < elapsed < 600ms
3. All 4 fingers must be extended (open palm pose)
4. dz < -0.06 → Push forward (confirm)
5. dz > +0.06 → Pull back (cancel)
```

---

## Configuration

| Setting | Value | Description |
|---------|-------|-------------|
| `REQUIRED_FRAMES` | 5 | Consecutive frames needed to confirm gesture |
| `GESTURE_COOLDOWN_MS` | 1200 | Milliseconds between gesture triggers |
| `MOTION_BUFFER_SIZE` | 20 | Max frames in wrist/index history buffers |
| `MAX_TRAIL_LENGTH` | 60 | Max points in air-drawing trail |
| `modelComplexity` | 0 | MediaPipe model (0=lite, 1=full) |
| `minDetectionConfidence` | 0.6 | Minimum hand detection confidence |
| `minTrackingConfidence` | 0.5 | Minimum hand tracking confidence |

---

## Adding New Gestures

To add a new gesture:

1. **Define the finger pattern** in `classifyGesture()` inside `GestureControl.svelte` — place it in priority order
2. **Add an emoji** to the `GESTURE_EMOJIS` map
3. **Add an action** in `executeGestureAction()` — call `session.sendCommand()` or `session.addSystemMessage()`
4. **Handle in `App.svelte`** if it needs to trigger UI navigation (tab switching, etc.)

---

## Source Files

| File | Role |
|------|------|
| `tauri-app/ui/src/lib/components/GestureControl.svelte` | Core gesture engine |
| `tauri-app/ui/src/App.svelte` | Gesture to UI navigation handler |
| `tauri-app/src-tauri/tauri.conf.json` | CSP allowing MediaPipe CDN |
