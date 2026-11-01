# Disappear: leave meetings in style

Touch your index and middle finger to your forehead (Goku's Instant Transmission)
and you vanish from the video call, leaving the empty room. Do it again to come back.

How it works:

1. The script reads your real webcam.
2. MediaPipe finds your face and hands, and cuts your silhouette out of each frame.
3. As soon as the gesture has been held for 0.2 s, you flicker out with white
   wavy streaks and an outline. Add `--countdown 1` for a countdown first.
4. Where you were, it paints the empty room. You never have to leave your
   seat for that: an AI inpainting model (LaMa) erases you from one picture
   and rebuilds the room behind you.
5. The result goes to **OBS Virtual Camera**. You pick that as your camera in
   Teams, Zoom or Meet.

## 1. Install Python 3.12 and OBS Studio (once)

Already done on this PC: Python 3.12.10 and OBS Studio are installed. On
another PC, open PowerShell and run:

```bash
winget install -e --id Python.Python.3.12
```

```bash
winget install -e --id OBSProject.OBSStudio
```

You don't need to open OBS. Installing it adds the "OBS Virtual Camera" driver
that this script sends video to. Use Python 3.12 rather than 3.14: MediaPipe has
known crashes on 3.14.

## 2. Start it

Double-click `start.bat`. The first run takes a few minutes: it sets up a
Python environment in `%LOCALAPPDATA%\disappear\venv` and installs mediapipe,
opencv, numpy, pyvirtualcam, onnxruntime and mss there. It also downloads the
models into `models\`: three small MediaPipe models (8 MB) and the LaMa
inpainting model (198 MB). The environment lives at that short path because pip
fails on Windows when its files end up more than 260 characters deep, which
happens if the project sits in a deep folder.

A preview window opens. Just sit there:

1. **"Loading the AI model (about 30 s after start)..."**: on this laptop the
   model takes about 30–45 seconds to load.
2. **"Building the room behind you..."**: it takes one picture of you, erases
   you from it and rebuilds the background (about 7 s).
3. The message goes away, and the gesture now works.

Until then the gesture does nothing, rather than hiding you over a blurry guess.
If you move the laptop or the lighting changes a lot, the room picture is
rebuilt the same way, and if you were hidden you reappear meanwhile. Press `B`
to rebuild it by hand.

The rebuilt background is made up by the AI. Plain backgrounds like a curtain
or a wall look real. Detailed things right behind you may come out a little
off. If you ever do leave the camera's view for 2 seconds, with no face, hand
or movement in the picture, a real photo of the empty room replaces it.

## 3. Pick the camera in your meeting app

- **Teams:** Settings → Devices → Camera → *OBS Virtual Camera*
- **Zoom:** Settings → Video → Camera → *OBS Virtual Camera*
- **Google Meet:** Settings → Video → *OBS Virtual Camera*

Your real webcam is busy feeding the script, so the meeting app must use the
virtual one.

## 4. Use it

| Do this | What happens |
|---|---|
| Index + middle finger on your forehead, ring + pinky folded, hold ~0.2 s | `Disappearing!`, and you vanish right away |
| Same gesture while hidden | `Reappearing!`, and you're back |
| `Space` in the preview window | Same toggle, no gesture needed |
| `B` | Rebuild the room behind you now with the AI model (a few seconds) |

Keys only work while the preview window is in front, so click it first.
Upper or lower case doesn't matter.
| `D` | Debug view: green box = forehead zone, dot = fingertips detected |
| `Q` / `Esc` | Quit |

The preview shows exactly what the call sees, plus a small text overlay that
only you see.

On this laptop (HP HD Camera, i7-1165G7) it runs at about 30 fps while you're
visible and about 20 fps while you're hidden. Hiding is heavier because your
cut-out is recomputed on every frame.

## Camera numbers on this PC

- `0`: HP HD Camera, the real webcam. This is the default, so no option is needed.
- `1`: OBS Virtual Camera, the camera this script feeds.
- The HP IR Camera (Windows Hello) doesn't show up, so it can't be picked by mistake.

## Options

Pass them to `start.bat`, e.g. `start.bat --camera 1 --countdown 3`.

| Option | Default | Meaning |
|---|---|---|
| `--camera N` | 0 | Which webcam to read |
| `--list-cameras` | | Saves `camera_0.jpg`, `camera_1.jpg`, ... so you can see which index is which |
| `--width/--height/--fps` | 1280 / 720 / 30 | Webcam mode to request |
| `--countdown S` | 0 | Seconds between the gesture and the effect (0 = instant) |
| `--transition S` | 0.4 | Length of the vanish / appear effect |
| `--hold S` | 0.2 | How long the gesture must be held |
| `--calibrate S` | 0 | Also ask for a real step-out photo this many seconds after start (0 = no, the AI rebuilds the room) |
| `--loose-gesture` | off | Any index fingertip on the forehead counts |
| `--no-virtual` | off | Preview only, for testing without OBS |
| `--virtual-backend` | auto | `obs` or `unitycapture` |

## Troubleshooting

- **The preview shows the OBS logo or a black image.** Index 0 is the OBS
  virtual camera itself, not your webcam. Run `start.bat --list-cameras`, look
  at the saved pictures, then start with `start.bat --camera 1` (or whichever
  index shows your real webcam).
- **"Virtual camera not available".** OBS isn't installed, or OBS is open with
  *Start Virtual Camera* switched on. Close OBS and run again. If it still fails,
  open OBS once, click *Start Virtual Camera*, then *Stop*, close OBS and retry.
- **"Could not open webcam".** Another app has the real webcam. Close the
  meeting app's camera preview, or switch it to OBS Virtual Camera first.
- **The gesture isn't detected.** Press `D`. The box must turn green when your
  fingertips are inside it. Keep your face visible and your ring and pinky
  fingers folded, or start with `--loose-gesture`.
- **Ghost edges or a smeared patch where you were.** Step out of the camera's
  view for a few seconds; the room is recaptured automatically. Don't move the camera after capturing. Even lighting helps the
  cut-out. Brightness changes are corrected automatically: webcams keep
  adjusting their exposure, so the saved room is matched to the live picture.
- **Low frame rate.** The webcam is opened through Windows Media Foundation.
  Through the older DirectShow interface this webcam only reaches about 10 fps
  at 720p. A very dark room can also slow the webcam itself, so turn on a light.
  Closing heavy apps helps while you're hidden.

## Uninstall

Delete this folder and `%LOCALAPPDATA%\disappear`. Python and OBS Studio can
be removed from Windows Settings → Apps.
