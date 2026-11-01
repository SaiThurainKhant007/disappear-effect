"""Leave meetings in style: touch two fingers to your forehead and vanish from
the call (Instant Transmission), do it again to come back.

Reads the real webcam, spots the gesture with MediaPipe, hides you by painting
the empty room over your silhouette, and sends the result to the OBS Virtual
Camera so Teams / Zoom / Meet can pick it like any other webcam.
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision

try:
    from mediapipe.tasks.python.vision.core.image import Image as MpImage, ImageFormat
except ImportError:  # older mediapipe releases expose these at the top level
    MpImage, ImageFormat = mp.Image, mp.ImageFormat

MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "blaze_face_short_range.tflite": "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite",
    "selfie_segmenter.tflite": "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    # LaMa inpainting (198 MB): rebuilds the room behind you, so you never have to leave the picture.
    "lama_fp32.onnx": "https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx",
}

# The models work on small images anyway, so face and hand detection run on a copy
# this wide, and the person cut-out (and everything built on it) on an even smaller one.
DETECT_WIDTH = 640
MASK_WIDTH = 320

# How long the room capture waits for you to leave the picture before it makes do.
CALIBRATE_WAIT = 30.0
# Average brightness change between checks (0-255) below which the picture counts as
# still, i.e. nobody is moving in it; webcam noise alone stays around 1-2.
STILL_MOTION = 4.0
# Average difference (0-255) between the live room and the brightness-matched room photo,
# around you, above which the camera is taken to have moved to a different view.
SCENE_CHANGED = 20.0

# Hand landmark indices (https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker)
WRIST = 0
INDEX_PIP, INDEX_TIP = 6, 8
MIDDLE_PIP, MIDDLE_TIP = 10, 12
RING_PIP, RING_TIP = 14, 16
PINKY_PIP, PINKY_TIP = 18, 20


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vanish from video calls with the Instant Transmission gesture.")
    p.add_argument("--camera", type=int, default=0, help="webcam index (default 0, see --list-cameras)")
    p.add_argument("--list-cameras", action="store_true", help="save a snapshot from each webcam index and exit")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--countdown", type=float, default=0.0, help="seconds between the gesture and the effect")
    p.add_argument("--transition", type=float, default=0.4, help="length of the vanish / appear effect in seconds")
    p.add_argument("--hold", type=float, default=0.2, help="seconds the gesture must be held to count")
    p.add_argument("--calibrate", type=float, default=0.0,
                   help="also wait this many seconds for you to step out for a real photo of the empty room "
                        "(0 = no: the room behind you is rebuilt by the AI model)")
    p.add_argument("--loose-gesture", action="store_true",
                   help="an index fingertip on the forehead is enough, whatever the other fingers do")
    p.add_argument("--no-virtual", action="store_true", help="preview window only, no virtual camera")
    p.add_argument("--virtual-backend", default=None, help="pyvirtualcam backend: obs or unitycapture")
    p.add_argument("--virtual-device", default=None, help="pyvirtualcam device name")
    p.add_argument("--no-preview", action="store_true", help="don't open the preview window")
    p.add_argument("--models-dir", default=str(Path(__file__).with_name("models")))
    return p.parse_args()


def download_models(models_dir: Path) -> dict[str, Path]:
    models_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in MODEL_URLS.items():
        path = models_dir / name
        if not path.exists():
            print(f"[Models] Downloading {name} ...")
            tmp = path.with_name(name + ".part")
            urllib.request.urlretrieve(url, tmp)
            tmp.replace(path)
        paths[name] = path
    return paths


def _capture(index: int) -> cv2.VideoCapture:
    if sys.platform != "win32":
        return cv2.VideoCapture(index, cv2.CAP_ANY)
    # Media Foundation first: through DirectShow many laptop webcams only offer
    # uncompressed 720p, which runs at about 10 fps instead of 30.
    cap = cv2.VideoCapture(index, cv2.CAP_MSMF)
    if cap.isOpened():
        return cap
    cap.release()
    return cv2.VideoCapture(index, cv2.CAP_DSHOW)


def list_cameras(out_dir: Path) -> None:
    for index in range(6):
        cap = _capture(index)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            path = out_dir / f"camera_{index}.jpg"
            cv2.imwrite(str(path), frame)
            print(f"[Webcam] index {index}: {frame.shape[1]}x{frame.shape[0]} -> {path}")
        else:
            print(f"[Webcam] index {index}: nothing")


def open_camera(args: argparse.Namespace) -> tuple[cv2.VideoCapture, np.ndarray]:
    cap = _capture(args.camera)
    if not cap.isOpened():
        sys.exit(f"[Webcam] Could not open webcam {args.camera}. Is another app (Teams, Zoom, OBS) using it?")
    if cap.getBackendName() == "DSHOW":
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc(*"MJPG"))
        except (AttributeError, cv2.error):
            pass
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    ok, frame = cap.read()
    if not ok:
        sys.exit(f"[Webcam] Webcam {args.camera} opened but returned no image.")
    return cap, frame


def open_virtual_camera(args: argparse.Namespace, width: int, height: int):
    try:
        import pyvirtualcam
        camera = pyvirtualcam.Camera(
            width=width, height=height, fps=args.fps,
            fmt=pyvirtualcam.PixelFormat.BGR,
            backend=args.virtual_backend, device=args.virtual_device,
        )
    except (ImportError, RuntimeError, OSError, ValueError) as error:
        print("[Webcam] Virtual camera not available. Install OBS Studio (it ships the 'OBS Virtual Camera' "
              "driver) and make sure OBS's own 'Start Virtual Camera' is NOT running. "
              f"Details: {error}")
        return None
    print(f"[Webcam] Sending video to virtual camera '{camera.device}' ({width}x{height} @ {args.fps} fps)")
    return camera


class Detectors:
    def __init__(self, models: dict[str, Path]):
        def base(name: str) -> mp_tasks.BaseOptions:
            return mp_tasks.BaseOptions(model_asset_path=str(models[name]))

        video = vision.RunningMode.VIDEO
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=base("hand_landmarker.task"), running_mode=video, num_hands=2))
        self.faces = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
            base_options=base("blaze_face_short_range.tflite"), running_mode=video,
            min_detection_confidence=0.5))
        self.segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=base("selfie_segmenter.tflite"), running_mode=video,
            output_confidence_masks=True, output_category_mask=False))

    def close(self) -> None:
        for task in (self.hands, self.faces, self.segmenter):
            task.close()


@dataclass
class Forehead:
    x0: float
    x1: float
    y0: float
    y1: float
    seen_at: float

    def contains(self, point: np.ndarray) -> bool:
        return self.x0 <= point[0] <= self.x1 and self.y0 <= point[1] <= self.y1


def forehead_zone(detection, w: int, h: int, now: float) -> Forehead:
    box = detection.bounding_box
    if detection.keypoints and len(detection.keypoints) >= 2:
        eye_y = (detection.keypoints[0].y + detection.keypoints[1].y) / 2 * h
    else:
        eye_y = box.origin_y + 0.4 * box.height
    # From a bit above the hairline down to the eyes, a little wider than the face.
    return Forehead(
        x0=box.origin_x - 0.2 * box.width, x1=box.origin_x + 1.2 * box.width,
        y0=box.origin_y - 0.7 * box.height, y1=eye_y + 0.05 * box.height,
        seen_at=now,
    )


def _point(landmarks, i: int, w: int, h: int) -> np.ndarray:
    return np.array((landmarks[i].x * w, landmarks[i].y * h))


def _extended(landmarks, tip: int, pip: int, w: int, h: int) -> bool:
    wrist = _point(landmarks, WRIST, w, h)
    return (np.linalg.norm(_point(landmarks, tip, w, h) - wrist)
            > 1.15 * np.linalg.norm(_point(landmarks, pip, w, h) - wrist))


def find_forehead_touch(hands, forehead: Forehead, w: int, h: int, loose: bool):
    """Index + middle finger out, ring + pinky folded, fingertips on the forehead."""
    for lm in hands:
        if loose:
            tip = _point(lm, INDEX_TIP, w, h)
        else:
            if not (_extended(lm, INDEX_TIP, INDEX_PIP, w, h) and _extended(lm, MIDDLE_TIP, MIDDLE_PIP, w, h)):
                continue
            if _extended(lm, RING_TIP, RING_PIP, w, h) or _extended(lm, PINKY_TIP, PINKY_PIP, w, h):
                continue
            tip = (_point(lm, INDEX_TIP, w, h) + _point(lm, MIDDLE_TIP, w, h)) / 2
        if forehead.contains(tip):
            return tip
    return None


class GestureWatcher:
    """Runs face + hand detection on the newest frame in a background thread, so the slow
    hand model never holds up the video (MediaPipe releases the GIL while it works)."""

    def __init__(self, detectors: Detectors, dw: int, dh: int, loose: bool):
        self.detectors, self.dw, self.dh, self.loose = detectors, dw, dh, loose
        self.touch = None
        self.forehead = None
        self.faces = []
        self.hands = []  # one (21, 2) array of normalized landmark positions per hand
        self.checked_at = None  # capture time of the frame behind the latest result
        self._job = None
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def submit(self, small: np.ndarray, now: float) -> None:
        with self._lock:
            self._job = (small, now)
        self._wake.set()

    def latest(self):
        with self._lock:
            return self.touch, self.forehead, self.faces, self.hands, self.checked_at

    def stop(self) -> None:
        self._running = False
        self._wake.set()
        self._thread.join(timeout=2)

    def _run(self) -> None:
        ts = 0
        started = 0.0
        while self._running:
            self._wake.wait(0.5)
            # About 12 checks a second is plenty and leaves the CPU to the video.
            time.sleep(max(0.0, started + 0.08 - time.perf_counter()))
            self._wake.clear()
            with self._lock:
                job, self._job = self._job, None
            if job is None:
                continue
            started = time.perf_counter()
            small, now = job
            ts = max(ts + 1, int(now * 1000))  # VIDEO mode needs strictly increasing timestamps
            image = MpImage(image_format=ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            faces = self.detectors.faces.detect_for_video(image, ts).detections
            forehead = self.forehead
            if faces:
                best = max(faces, key=lambda d: d.bounding_box.width * d.bounding_box.height)
                forehead = forehead_zone(best, self.dw, self.dh, now)
            elif forehead is not None and now - forehead.seen_at > 1.0:
                forehead = None  # a hand over the face can hide it briefly, so keep the last one for a second
            hands = self.detectors.hands.detect_for_video(image, ts).hand_landmarks
            touch = find_forehead_touch(hands, forehead, self.dw, self.dh, self.loose) if forehead else None
            points = [np.array([(p.x, p.y) for p in lm], np.float32) for lm in hands]
            with self._lock:
                self.faces, self.forehead, self.touch, self.checked_at = faces, forehead, touch, now
                self.hands = points


def person_mask(result, shape: tuple[int, int]) -> np.ndarray:
    masks = result.confidence_masks
    mask = np.squeeze(masks[0].numpy_view()).astype(np.float32)
    if len(masks) > 1:  # several categories: the first one is the background
        mask = 1.0 - mask
    if mask.shape != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]))
    return mask


class MaskTracker:
    """Smooths the person mask over time and grows it so no edge of you leaks through.
    Works at mask size (`small`, `raw`) and keeps a full-size copy (`mask`)."""

    def __init__(self, w: int, h: int, mw: int, mh: int):
        self.size = (w, h)
        # Grow by r, then feather by no more than r, so your own outline stays fully covered.
        r = max(2, mw // 40)
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        self.blur = 2 * r + 1
        self.prev = None
        self.raw = np.zeros((mh, mw), np.float32)
        self.small = np.zeros((mh, mw), np.float32)
        self.mask = np.zeros((h, w), np.float32)

    def update(self, raw: np.ndarray, hands=()) -> None:
        if hands:
            # The cut-out model often misses a hand held away from the body, so every
            # tracked hand is added, plus a band for the (untracked) forearm behind it.
            raw = raw.copy()
            mh, mw = raw.shape
            for pts in hands:
                p = pts * np.float32((mw, mh))
                size = max(float(np.linalg.norm(p[9] - p[0])), 1.0)  # wrist to middle knuckle
                hull = cv2.convexHull(p).astype(np.int32)
                cv2.fillConvexPoly(raw, hull, 1.0)
                cv2.polylines(raw, [hull], True, 1.0, max(2, int(size * 0.6)))
                elbow = p[0] + (p[0] - p[9]) * 2.5
                cv2.line(raw, tuple(int(v) for v in p[0]), tuple(int(v) for v in elbow), 1.0, max(2, int(size * 1.1)))
        self.raw = raw
        m = np.clip((raw - 0.25) / 0.35, 0.0, 1.0)
        if self.prev is not None:  # hiding a bit too much is harmless, leaking a hand is not
            m = np.maximum(m, self.prev * 0.6)
        self.prev = m
        m = cv2.dilate(m, self.kernel)
        self.small = cv2.GaussianBlur(m, (self.blur, self.blur), 0)
        self.mask = cv2.resize(self.small, self.size, interpolation=cv2.INTER_LINEAR)


def fill_holes(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Guess unknown pixels (weight 0) from the nearest known ones (push-pull).
    `img` must already be multiplied by `weight`."""
    pyramid = []
    while min(weight.shape) > 4:
        pyramid.append((img, weight))
        size = (max(1, weight.shape[1] // 2), max(1, weight.shape[0] // 2))
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        weight = cv2.resize(weight, size, interpolation=cv2.INTER_AREA)
    filled = img / np.maximum(weight, 1e-6)[..., None]
    for img_l, weight_l in reversed(pyramid):
        up = cv2.resize(filled, (weight_l.shape[1], weight_l.shape[0]), interpolation=cv2.INTER_LINEAR)
        own = img_l / np.maximum(weight_l, 1e-6)[..., None]
        a = np.clip(weight_l * 4.0, 0.0, 1.0)[..., None]
        filled = own * a + up * (1.0 - a)
    return filled


class Background:
    """The empty room. Photographed during calibration and then kept as is: the
    webcam keeps changing its exposure, so instead of re-learning the room it is
    brightness-matched to the live picture (`gain`) every time it's painted."""

    def __init__(self, shape: tuple[int, ...], small_size: tuple[int, int]):
        self.plate = np.zeros(shape, np.float32)  # stays 0 wherever the room hasn't been seen
        self.known = np.zeros(shape[:2], bool)
        # Really photographed, as opposed to made up by the AI model behind you. Only these
        # parts are compared with the live picture: the made-up part never matches the real
        # room you uncover by moving, which looked like a moved camera or changed lighting.
        self.seen = np.zeros(shape[:2], bool)
        self.small_size = small_size
        self.complete = False
        self.gain = np.ones(3, np.float32)
        self.gain_map = None  # per-area gain at the small size
        self._image = None
        self._small = None

    def learn(self, frame: np.ndarray, free: np.ndarray, rate: float | None = None) -> None:
        """Takes the room from `free` pixels; without a rate only fills in pixels never seen yet."""
        first = free & ~self.known
        if rate is not None:
            f = frame.astype(np.float32)
            self.plate += (f - self.plate) * (free * np.float32(rate))[..., None]
            self.plate[first] = f[first]
        elif first.any():
            self.plate[first] = frame[first] / self.gain
        else:
            return
        self.known |= free
        self.seen |= free
        self.complete = bool(self.known.all())
        self._image = self._small = None

    def set_rebuilt(self, room: np.ndarray, made_up: np.ndarray) -> None:
        """Use a room picture where the `made_up` area was filled in by the AI model."""
        self.reset()
        self.learn(room, np.ones(self.known.shape, bool), rate=1.0)
        self.seen = ~made_up
        self._small = None

    def coverage(self) -> float:
        return float(self.known.mean())

    def _small_view(self) -> None:
        if self._small is None:
            known_small = cv2.resize(self.known.astype(np.float32), self.small_size, interpolation=cv2.INTER_AREA)
            seen_small = cv2.resize(self.seen.astype(np.float32), self.small_size, interpolation=cv2.INTER_AREA)
            self._small = cv2.resize(self.plate, self.small_size, interpolation=cv2.INTER_AREA)
            self._small_known = seen_small > 0.99
            self._small_weight = known_small

    def image(self) -> np.ndarray:
        """The full-size room, with never-seen parts guessed. Only built when it gets painted."""
        if self._image is None:
            full = self.plate
            if not self.complete:
                self._small_view()
                guess = cv2.resize(fill_holes(self._small, self._small_weight),
                                   (self.plate.shape[1], self.plate.shape[0]), interpolation=cv2.INTER_LINEAR)
                full = np.where(self.known[..., None], self.plate, guess)
            self._image = np.clip(full, 0, 255).astype(np.uint8)
        return self._image

    def reset(self) -> None:
        """Forget the room photo (the camera now sees a different room)."""
        self.plate[:] = 0
        self.known[:] = False
        self.seen = np.zeros(self.known.shape, bool)
        self.complete = False
        self.gain[:] = 1.0
        self.gain_map = None
        self._image = self._small = None

    def update_gain(self, small_frame: np.ndarray, hide_small: np.ndarray) -> float:
        """Compare the room with the live picture wherever you aren't. Returns how different
        they still are after brightness matching (0-255), or 0 when there's nothing to compare."""
        self._small_view()
        sel = ((hide_small < 0.05) & self._small_known).astype(np.uint8)
        if cv2.countNonZero(sel) < 500:
            return 0.0
        live = np.array(cv2.mean(small_frame, mask=sel)[:3])
        room = np.array(cv2.mean(self._small, mask=sel)[:3])
        target = np.clip(live / np.maximum(room, 1.0), 0.5, 2.0).astype(np.float32)
        self.gain = 0.7 * self.gain + 0.3 * target
        mismatch = float(np.mean(cv2.mean(cv2.absdiff(small_frame.astype(np.float32), self._small * self.gain),
                                          mask=sel)[:3]))

        # Lamps and shadows change brightness differently across the room, so also keep a
        # smooth per-area gain, taken from the live room right around you.
        # Worked out at a quarter of the mask size: the map is very smooth anyway, and the
        # wide blurs would cost ~15 ms a frame at full mask size.
        mw, mh = self.small_size
        tiny = (max(1, mw // 4), max(1, mh // 4))
        free = cv2.resize(sel.astype(np.float32), tiny, interpolation=cv2.INTER_AREA)[..., None]
        live_t = cv2.resize(small_frame, tiny, interpolation=cv2.INTER_AREA).astype(np.float32)
        room_t = cv2.resize(self._small, tiny, interpolation=cv2.INTER_AREA)
        sigma = tiny[0] / 16
        live_near = cv2.GaussianBlur(live_t * free, (0, 0), sigma)
        room_near = cv2.GaussianBlur(room_t * free, (0, 0), sigma)
        trust = np.clip(cv2.GaussianBlur(free[..., 0], (0, 0), sigma) * 5.0, 0.0, 1.0)[..., None]
        local = np.clip(live_near / np.maximum(room_near, 1.0), 0.5, 2.0)
        target_map = (local * trust + self.gain * (1.0 - trust)).astype(np.float32)
        target_map = cv2.resize(target_map, self.small_size, interpolation=cv2.INTER_LINEAR)
        self.gain_map = target_map if self.gain_map is None else 0.7 * self.gain_map + 0.3 * target_map
        return mismatch


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = min(1.0, max(0.0, (x - e0) / (e1 - e0)))
    return t * t * (3.0 - 2.0 * t)


class Controller:
    """Turns the gesture into countdown -> effect -> hidden / visible."""

    def __init__(self, countdown: float, transition: float, hold: float):
        self.countdown = countdown
        self.transition = max(transition, 1e-3)
        self.hold = hold
        self.hidden = False
        self.pending_at = None
        self.effect_start = None
        self.announced = None
        self.gesture_since = None
        self.released_since = None
        self.armed = True

    def busy(self) -> bool:
        return self.pending_at is not None or self.effect_start is not None

    def update_gesture(self, active: bool, now: float) -> None:
        if not active:
            self.gesture_since = None
            if self.released_since is None:
                self.released_since = now
            if now - self.released_since >= 0.5:  # hand down for a moment before it can fire again
                self.armed = True
            return
        self.released_since = None
        if self.gesture_since is None:
            self.gesture_since = now
        if self.armed and not self.busy() and now - self.gesture_since >= self.hold:
            self.armed = False
            self.trigger(now)

    def force_visible(self) -> None:
        self.hidden = False
        self.pending_at = self.effect_start = None

    def trigger(self, now: float) -> None:
        if not self.busy():
            self.pending_at = now + self.countdown
            self.announced = None

    def tick(self, now: float) -> tuple[float, float]:
        """Returns (how visible you are 0..1, effect strength 0..1)."""
        verb = "Reappearing" if self.hidden else "Disappearing"
        if self.pending_at is not None:
            left = self.pending_at - now
            if left > 0:
                n = math.ceil(left)
                if n != self.announced:
                    print(f"{verb} in {n}...")
                    self.announced = n
                return (0.0 if self.hidden else 1.0), 0.0
            self.pending_at = None
            self.effect_start = now
            print(f"{verb}!")
        if self.effect_start is not None:
            p = (now - self.effect_start) / self.transition
            if p >= 1.0:
                self.effect_start = None
                self.hidden = not self.hidden
            else:
                v = smoothstep(0.2, 0.9, p) if self.hidden else 1.0 - smoothstep(0.1, 0.8, p)
                if random.random() < 0.3:  # flicker
                    v = min(1.0, max(0.0, v + random.uniform(-0.4, 0.4)))
                return v, math.sin(math.pi * p)
        return (0.0 if self.hidden else 1.0), 0.0


class Streaks:
    """Thin wavy vertical lines, the Instant Transmission look."""

    def __init__(self, w: int, h: int):
        rng = random.Random()
        self.ys = np.arange(0, h + 8, 8, dtype=np.float32)
        self.lines = [
            (rng.uniform(0, w), rng.uniform(2, 0.012 * w), rng.uniform(0.004, 0.02),
             rng.uniform(0, 2 * math.pi), rng.uniform(2, 6), rng.uniform(0.4, 1.0))
            for _ in range(max(12, w // 45))
        ]

    def draw(self, canvas: np.ndarray, t: float) -> None:
        for x, amp, freq, phase, speed, brightness in self.lines:
            xs = x + amp * np.sin(self.ys * freq + phase + t * speed)
            points = np.stack([xs, self.ys], axis=1).astype(np.int32)
            cv2.polylines(canvas, [points], False, int(255 * brightness), 1, cv2.LINE_AA)


def paint_room(frame: np.ndarray, background: Background, tracker: MaskTracker, amount: float) -> np.ndarray:
    """Paints the brightness-matched empty room over you; only touches the box around you."""
    small = tracker.small
    rows = np.flatnonzero(small.max(axis=1) > 0.004)
    if rows.size == 0:
        return frame
    cols = np.flatnonzero(small.max(axis=0) > 0.004)
    h, w = frame.shape[:2]
    sy, sx = h / small.shape[0], w / small.shape[1]
    y0, y1 = int(rows[0] * sy), min(h, math.ceil((rows[-1] + 1) * sy))
    x0, x1 = int(cols[0] * sx), min(w, math.ceil((cols[-1] + 1) * sx))
    if background.gain_map is None:
        room = cv2.transform(background.image()[y0:y1, x0:x1], np.diag(background.gain).astype(np.float32))
    else:
        gain = cv2.resize(background.gain_map[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], (x1 - x0, y1 - y0),
                          interpolation=cv2.INTER_LINEAR)
        room = cv2.convertScaleAbs(background.image()[y0:y1, x0:x1].astype(np.float32) * gain)
    a = np.ascontiguousarray(tracker.mask[y0:y1, x0:x1] * np.float32(amount))
    out = frame.copy()
    out[y0:y1, x0:x1] = cv2.blendLinear(frame[y0:y1, x0:x1], room, 1.0 - a, a)
    return out


def erase_person(session, frame: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rebuild the room behind you with the LaMa inpainting model: returns the frame with
    the `mask` area (you) filled in with made-up but matching background."""
    h, w = frame.shape[:2]
    ys, xs = np.nonzero(mask)
    if ys.size == 0:
        return frame.copy()
    # A square crop around you, so the model's fixed 512x512 input keeps as much detail as it can.
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    pad_y, pad_x = (y1 - y0) // 7 + 16, (x1 - x0) // 7 + 16
    y0, y1, x0, x1 = max(0, y0 - pad_y), min(h, y1 + pad_y), max(0, x0 - pad_x), min(w, x1 + pad_x)
    side = max(y1 - y0, x1 - x0)

    def widen(a0: int, a1: int, limit: int) -> tuple[int, int]:
        a0 = max(0, min(a0 - (side - (a1 - a0)) // 2, limit - side))
        return a0, min(limit, a0 + side)

    y0, y1 = widen(y0, y1, h)
    x0, x1 = widen(x0, x1, w)
    crop, crop_mask = frame[y0:y1, x0:x1], mask[y0:y1, x0:x1]
    image = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB), (512, 512), interpolation=cv2.INTER_AREA)
    holes = cv2.resize(crop_mask.astype(np.uint8), (512, 512), interpolation=cv2.INTER_NEAREST)
    names = [i.name for i in session.get_inputs()]
    filled = session.run(None, {names[0]: image.transpose(2, 0, 1)[None].astype(np.float32) / 255.0,
                                names[1]: holes[None, None].astype(np.float32)})[0][0].transpose(1, 2, 0)
    filled = cv2.cvtColor(np.clip(filled, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    filled = cv2.resize(filled, (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
    plate = frame.copy()
    plate[y0:y1, x0:x1][crop_mask] = filled[crop_mask]
    return plate


def _room_worker(model_path: str, jobs, results) -> None:
    """Runs in its own process: as a thread in the video's process, one rebuild took ~50 s
    instead of ~6 s, because the video loop kept it waiting for Python's interpreter lock."""
    try:
        import onnxruntime
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 4  # leave some of the CPU to the video
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        session = onnxruntime.InferenceSession(model_path, options, providers=["CPUExecutionProvider"])
    except Exception as error:
        results.put(("error", str(error)))
        return
    results.put(("ready", None))
    while (job := jobs.get()) is not None:
        started = time.perf_counter()
        try:
            results.put(("room", (erase_person(session, *job), time.perf_counter() - started)))
        except Exception as error:
            results.put(("error", str(error)))


class RoomBuilder:
    """Rebuilds the room behind you with the AI model in a separate process, so the video keeps
    going meanwhile. Loading the model takes ~30 s on a laptop, so that starts right away."""

    def __init__(self, model_path: Path):
        import multiprocessing
        self._jobs, self._results = multiprocessing.Queue(), multiprocessing.Queue()
        self._process = multiprocessing.Process(target=_room_worker, daemon=True,
                                                args=(str(model_path), self._jobs, self._results))
        self._process.start()
        self.ready = False   # model loaded
        self.failed = False  # model or library missing: only the step-out photo works then
        self._busy = False

    def busy(self) -> bool:
        return self._busy

    def start(self, frame: np.ndarray, mask: np.ndarray) -> None:
        if not self._busy and not self.failed:
            self._jobs.put((frame, mask))
            self._busy = True

    def take(self):
        """The rebuilt room once it's ready (only once), else None."""
        while True:
            try:
                kind, value = self._results.get_nowait()
            except Exception:  # nothing new
                return None
            if kind == "ready":
                self.ready = True
            elif kind == "error":
                print(f"[Background] Couldn't rebuild the room with the AI model: {value}")
                self._busy = False
                self.failed = not self.ready
            else:
                self._busy = False
                print(f"[Background] (the AI rebuild took {value[1]:.1f} s)")
                return value[0]

    def close(self) -> None:
        self._jobs.put(None)
        self._process.join(timeout=3)
        if self._process.is_alive():
            self._process.terminate()


def add_glow(img: np.ndarray, streaks: Streaks, person_small: np.ndarray, strength: float, t: float) -> np.ndarray:
    h, w = img.shape[:2]
    canvas = np.zeros((h, w), np.uint8)
    streaks.draw(canvas, t)
    person = cv2.resize(person_small, (w, h), interpolation=cv2.INTER_LINEAR)
    contours = cv2.findContours((person > 0.5).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(canvas, contours, -1, 255, max(2, w // 400), cv2.LINE_AA)
    glow = cv2.add(cv2.convertScaleAbs(canvas, alpha=strength), cv2.convertScaleAbs(person, alpha=0.3 * 255 * strength))
    glow = cv2.cvtColor(glow, cv2.COLOR_GRAY2BGR)
    # Lighten towards white: img + (255 - img) * glow / 255
    return cv2.add(img, cv2.multiply(cv2.bitwise_not(img), glow, scale=1 / 255))


def draw_hud(img, ctl, touch, forehead, coverage, fps, note, debug, scale):
    view = img.copy()
    state = "HIDDEN" if ctl.hidden else "VISIBLE"
    if ctl.busy():
        state += " -> " + ("VISIBLE" if ctl.hidden else "HIDDEN")
    lines = [f"{state}   {fps:4.1f} fps   room seen: {coverage * 100:3.0f}%",
             "gesture: YES" if touch is not None else "gesture: -",
             "Q quit  SPACE toggle  B recapture room  D debug"]
    if note:
        lines.insert(0, note)
    for i, text in enumerate(lines):
        org = (12, 30 + 28 * i)
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    if debug and forehead is not None:
        cv2.rectangle(view, (int(forehead.x0 * scale), int(forehead.y0 * scale)),
                      (int(forehead.x1 * scale), int(forehead.y1 * scale)),
                      (0, 255, 0) if touch is not None else (0, 200, 255), 2)
    if debug and touch is not None:
        cv2.circle(view, (int(touch[0] * scale), int(touch[1] * scale)), 8, (0, 255, 0), -1)
    if view.shape[1] > 960:
        view = cv2.resize(view, (960, int(view.shape[0] * 960 / view.shape[1])))
    return view


def main() -> None:
    args = parse_args()
    if args.list_cameras:
        list_cameras(Path(__file__).parent)
        return

    models = download_models(Path(args.models_dir))
    cap, frame = open_camera(args)
    h, w = frame.shape[:2]
    dw = min(w, DETECT_WIDTH)
    dh = round(h * dw / w)
    mw = min(w, MASK_WIDTH)
    mh = round(h * mw / w)
    print(f"[Webcam] Webcam {args.camera}: {w}x{h}")

    detectors = Detectors(models)
    camera = None if args.no_virtual else open_virtual_camera(args, w, h)
    if camera is None and args.no_preview:
        sys.exit("[Webcam] No virtual camera and no preview: nothing to show.")

    background = Background(frame.shape, (mw, mh))
    tracker = MaskTracker(w, h, mw, mh)
    ctl = Controller(args.countdown, args.transition, args.hold)
    streaks = Streaks(w, h)
    watcher = GestureWatcher(detectors, dw, dh, args.loose_gesture)
    last_checked = None
    debug = False
    calib_at = time.monotonic() + args.calibrate if args.calibrate > 0 else None
    calib_frames = calib_occupied = 0
    calib_waiting = lighting_drifted = False
    empty_since, last_capture = None, -math.inf
    prev_gray = None
    changed_frames, last_not_ready = 0, -math.inf
    builder = RoomBuilder(models["lama_fp32.onnx"])
    last_build, build_mask = -math.inf, np.zeros((h, w), bool)
    if calib_at is not None:
        print(f"[Background] Step out of the frame: capturing the empty room in {args.calibrate:.0f} s.")
    window = "Disappear - preview"
    ts = 0
    frame_index = 0
    fps, fps_count, fps_since = 0.0, 0, time.monotonic()

    try:
        while True:
            now = time.monotonic()
            ts = max(ts + 1, int(now * 1000))  # VIDEO mode needs strictly increasing timestamps
            small = frame if dw == w else cv2.resize(frame, (dw, dh), interpolation=cv2.INTER_AREA)
            watcher.submit(small, now)
            touch, forehead, faces, hands, checked_at = watcher.latest()
            rebuilt = builder.take()
            if rebuilt is not None and calib_at is None:
                background.set_rebuilt(rebuilt, build_mask)
                last_capture, lighting_drifted = now, False
                print("[Background] Rebuilt the room behind you: you can disappear now.")
            # Disappearing on a guessed room leaves a blurry blob, so it waits for a real room photo.
            room_ready = background.complete and calib_at is None
            if checked_at is not None and checked_at != last_checked:
                last_checked = checked_at
                wants = touch is not None
                if wants and not ctl.hidden and not room_ready:
                    if now - last_not_ready >= 3.0:
                        print("[Background] Can't disappear yet: the room behind you isn't rebuilt yet "
                              "(the AI model needs ~30 s after start, then a few seconds).")
                        last_not_ready = now
                    wants = False
                ctl.update_gesture(wants, checked_at)
            visibility, effect = ctl.tick(now)

            calibrating = calib_at is not None and now >= calib_at
            # While visible the cut-out only keeps the room's brightness match fresh.
            if calibrating or visibility < 1.0 or effect > 0.0 or frame_index % 3 == 0:
                tiny = cv2.resize(small, (mw, mh), interpolation=cv2.INTER_AREA) if mw != dw else small
                tiny_image = MpImage(image_format=ImageFormat.SRGB, data=cv2.cvtColor(tiny, cv2.COLOR_BGR2RGB))
                raw = person_mask(detectors.segmenter.segment_for_video(tiny_image, ts), (mh, mw))
                tracker.update(raw, hands)
                # Is anyone there? The cut-out model can't tell: chairs and pillows right behind
                # you read as "person" even in an empty room. So: any face (of anyone, at any
                # size, or they end up baked into the room photo), a hand, or movement.
                gray = cv2.cvtColor(tiny, cv2.COLOR_BGR2GRAY).astype(np.float32)
                motion = 0.0 if prev_gray is None else float(np.abs(gray - prev_gray).mean())
                prev_gray = gray
                someone_there = bool(faces) or bool(hands) or motion > STILL_MOTION
                if calibrating:
                    occupied = someone_there
                    if occupied and calib_frames == 0 and now - calib_at < CALIBRATE_WAIT:
                        if not calib_waiting:  # don't start until you're out of the picture
                            print("[Background] Waiting for you to step out of the frame...")
                            calib_waiting = True
                    else:
                        calib_occupied += occupied
                        free = (tracker.mask < 0.05) if occupied else np.ones((h, w), bool)
                        background.learn(frame, free, rate=0.3)
                        calib_frames += 1
                    if calib_frames >= 20:
                        print(f"[Background] Captured. Room seen without you: {background.coverage() * 100:.0f}%")
                        if calib_occupied:
                            print("[Background] You still seemed to be in the picture; the room behind you is "
                                  "guessed. Press B and step out of the frame to capture it properly.")
                        calib_at, calib_frames, calib_occupied, calib_waiting = None, 0, 0, False
                        background.gain[:] = 1.0  # the room was just photographed at today's exposure
                        background.gain_map = None
                        last_capture = now
                else:
                    # The room only ever comes from a deliberate capture of an empty picture:
                    # learning it bit by bit around you also learnt halos, shadows and passers-by.
                    mismatch = background.update_gain(tiny, tracker.small)
                    changed_frames = changed_frames + 1 if mismatch > SCENE_CHANGED else 0
                    if changed_frames >= 8:
                        print("[Background] The picture no longer matches the room photo (camera moved?). "
                              "Making a new one.")
                        background.reset()
                        changed_frames = 0
                        if ctl.hidden or ctl.busy():
                            ctl.force_visible()
                            print("Reappearing: there's no room photo to hide you with.")
                    drifted = bool(np.any((background.gain < 0.67) | (background.gain > 1.5)))
                    if drifted and not lighting_drifted:
                        print("[Background] The lighting changed a lot since the room photo. Making a new one.")
                    lighting_drifted = drifted
                    # No room photo (or a stale one) while you're in the picture: let the AI model
                    # rebuild the room behind you, so you never have to get up and leave.
                    you_in_mask = (tracker.mask > 0.5).mean() > 0.02
                    if (someone_there and you_in_mask and (not background.complete or lighting_drifted)
                            and builder.ready and not builder.busy() and now - last_build >= 10.0):
                        print("[Background] Rebuilding the room behind you with the AI model (a few seconds)...")
                        build_mask = tracker.mask > 0.02
                        builder.start(frame, build_mask)
                        last_build = now
                    # Whenever you're out of the picture for a moment and the room photo is
                    # incomplete or out of date, take a new one, so no key press is needed.
                    outdated = not background.complete or lighting_drifted or now - last_capture > 60
                    if not someone_there:
                        empty_since = empty_since or now
                        if outdated and now - empty_since >= 2.0:
                            print("[Background] You're out of the picture: capturing the empty room.")
                            calib_at, calib_frames, calib_occupied, calib_waiting = now, 0, 0, False
                            empty_since = None
                    else:
                        empty_since = None

            out = frame
            if visibility < 1.0:
                out = paint_room(frame, background, tracker, 1.0 - visibility)
            if effect > 0.0:
                out = add_glow(out, streaks, tracker.raw, effect, now)

            if camera is not None:
                if out.shape[:2] != (camera.height, camera.width):
                    out = cv2.resize(out, (camera.width, camera.height))
                camera.send(np.ascontiguousarray(out))

            fps_count += 1
            if now - fps_since >= 1.0:
                fps, fps_count, fps_since = fps_count / (now - fps_since), 0, now

            if not args.no_preview:
                note = None
                if not background.complete and not builder.ready and not builder.failed:
                    note = "Loading the AI model (about 30 s after start)..."
                elif builder.busy():
                    note = "Building the room behind you (a few seconds)..."
                elif not background.complete:
                    note = "No room photo yet"
                if calib_at is not None:
                    left = calib_at - now
                    if left > 0:
                        note = f"Step out of the frame: capturing room in {math.ceil(left)}"
                    elif calib_frames == 0 and calib_waiting:
                        note = "Waiting for an empty room: step out of the frame"
                    else:
                        note = "Capturing the empty room..."
                cv2.imshow(window, draw_hud(out, ctl, touch, forehead, background.coverage(), fps, note,
                                            debug, w / dw))
                key = cv2.waitKey(1) & 0xFF
                if ord("A") <= key <= ord("Z"):  # Caps Lock or Shift: B works like b
                    key += ord("a") - ord("A")
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    if ctl.hidden or room_ready:
                        ctl.trigger(now)
                    else:
                        print("[Background] Can't disappear yet: the room behind you is still being rebuilt.")
                elif key == ord("b") and not ctl.hidden:
                    build_mask = tracker.mask > 0.02
                    builder.start(frame, build_mask)
                    last_build = now
                    print("[Background] Rebuilding the room behind you with the AI model (a few seconds)...")
                elif key == ord("d"):
                    debug = not debug

            ok, frame = cap.read()
            if not ok:
                print("[Webcam] Lost the webcam feed.")
                break
            frame_index += 1
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if camera is not None:
            camera.close()
        watcher.stop()
        builder.close()
        detectors.close()
        cv2.destroyAllWindows()
        print("[Webcam] Stopped.")


if __name__ == "__main__":
    main()
