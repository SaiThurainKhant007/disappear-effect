"""Record the main screen to an MP4 in your Videos folder (no sound).

Stops on Ctrl+C, after --max-minutes, or as soon as a file named STOP_RECORDING
appears next to this script.
Usage: python record_screen.py [--fps 15] [--width 1280] [--max-minutes 10]
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2
import mss
import numpy as np

STOP_FILE = Path(__file__).with_name("STOP_RECORDING")


def main() -> None:
    p = argparse.ArgumentParser(description="Record the main screen to an MP4.")
    p.add_argument("--fps", type=float, default=15.0)
    p.add_argument("--width", type=int, default=1280, help="width of the saved video (height follows)")
    p.add_argument("--max-minutes", type=float, default=10.0)
    p.add_argument("--out-dir", default=str(Path.home() / "Videos"))
    args = p.parse_args()

    STOP_FILE.unlink(missing_ok=True)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"screen-{datetime.now():%Y-%m-%d_%H-%M-%S}.mp4"

    with getattr(mss, "MSS", mss.mss)() as sct:  # mss 10 renamed mss.mss to mss.MSS
        monitor = sct.monitors[1]
        w = args.width
        h = round(monitor["height"] * w / monitor["width"]) // 2 * 2
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter.fourcc(*"mp4v"), args.fps, (w, h))
        if not writer.isOpened():
            raise SystemExit(f"Could not create {path}")
        print(f"[Recorder] Recording {monitor['width']}x{monitor['height']} -> {path}", flush=True)
        started = time.perf_counter()
        frames = 0
        try:
            while not STOP_FILE.exists() and time.perf_counter() - started < args.max_minutes * 60:
                # Write frames on a fixed clock so the video plays at real speed even if capture lags.
                due = int((time.perf_counter() - started) * args.fps) + 1
                shot = cv2.resize(cv2.cvtColor(np.asarray(sct.grab(monitor)), cv2.COLOR_BGRA2BGR), (w, h),
                                  interpolation=cv2.INTER_AREA)
                while frames < due:
                    writer.write(shot)
                    frames += 1
                time.sleep(max(0.0, started + frames / args.fps - time.perf_counter()))
        except KeyboardInterrupt:
            pass
        finally:
            writer.release()
            STOP_FILE.unlink(missing_ok=True)
    print(f"[Recorder] Saved {frames / args.fps:.0f} s of video to {path}", flush=True)


if __name__ == "__main__":
    main()
