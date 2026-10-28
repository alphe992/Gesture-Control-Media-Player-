"""
main.py
-------
Gesture-Controlled Media Player
Touchless media playback and volume control using OpenCV, MediaPipe, and Pillow.

Usage:
    python main.py                  # Standard webcam mode
    python main.py --camera 1       # Use external camera
    python main.py --demo           # Synthetic test/demo mode without physical camera
    python main.py --help           # Show CLI options
"""

import argparse
import sys
import time
import cv2
import numpy as np

from gesture_engine import GestureEngine
from hand_tracker import HandDetection, HandTracker
from hud_renderer import HUDRenderer
from media_controller import MediaController


def parse_args():
    parser = argparse.ArgumentParser(
        description="Gesture-Control Media Player (OpenCV + MediaPipe + Pillow)"
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--width", type=int, default=1280, help="Camera frame width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Camera frame height (default: 720)")
    parser.add_argument("--confidence", type=float, default=0.5, help="Hand detection confidence threshold (0.1 - 1.0)")
    parser.add_argument("--model", type=str, default=None, help="Path to gesture_recognizer.task model")
    parser.add_argument("--demo", action="store_true", help="Run in synthetic demo mode (no camera required)")
    parser.add_argument("--headless", action="store_true", help="Run without opening GUI windows (for testing)")
    return parser.parse_args()


def generate_synthetic_frame(w: int, h: int, t: float):
    """Generates synthetic animated frame for testing and automated demo verification."""
    # Dark modern canvas
    frame = np.full((h, w, 3), (25, 20, 18), dtype=np.uint8)

    # Animated circular grid
    for r in range(50, 400, 60):
        cv2.circle(frame, (w // 2, h // 2), r, (35, 30, 25), 1, cv2.LINE_AA)

    # Simulated hand moving in a smooth pattern
    center_x = int(w * 0.5 + np.sin(t * 1.5) * 180)
    center_y = int(h * 0.55 + np.cos(t * 1.0) * 80)

    # Simulated pinch distance oscillation (cycles between open and pinch)
    pinch_open = (np.sin(t * 2.0) + 1.0) * 0.5  # 0.0 to 1.0

    # Build 21 simulated hand landmarks
    wrist = (center_x, center_y + 120)
    thumb_tip = (center_x - int(30 + pinch_open * 70), center_y - 20)
    index_tip = (center_x + int(30 + pinch_open * 70), center_y - 20)

    # Construct synthetic detection
    px = [wrist] * 21
    px[0] = wrist
    px[4] = thumb_tip
    px[8] = index_tip
    px[9] = (center_x, center_y)  # middle knuckle

    gesture = "None"
    if pinch_open < 0.2:
        gesture = "None"
    elif np.sin(t * 0.5) > 0.6:
        gesture = "Victory"
    elif np.sin(t * 0.5) < -0.6:
        gesture = "Open_Palm"

    simulated_det = HandDetection(
        handedness="Right",
        gesture_name=gesture,
        gesture_score=0.92,
        landmarks_px=px,
        landmarks_norm=[(x / w, y / h, 0.0) for x, y in px]
    )

    return frame, [simulated_det]


def main():
    args = parse_args()

    print("=" * 60)
    print("      GESTURE-CONTROL MEDIA PLAYER (OPENCV + PILLOW)      ")
    print("=" * 60)
    print("Initializing components...")

    # 1. Media and Audio Controller
    media_ctrl = MediaController()

    # 2. Gesture Engine
    engine = GestureEngine(media_controller=media_ctrl, debounce_sec=1.0, hold_frames=3)

    # 3. HUD Visualizer
    hud = HUDRenderer(width=args.width, height=args.height)

    # 4. Hand Tracker
    hand_tracker = None
    if not args.demo:
        try:
            hand_tracker = HandTracker(
                model_path=args.model,
                num_hands=2,
                min_confidence=args.confidence
            )
            print("[System] MediaPipe Hand Tracker loaded successfully.")
        except Exception as e:
            print(f"[System] Warning: Could not initialize HandTracker ({e}). Switching to Demo Mode.")
            args.demo = True

    # 5. Video Capture
    cap = None
    if not args.demo:
        print(f"[System] Opening Camera Index {args.camera}...")
        cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

        if not cap.isOpened():
            print(f"[System] Warning: Could not open camera {args.camera}. Switching to Demo Mode.")
            args.demo = True
            if cap:
                cap.release()
                cap = None

    window_name = "Gesture-Control Media Player | OpenCV + Pillow"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, args.width, args.height)

    # Ensure UTF-8 output on Windows console
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    print("\n[Controls & Shortcuts]")
    print("  - Open Palm         : Play / Pause toggle")
    print("  - Closed Fist       : Mute / Unmute system audio")
    print("  - Victory / Peace   : Skip to Next Track >>")
    print("  - Pointing Up       : Previous Track <<")
    print("  - Thumb + Index     : Continuous Precision Volume Slider")
    print("  - Thumb Up / Down   : Volume Step (+/- 8%)")
    print("  - [H] Key           : Toggle on-screen Help HUD")
    print("  - [Q] or [ESC] Key  : Exit Application\n")
    print("[System] Application running. Press 'q' in the camera window to exit.\n")

    prev_time = time.time()
    fps_history = []
    start_time = time.time()

    try:
        while True:
            current_time = time.time()
            dt = current_time - prev_time
            prev_time = current_time
            if dt > 0:
                fps_history.append(1.0 / dt)
                if len(fps_history) > 15:
                    fps_history.pop(0)
            avg_fps = sum(fps_history) / max(1, len(fps_history))

            if args.demo:
                # Generate synthetic test frame
                t_elapsed = current_time - start_time
                frame, detections = generate_synthetic_frame(args.width, args.height, t_elapsed)
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[System] Failed to grab webcam frame. Retrying...")
                    time.sleep(0.05)
                    continue

                # Mirror frame horizontally for intuitive interaction
                frame = cv2.flip(frame, 1)

                # Track hands and classify gestures
                detections = hand_tracker.process_frame(frame)

            # Process gestures and update media playback
            engine_state = engine.process(detections)

            # Render HUD overlay using OpenCV & Pillow
            composited_frame = hud.render(frame, detections, engine_state, avg_fps)

            # Display frame
            if not args.headless:
                cv2.imshow(window_name, composited_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), ord('Q'), 27):  # 'q' or ESC
                    print("[System] Exit key pressed. Terminating...")
                    break
                elif key in (ord('h'), ord('H')):
                    hud.toggle_help()
                elif key == 32:  # Spacebar manual toggle
                    media_ctrl.play_pause()
                elif key in (ord('m'), ord('M')):
                    media_ctrl.toggle_mute()
            else:
                # Headless test iteration
                if current_time - start_time > 3.0:
                    print("[System] Headless test ran successfully for 3 seconds.")
                    break

    except KeyboardInterrupt:
        print("\n[System] Interrupted by user.")
    finally:
        if cap is not None:
            cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[System] Cleanup complete. Shutdown successful.")


if __name__ == "__main__":
    main()
