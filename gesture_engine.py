"""
gesture_engine.py
-----------------
Translates raw hand landmarks and MediaPipe classifications into debounced,
reliable media commands and smooth pinch-to-volume adjustments.
"""

import math
import time
from typing import Dict, List, Optional, Tuple
import numpy as np

from hand_tracker import HandDetection
from media_controller import MediaController


class GestureEngine:
    """Interprets gestures, performs geometric calculations, debouncing, and dispatches actions."""

    def __init__(self, media_controller: MediaController, debounce_sec: float = 1.0, hold_frames: int = 3):
        self.media = media_controller
        self.debounce_sec = debounce_sec
        self.hold_frames = hold_frames

        # State tracking
        self.last_triggered_time: float = 0.0
        self.consecutive_gesture: Optional[str] = None
        self.consecutive_count: int = 0

        # Pinch smoothing
        self.smoothed_volume: float = self.media.get_volume()
        self.smoothing_alpha: float = 0.25  # EMA weight for new frame
        self.is_pinching: bool = False
        self.pinch_points: Optional[Tuple[Tuple[int, int], Tuple[int, int], Tuple[int, int]]] = None

        # Cooldown percentage for HUD (0.0 = ready, 1.0 = just triggered)
        self.cooldown_progress: float = 0.0

    def process(self, detections: List[HandDetection]) -> Dict:
        """
        Processes detected hands for the current frame and triggers media actions.
        :param detections: List of HandDetection from HandTracker
        :return: State summary dictionary for HUD rendering
        """
        now = time.time()
        elapsed = now - self.last_triggered_time
        self.cooldown_progress = max(0.0, min(1.0, 1.0 - (elapsed / self.debounce_sec)))
        is_ready = elapsed >= self.debounce_sec

        self.is_pinching = False
        self.pinch_points = None

        if not detections:
            self.consecutive_gesture = None
            self.consecutive_count = 0
            return self._build_state_summary()

        # Prioritize right hand for volume slider, left hand for discrete actions (or whichever is present)
        primary_hand = detections[0]
        for det in detections:
            # If multiple hands, check if any is doing a pinch
            if self._check_pinch_control(det):
                break

        # Process discrete gesture commands if cooldown elapsed
        self._evaluate_discrete_gesture(primary_hand, is_ready, now)

        return self._build_state_summary()

    def _check_pinch_control(self, det: HandDetection) -> bool:
        """
        Calculates scale-invariant pinch distance between thumb and index tip.
        If pinching, smoothly adjusts system volume.
        """
        px = det.landmarks_px
        thumb = px[4]
        index = px[8]
        wrist = px[0]
        middle_mcp = px[9]

        # Calculate hand scale (wrist to middle knuckle) for camera-distance invariance
        hand_scale = math.hypot(wrist[0] - middle_mcp[0], wrist[1] - middle_mcp[1])
        if hand_scale < 20:
            return False

        # Euclidean distance between thumb and index tip
        pinch_dist = math.hypot(thumb[0] - index[0], thumb[1] - index[1])
        normalized_dist = pinch_dist / hand_scale

        # Check if other fingers (ring, pinky) are not in Open_Palm mode or if thumb/index are in focus
        # In pinch posture: thumb & index are interacting, middle_tip is usually further
        # Typical normalized_dist ranges from 0.18 (touching) to 0.95 (open)
        if 0.10 <= normalized_dist <= 1.25 and det.gesture_name not in ("Closed_Fist", "Open_Palm"):
            # Map normalized pinch distance to 0.0 - 1.0 volume range
            target_vol = float(np.interp(normalized_dist, [0.22, 0.95], [0.0, 1.0]))
            target_vol = max(0.0, min(1.0, target_vol))

            # Exponential Moving Average (EMA) for zero-jitter volume sliding
            self.smoothed_volume = (
                self.smoothing_alpha * target_vol + (1.0 - self.smoothing_alpha) * self.smoothed_volume
            )

            # Apply volume update to OS
            self.media.set_volume(self.smoothed_volume, log_action=False)

            self.is_pinching = True
            mid_pt = ((thumb[0] + index[0]) // 2, (thumb[1] + index[1]) // 2)
            self.pinch_points = (thumb, index, mid_pt)
            return True

        return False

    def _evaluate_discrete_gesture(self, det: HandDetection, is_ready: bool, now: float):
        """Debounces and executes discrete gesture triggers (Play/Pause, Next, Prev, Mute)."""
        gesture = det.gesture_name

        if gesture in ("None", "Pointing_Up") and self.is_pinching:
            # If pinching, don't trigger discrete gestures
            self.consecutive_gesture = None
            self.consecutive_count = 0
            return

        # Track consecutive frames of the same gesture for noise filtration
        if gesture == self.consecutive_gesture:
            self.consecutive_count += 1
        else:
            self.consecutive_gesture = gesture
            self.consecutive_count = 1

        # Only trigger if held for required consecutive frames and cooldown satisfied
        if is_ready and self.consecutive_count >= self.hold_frames:
            action_taken = False

            if gesture == "Open_Palm":
                self.media.play_pause()
                action_taken = True
            elif gesture == "Closed_Fist":
                self.media.toggle_mute()
                action_taken = True
            elif gesture == "Victory":
                self.media.next_track()
                action_taken = True
            elif gesture == "Pointing_Up":
                self.media.prev_track()
                action_taken = True
            elif gesture == "Thumb_Up":
                self.media.volume_step(+0.08)
                action_taken = True
            elif gesture == "Thumb_Down":
                self.media.volume_step(-0.08)
                action_taken = True

            if action_taken:
                self.last_triggered_time = now
                self.consecutive_count = 0  # Reset counter
                self.cooldown_progress = 1.0

    def _build_state_summary(self) -> Dict:
        """Returns structured state for HUD visualizer."""
        action_text, action_time = self.media.get_latest_action()
        return {
            "is_pinching": self.is_pinching,
            "pinch_points": self.pinch_points,
            "current_volume": self.media.get_volume(),
            "is_muted": self.media.is_muted,
            "playback_state": self.media.playback_state,
            "latest_action": action_text,
            "action_age": time.time() - action_time,
            "cooldown_progress": self.cooldown_progress,
            "consecutive_gesture": self.consecutive_gesture,
            "consecutive_count": self.consecutive_count
        }
