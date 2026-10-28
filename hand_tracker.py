"""
hand_tracker.py
---------------
Performs real-time hand detection, 21-landmark 3D coordinate extraction,
and gesture classification using Google MediaPipe Tasks API.
"""

import os
import urllib.request
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Official Google Cloud download URL for the pre-trained gesture recognizer task
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/1/gesture_recognizer.task"
MODEL_FILENAME = "gesture_recognizer.task"

# Standard 21 Hand Landmark skeletal connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (9, 10), (10, 11), (11, 12),
    # Ring finger
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm base connections
    (5, 9), (9, 13), (13, 17)
]


@dataclass
class HandDetection:
    """Encapsulates hand detection attributes, coordinates, and classification."""
    handedness: str  # 'Left' or 'Right'
    gesture_name: str  # e.g., 'Open_Palm', 'Closed_Fist', 'Pointing_Up', etc.
    gesture_score: float  # Confidence [0.0, 1.0]
    landmarks_px: List[Tuple[int, int]]  # 21 (x, y) pixel coordinates
    landmarks_norm: List[Tuple[float, float, float]]  # 21 (x, y, z) normalized coords

    @property
    def wrist(self) -> Tuple[int, int]:
        return self.landmarks_px[0]

    @property
    def thumb_tip(self) -> Tuple[int, int]:
        return self.landmarks_px[4]

    @property
    def index_mcp(self) -> Tuple[int, int]:
        return self.landmarks_px[5]

    @property
    def index_tip(self) -> Tuple[int, int]:
        return self.landmarks_px[8]

    @property
    def middle_mcp(self) -> Tuple[int, int]:
        return self.landmarks_px[9]

    @property
    def middle_tip(self) -> Tuple[int, int]:
        return self.landmarks_px[12]

    @property
    def ring_tip(self) -> Tuple[int, int]:
        return self.landmarks_px[16]

    @property
    def pinky_tip(self) -> Tuple[int, int]:
        return self.landmarks_px[20]


class HandTracker:
    """Manages MediaPipe GestureRecognizer life-cycle, model loading, and frame inference."""

    def __init__(self, model_path: Optional[str] = None, num_hands: int = 2, min_confidence: float = 0.5):
        if model_path is None:
            # Check script directory or cwd
            base_dir = os.path.dirname(os.path.abspath(__file__))
            candidate = os.path.join(base_dir, MODEL_FILENAME)
            if not os.path.exists(candidate):
                candidate = MODEL_FILENAME
            model_path = candidate

        self.model_path = self._ensure_model(model_path)
        self.num_hands = num_hands
        self.min_confidence = min_confidence

        # Initialize MediaPipe Task
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=self.num_hands,
            min_hand_detection_confidence=self.min_confidence,
            min_hand_presence_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

    def _ensure_model(self, path: str) -> str:
        """Downloads model file automatically if absent."""
        if not os.path.exists(path):
            print(f"[HandTracker] Downloading gesture model from Google Cloud Storage to '{path}'...")
            try:
                urllib.request.urlretrieve(MODEL_URL, path)
                print("[HandTracker] Model downloaded successfully.")
            except Exception as e:
                raise RuntimeError(f"Failed to auto-download model from {MODEL_URL}: {e}")
        return path

    def process_frame(self, frame_bgr: np.ndarray) -> List[HandDetection]:
        """
        Runs gesture and landmark recognition on a BGR video frame.
        :param frame_bgr: OpenCV image (H, W, 3)
        :return: List of HandDetection objects
        """
        h, w = frame_bgr.shape[:2]
        # MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        result = self.recognizer.recognize(mp_image)
        detections: List[HandDetection] = []

        if not result.hand_landmarks:
            return detections

        num_detected = len(result.hand_landmarks)
        for i in range(num_detected):
            # 1. Landmark coordinates
            raw_landmarks = result.hand_landmarks[i]
            px_coords = []
            norm_coords = []
            for lm in raw_landmarks:
                norm_coords.append((lm.x, lm.y, lm.z))
                px_x = int(round(lm.x * w))
                px_y = int(round(lm.y * h))
                px_coords.append((px_x, px_y))

            # 2. Handedness
            handedness_label = "Right"
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                handedness_label = result.handedness[i][0].category_name

            # 3. Gesture classification
            gesture_name = "None"
            gesture_score = 0.0
            if result.gestures and i < len(result.gestures) and result.gestures[i]:
                top_cat = result.gestures[i][0]
                gesture_name = top_cat.category_name
                gesture_score = float(top_cat.score)

            detections.append(HandDetection(
                handedness=handedness_label,
                gesture_name=gesture_name,
                gesture_score=gesture_score,
                landmarks_px=px_coords,
                landmarks_norm=norm_coords
            ))

        return detections

    @staticmethod
    def draw_skeleton(frame_bgr: np.ndarray, detection: HandDetection, color_primary=(0, 240, 255), color_joints=(255, 120, 0)):
        """Renders anti-aliased neon skeletal connections and joint nodes."""
        px = detection.landmarks_px

        # Draw connecting bones
        for p1, p2 in HAND_CONNECTIONS:
            cv2.line(frame_bgr, px[p1], px[p2], color_primary, 2, cv2.LINE_AA)

        # Draw joint nodes
        for i, pt in enumerate(px):
            # Highlight fingertips (4, 8, 12, 16, 20) with glowing rings
            if i in (4, 8, 12, 16, 20):
                cv2.circle(frame_bgr, pt, 6, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame_bgr, pt, 9, color_joints, 2, cv2.LINE_AA)
            else:
                cv2.circle(frame_bgr, pt, 4, color_joints, -1, cv2.LINE_AA)
