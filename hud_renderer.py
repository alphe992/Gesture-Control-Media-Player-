"""
hud_renderer.py
---------------
Renders a modern, glassmorphic Heads-Up Display (HUD) overlay using OpenCV and Pillow.
Features:
- Anti-aliased typography via Pillow TrueType fonts
- Neon hand skeletal tracking with joint highlights
- Dynamic pinch laser line and volume orb
- Interactive volume bar with dynamic audio icons
- Animated gesture toast notifications and debouncing cooldown meter
- Toggleable gesture cheat sheet overlay
"""

import os
import sys
import time
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from hand_tracker import HandDetection


class HUDRenderer:
    """Manages visual telemetry, glassmorphism cards, and text rendering on OpenCV frames."""

    def __init__(self, width: int = 1280, height: int = 720):
        self.width = width
        self.height = height
        self.show_help: bool = False

        # Font configuration
        self._init_fonts()

        # Color Palette (BGR for OpenCV, RGB for PIL)
        self.PALETTE = {
            "bg_dark": (20, 24, 30),
            "accent_cyan": (255, 200, 0),      # BGR
            "accent_cyan_rgb": (0, 200, 255),  # RGB
            "accent_purple": (200, 50, 160),
            "accent_green": (80, 220, 100),
            "accent_amber": (50, 180, 255),
            "text_white": (245, 245, 245),
            "text_gray": (170, 175, 185),
            "card_bg": (15, 18, 24),
            "border_glow": (100, 70, 40)
        }

    def _init_fonts(self):
        """Finds and loads TrueType fonts on the host system with fallback."""
        font_candidates = [
            "C:\\Windows\\Fonts\\segoeui.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ]
        chosen_font = None
        for candidate in font_candidates:
            if os.path.exists(candidate):
                chosen_font = candidate
                break

        try:
            if chosen_font:
                self.font_sm = ImageFont.truetype(chosen_font, 14)
                self.font_md = ImageFont.truetype(chosen_font, 18)
                self.font_lg = ImageFont.truetype(chosen_font, 26)
                self.font_xl = ImageFont.truetype(chosen_font, 36)
            else:
                self.font_sm = ImageFont.load_default()
                self.font_md = ImageFont.load_default()
                self.font_lg = ImageFont.load_default()
                self.font_xl = ImageFont.load_default()
        except Exception:
            self.font_sm = ImageFont.load_default()
            self.font_md = ImageFont.load_default()
            self.font_lg = ImageFont.load_default()
            self.font_xl = ImageFont.load_default()

    def toggle_help(self):
        """Toggles cheat sheet modal."""
        self.show_help = not self.show_help

    def render(
        self,
        frame: np.ndarray,
        detections: List[HandDetection],
        engine_state: Dict,
        fps: float
    ) -> np.ndarray:
        """
        Composites HUD elements onto the camera frame.
        :param frame: BGR image from webcam
        :param detections: Detected hands
        :param engine_state: State dictionary from GestureEngine
        :param fps: Estimated frames per second
        :return: Composited BGR image
        """
        h, w = frame.shape[:2]
        self.height, self.width = h, w

        # 1. Draw hand skeletons and pinch laser line directly on frame
        self._render_hand_geometry(frame, detections, engine_state)

        # 2. Draw glassmorphism UI overlay using hybrid OpenCV alpha blend + Pillow text
        overlay = frame.copy()

        # Top Header Bar
        self._draw_header_card(overlay, fps)

        # Right-side Volume Card
        self._draw_volume_card(overlay, engine_state)

        # Left-side Status / Gesture Card
        self._draw_status_card(overlay, detections, engine_state)

        # Action Toast Notification
        self._draw_action_toast(overlay, engine_state)

        # Alpha blend HUD overlay with camera frame (80% opacity for HUD elements)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # 3. Render crisp TrueType text via Pillow on top
        frame = self._render_typography(frame, detections, engine_state, fps)

        # 4. Optional Help Modal
        if self.show_help:
            frame = self._render_help_modal(frame)

        return frame

    def _render_hand_geometry(self, frame: np.ndarray, detections: List[HandDetection], state: Dict):
        """Draws glowing joints, bones, and pinch laser line."""
        for det in detections:
            # Draw bones
            from hand_tracker import HandTracker
            HandTracker.draw_skeleton(frame, det)

            # Draw gesture badge above the wrist
            wrist = det.wrist
            gesture_text = det.gesture_name.replace("_", " ")
            if gesture_text != "None":
                cv2.circle(frame, (wrist[0], wrist[1] - 25), 5, (0, 255, 120), -1, cv2.LINE_AA)

        # Pinch Laser Line
        if state.get("is_pinching") and state.get("pinch_points"):
            p1, p2, mid = state["pinch_points"]
            # Bright cyan laser line
            cv2.line(frame, p1, p2, (255, 240, 0), 3, cv2.LINE_AA)
            cv2.line(frame, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
            # Pulsing center orb
            vol_pct = int(state.get("current_volume", 0.0) * 100)
            cv2.circle(frame, mid, 16, (20, 24, 30), -1, cv2.LINE_AA)
            cv2.circle(frame, mid, 16, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.circle(frame, mid, 6, (255, 255, 255), -1, cv2.LINE_AA)

    def _draw_header_card(self, overlay: np.ndarray, fps: float):
        """Glassmorphism top banner with application title and telemetry."""
        w = self.width
        # Rounded header bar
        cv2.rectangle(overlay, (20, 15), (w - 20, 65), (15, 18, 25), -1)
        cv2.rectangle(overlay, (20, 15), (w - 20, 65), (45, 55, 75), 1)

        # Camera live indicator dot
        cv2.circle(overlay, (42, 40), 6, (50, 220, 100), -1, cv2.LINE_AA)

    def _draw_volume_card(self, overlay: np.ndarray, state: Dict):
        """Vertical sleek volume meter on the right side."""
        w, h = self.width, self.height
        card_w = 90
        card_h = 320
        top_y = 90
        left_x = w - card_w - 25

        # Container
        cv2.rectangle(overlay, (left_x, top_y), (left_x + card_w, top_y + card_h), (18, 22, 30), -1)
        cv2.rectangle(overlay, (left_x, top_y), (left_x + card_w, top_y + card_h), (45, 55, 75), 1)

        # Volume slider background track
        bar_x = left_x + 35
        bar_w = 20
        bar_top = top_y + 60
        bar_bottom = top_y + card_h - 45
        bar_h = bar_bottom - bar_top

        cv2.rectangle(overlay, (bar_x, bar_top), (bar_x + bar_w, bar_bottom), (30, 36, 48), -1)

        # Filled volume level
        vol = max(0.0, min(1.0, state.get("current_volume", 0.0)))
        fill_h = int(bar_h * vol)
        fill_top = bar_bottom - fill_h

        if fill_h > 0:
            fill_color = (60, 200, 255) if not state.get("is_muted") else (80, 80, 100)
            cv2.rectangle(overlay, (bar_x, fill_top), (bar_x + bar_w, bar_bottom), fill_color, -1)
            # Glowing slider thumb line
            cv2.rectangle(overlay, (bar_x - 3, fill_top - 2), (bar_x + bar_w + 3, fill_top + 3), (255, 255, 255), -1)

    def _draw_status_card(self, overlay: np.ndarray, detections: List[HandDetection], state: Dict):
        """Left panel showing playback state, detected gesture, and cooldown."""
        top_y = 90
        left_x = 25
        card_w = 260
        card_h = 190

        # Background card
        cv2.rectangle(overlay, (left_x, top_y), (left_x + card_w, top_y + card_h), (18, 22, 30), -1)
        cv2.rectangle(overlay, (left_x, top_y), (left_x + card_w, top_y + card_h), (45, 55, 75), 1)

        # Cooldown progress bar at bottom of card
        cooldown_pct = state.get("cooldown_progress", 0.0)
        cd_bar_w = int((card_w - 30) * (1.0 - cooldown_pct))
        bar_y = top_y + card_h - 15
        cv2.rectangle(overlay, (left_x + 15, bar_y), (left_x + card_w - 15, bar_y + 4), (35, 40, 55), -1)
        if cd_bar_w > 0:
            cd_color = (0, 220, 120) if cooldown_pct == 0.0 else (0, 180, 255)
            cv2.rectangle(overlay, (left_x + 15, bar_y), (left_x + 15 + cd_bar_w, bar_y + 4), cd_color, -1)

    def _draw_action_toast(self, overlay: np.ndarray, state: Dict):
        """Animated toast popup when an action is executed."""
        action_age = state.get("action_age", 99.0)
        if action_age > 1.8:
            return  # Toast expired

        text = state.get("latest_action", "")
        if not text or text == "Ready":
            return

        w, h = self.width, self.height
        toast_w = 340
        toast_h = 52
        toast_x = (w - toast_w) // 2
        toast_y = h - 90

        # Card with cyan border
        cv2.rectangle(overlay, (toast_x, toast_y), (toast_x + toast_w, toast_y + toast_h), (20, 26, 36), -1)
        cv2.rectangle(overlay, (toast_x, toast_y), (toast_x + toast_w, toast_y + toast_h), (0, 220, 255), 2)

    def _render_typography(
        self,
        frame: np.ndarray,
        detections: List[HandDetection],
        state: Dict,
        fps: float
    ) -> np.ndarray:
        """Uses Pillow for crisp anti-aliased font rendering."""
        # Convert BGR to RGB for PIL
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        draw = ImageDraw.Draw(pil_img)

        w = self.width

        # 1. Header Text
        draw.text((60, 27), "TOUCHLESS MEDIA CONTROLLER", font=self.font_md, fill=(245, 245, 245))
        fps_color = (80, 230, 120) if fps >= 25 else (255, 180, 50)
        draw.text((w - 220, 27), f"FPS: {fps:4.1f}", font=self.font_md, fill=fps_color)
        draw.text((w - 110, 28), "[H] Help", font=self.font_sm, fill=(160, 170, 185))

        # 2. Right Volume Card Text
        card_w = 90
        left_x = w - card_w - 25
        top_y = 90
        vol_pct = int(state.get("current_volume", 0.0) * 100)
        draw.text((left_x + 18, top_y + 15), "VOLUME", font=self.font_sm, fill=(160, 170, 190))
        vol_str = f"{vol_pct}%" if not state.get("is_muted") else "MUTED"
        draw.text((left_x + 15, top_y + 35), vol_str, font=self.font_md, fill=(0, 220, 255) if not state.get("is_muted") else (255, 100, 100))

        # 3. Left Status Card Text
        st_x = 40
        st_y = 105
        draw.text((st_x, st_y), "PLAYBACK STATUS", font=self.font_sm, fill=(160, 170, 190))
        pb_state = state.get("playback_state", "Playing").upper()
        draw.text((st_x, st_y + 20), f">> {pb_state}", font=self.font_lg, fill=(80, 230, 120) if pb_state == "PLAYING" else (255, 180, 50))

        # Detected Gesture Display
        draw.text((st_x, st_y + 60), "ACTIVE GESTURE", font=self.font_sm, fill=(160, 170, 190))
        active_gesture = "None"
        if detections:
            active_gesture = detections[0].gesture_name.replace("_", " ")
        if state.get("is_pinching"):
            active_gesture = "Pinch Slider"

        gesture_color = (0, 220, 255) if active_gesture != "None" else (130, 140, 155)
        draw.text((st_x, st_y + 80), active_gesture, font=self.font_md, fill=gesture_color)

        cooldown_str = "READY" if state.get("cooldown_progress", 0.0) == 0.0 else "DEBOUNCING..."
        draw.text((st_x, st_y + 115), cooldown_str, font=self.font_sm, fill=(130, 220, 150) if cooldown_str == "READY" else (255, 160, 60))

        # 4. Action Toast Text
        action_age = state.get("action_age", 99.0)
        if action_age <= 1.8:
            text = state.get("latest_action", "")
            if text and text != "Ready":
                toast_w = 340
                toast_x = (w - toast_w) // 2
                toast_y = self.height - 90
                draw.text((toast_x + 30, toast_y + 14), text, font=self.font_md, fill=(255, 255, 255))

        # Convert back to BGR for OpenCV display
        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    def _render_help_modal(self, frame: np.ndarray) -> np.ndarray:
        """Renders semi-transparent help guide overlay."""
        overlay = frame.copy()
        w, h = self.width, self.height
        modal_w = 560
        modal_h = 420
        mx = (w - modal_w) // 2
        my = (h - modal_h) // 2

        # Dim background
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Modal box
        cv2.rectangle(frame, (mx, my), (mx + modal_w, my + modal_h), (20, 25, 35), -1)
        cv2.rectangle(frame, (mx, my), (mx + modal_w, my + modal_h), (0, 200, 255), 2)

        # Pillow text rendering for modal
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)

        draw.text((mx + 30, my + 25), "GESTURE CHEAT SHEET", font=self.font_lg, fill=(0, 220, 255))
        draw.text((mx + modal_w - 140, my + 30), "Press [H] to Close", font=self.font_sm, fill=(180, 190, 200))

        shortcuts = [
            ("Open Palm", "Play / Pause playback toggle"),
            ("Closed Fist", "Mute / Unmute system audio"),
            ("Victory / Peace", "Skip to Next Track >>"),
            ("Pointing Up", "Rewind to Previous Track <<"),
            ("Thumb + Index Pinch", "Precision Continuous Volume Slider"),
            ("Thumb Up / Down", "Step Volume Up / Down (+/- 8%)"),
            ("[H] Key", "Toggle this Help Overlay"),
            ("[Q] or [ESC]", "Exit application")
        ]

        start_y = my + 80
        for i, (gest, desc) in enumerate(shortcuts):
            row_y = start_y + (i * 38)
            draw.text((mx + 35, row_y), gest, font=self.font_md, fill=(255, 220, 100))
            draw.text((mx + 220, row_y), f"-  {desc}", font=self.font_md, fill=(240, 240, 240))

        return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
