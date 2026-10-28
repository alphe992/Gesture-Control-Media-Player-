"""
media_controller.py
-------------------
Handles system-level media playback and master volume control on Windows.
Supports:
- Master volume read & write via pycaw (Core Audio API)
- Touchless Play/Pause, Next Track, Previous Track, Mute via Windows Virtual Media Keys
- Toast/notification state management for HUD overlay
"""

import sys
import time
from typing import Optional, Tuple

# Windows API Virtual Key Codes for Media
VK_VOLUME_MUTE = 0xAD
VK_VOLUME_DOWN = 0xAE
VK_VOLUME_UP = 0xAF
VK_MEDIA_NEXT_TRACK = 0xB0
VK_MEDIA_PREV_TRACK = 0xB1
VK_MEDIA_STOP = 0xB2
VK_MEDIA_PLAY_PAUSE = 0xB3
KEYEVENTF_KEYUP = 0x0002


class MediaController:
    """Controls OS media playback and system volume."""

    def __init__(self):
        self._is_windows = sys.platform.startswith("win")
        self._volume_interface = None
        self._current_volume: float = 0.5  # 0.0 to 1.0
        self._is_muted: bool = False
        self._last_action_text: str = "Ready"
        self._last_action_time: float = time.time()
        self._playback_state: str = "Playing"  # 'Playing', 'Paused', 'Stopped'

        self._init_pycaw()

    def _init_pycaw(self):
        """Initializes PyCAW audio endpoint on Windows."""
        if not self._is_windows:
            print("[MediaController] Non-Windows OS detected; using virtual volume simulation.")
            return

        try:
            from pycaw.pycaw import AudioUtilities
            speakers = AudioUtilities.GetSpeakers()
            if hasattr(speakers, "EndpointVolume") and speakers.EndpointVolume is not None:
                self._volume_interface = speakers.EndpointVolume
            elif hasattr(speakers, "Activate"):
                from ctypes import POINTER, cast
                from comtypes import CLSCTX_ALL
                from pycaw.pycaw import IAudioEndpointVolume
                interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
                self._volume_interface = cast(interface, POINTER(IAudioEndpointVolume))

            if self._volume_interface:
                self._current_volume = float(self._volume_interface.GetMasterVolumeLevelScalar())
                self._is_muted = bool(self._volume_interface.GetMute())
                print(f"[MediaController] Audio endpoint connected. Current Volume: {int(self._current_volume * 100)}%")
        except Exception as e:
            print(f"[MediaController] Warning: Could not initialize pycaw audio endpoint: {e}")
            self._volume_interface = None

    def _send_key_event(self, vk_code: int):
        """Dispatches key press and key release to Windows user32."""
        if not self._is_windows:
            return
        try:
            import ctypes
            ctypes.windll.user32.keybd_event(vk_code, 0, 0, 0)
            time.sleep(0.01)
            ctypes.windll.user32.keybd_event(vk_code, 0, KEYEVENTF_KEYUP, 0)
        except Exception as e:
            print(f"[MediaController] Error sending virtual key {hex(vk_code)}: {e}")

    def set_volume(self, level: float, log_action: bool = True):
        """
        Sets master volume scalar.
        :param level: Volume between 0.0 and 1.0
        :param log_action: Whether to update the HUD action notification
        """
        level = max(0.0, min(1.0, float(level)))
        self._current_volume = level

        if self._volume_interface:
            try:
                self._volume_interface.SetMasterVolumeLevelScalar(level, None)
                if self._is_muted and level > 0.0:
                    self.set_mute(False)
            except Exception as e:
                print(f"[MediaController] Failed to set volume via pycaw: {e}")
        else:
            # Fallback if no pycaw endpoint
            pass

        if log_action:
            self._record_action(f"Volume: {int(level * 100)}%")

    def get_volume(self) -> float:
        """Returns current volume as scalar 0.0 to 1.0."""
        if self._volume_interface:
            try:
                self._current_volume = float(self._volume_interface.GetMasterVolumeLevelScalar())
            except Exception:
                pass
        return self._current_volume

    def set_mute(self, muted: bool):
        """Mutes or unmutes system sound."""
        self._is_muted = muted
        if self._volume_interface:
            try:
                self._volume_interface.SetMute(int(muted), None)
            except Exception as e:
                print(f"[MediaController] Failed to set mute: {e}")
        else:
            self._send_key_event(VK_VOLUME_MUTE)

        self._record_action("Muted" if muted else "Unmuted")

    def toggle_mute(self):
        """Toggles current mute status."""
        new_mute = not self._is_muted
        self.set_mute(new_mute)

    def play_pause(self):
        """Toggles play / pause state via global media key."""
        self._send_key_event(VK_MEDIA_PLAY_PAUSE)
        if self._playback_state == "Playing":
            self._playback_state = "Paused"
            self._record_action("Playback: Paused")
        else:
            self._playback_state = "Playing"
            self._record_action("Playback: Playing")

    def next_track(self):
        """Skips to the next track."""
        self._send_key_event(VK_MEDIA_NEXT_TRACK)
        self._record_action("Next Track >>")

    def prev_track(self):
        """Rewinds to the previous track."""
        self._send_key_event(VK_MEDIA_PREV_TRACK)
        self._record_action("<< Previous Track")

    def stop(self):
        """Stops media playback."""
        self._send_key_event(VK_MEDIA_STOP)
        self._playback_state = "Stopped"
        self._record_action("Playback: Stopped")

    def volume_step(self, delta: float):
        """Increments or decrements volume by scalar delta (e.g., +0.05)."""
        current = self.get_volume()
        self.set_volume(current + delta)

    def _record_action(self, action_text: str):
        """Updates action HUD toast and timestamp."""
        self._last_action_text = action_text
        self._last_action_time = time.time()

    def get_latest_action(self) -> Tuple[str, float]:
        """Returns tuple of (action_text, timestamp)."""
        return self._last_action_text, self._last_action_time

    @property
    def is_muted(self) -> bool:
        return self._is_muted

    @property
    def playback_state(self) -> str:
        return self._playback_state
