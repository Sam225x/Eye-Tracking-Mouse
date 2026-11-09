"""Desktop GUI for Eye Tracking Mouse."""
from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

import cv2
import numpy as np
import pyautogui
from PIL import Image, ImageTk

from calibration import GazeCalibration, nine_screen_targets
from eye_tracker import EyeTracker
from gestures import GestureDetector
from settings import Settings, load_settings, save_settings

pyautogui.FAILSAFE = True


class EyeTrackingApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Eye Tracking Mouse")
        self.root.geometry("1040x720")
        self.root.minsize(820, 620)
        self.settings = load_settings()
        self.calibration = GazeCalibration()
        self.calibration.load()
        self.tracker: Optional[EyeTracker] = None
        self.running = False
        self.worker: Optional[threading.Thread] = None
        self.stop_event = threading.Event()
        self.frame_lock = threading.Lock()
        self.latest_frame = None
        self.latest_features: Optional[list[float]] = None
        self.latest_eyes_closed = False
        self.latest_face_detected = False
        self.status = tk.StringVar(value="Ready — start tracking, then calibrate.")
        self.camera_var = tk.IntVar(value=self.settings.camera_index)
        self.smoothing_var = tk.DoubleVar(value=self.settings.smoothing)
        self.blink_var = tk.BooleanVar(value=self.settings.blink_click_enabled)
        self.dwell_var = tk.BooleanVar(value=self.settings.dwell_click_enabled)
        self.right_var = tk.BooleanVar(value=self.settings.right_blink_enabled)
        self.dwell_seconds_var = tk.DoubleVar(value=self.settings.dwell_seconds)
        self._build_ui()
        self._refresh_preview()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill="both", expand=True)
        heading = ttk.Label(outer, text="Eye Tracking Mouse", font=("Helvetica", 22, "bold"))
        heading.pack(anchor="w")
        ttk.Label(
            outer,
            text="Control the pointer with gaze direction and optional blink/dwell gestures.",
        ).pack(anchor="w", pady=(2, 10))

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        self.preview = ttk.Label(body, text="Camera preview appears here", anchor="center")
        self.preview.pack(side="left", fill="both", expand=True, padx=(0, 12))

        side = ttk.Frame(body, width=280)
        side.pack(side="right", fill="y")
        side.pack_propagate(False)
        ttk.Label(side, text="Controls", font=("Helvetica", 14, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(side, text="Camera index").pack(anchor="w")
        ttk.Spinbox(side, from_=0, to=10, textvariable=self.camera_var, width=8).pack(anchor="w", pady=(0, 8))
        self.start_button = ttk.Button(side, text="Start tracking", command=self.start_tracking)
        self.start_button.pack(fill="x", pady=3)
        self.stop_button = ttk.Button(side, text="Stop tracking", command=self.stop_tracking, state="disabled")
        self.stop_button.pack(fill="x", pady=3)
        self.calibrate_button = ttk.Button(side, text="Calibrate (9 points)", command=self.start_calibration, state="disabled")
        self.calibrate_button.pack(fill="x", pady=(3, 10))

        ttk.Separator(side).pack(fill="x", pady=8)
        ttk.Label(side, text="Mouse gestures", font=("Helvetica", 12, "bold")).pack(anchor="w")
        ttk.Checkbutton(side, text="Blink = left click", variable=self.blink_var).pack(anchor="w", pady=3)
        ttk.Checkbutton(side, text="Long blink = right click", variable=self.right_var).pack(anchor="w", pady=3)
        ttk.Checkbutton(side, text="Dwell = left click", variable=self.dwell_var).pack(anchor="w", pady=3)
        ttk.Label(side, text="Dwell duration (seconds)").pack(anchor="w", pady=(8, 0))
        ttk.Scale(side, from_=0.8, to=3.0, variable=self.dwell_seconds_var, orient="horizontal").pack(fill="x")
        ttk.Label(side, text="Cursor smoothing").pack(anchor="w", pady=(8, 0))
        ttk.Scale(side, from_=0.08, to=0.75, variable=self.smoothing_var, orient="horizontal").pack(fill="x")
        ttk.Button(side, text="Save settings", command=self.save_current_settings).pack(fill="x", pady=(12, 4))
        ttk.Button(side, text="Reset calibration", command=self.reset_calibration).pack(fill="x", pady=3)

        ttk.Separator(outer).pack(fill="x", pady=(12, 8))
        ttk.Label(outer, textvariable=self.status, wraplength=950).pack(anchor="w")
        ttk.Label(
            outer,
            text="Safety: keep your trackpad/mouse available. Press Stop tracking to disable control immediately.",
            foreground="#9a5b00",
        ).pack(anchor="w", pady=(5, 0))

    def _current_settings(self) -> Settings:
        self.settings.camera_index = max(0, int(self.camera_var.get()))
        self.settings.smoothing = float(self.smoothing_var.get())
        self.settings.blink_click_enabled = bool(self.blink_var.get())
        self.settings.dwell_click_enabled = bool(self.dwell_var.get())
        self.settings.right_blink_enabled = bool(self.right_var.get())
        self.settings.dwell_seconds = float(self.dwell_seconds_var.get())
        return self.settings

    def save_current_settings(self) -> None:
        save_settings(self._current_settings())
        self.status.set("Settings saved locally.")

    def start_tracking(self) -> None:
        if self.running:
            return
        self._current_settings()
        try:
            self.tracker = EyeTracker(self.settings.camera_index)
            self.tracker.start()
        except Exception as exc:
            if self.tracker:
                self.tracker.stop()
            self.tracker = None
            messagebox.showerror("Camera error", str(exc))
            return

        self.running = True
        self.stop_event.clear()
        self.worker = threading.Thread(target=self._tracking_loop, daemon=True)
        self.worker.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.calibrate_button.configure(state="normal")
        self.status.set("Camera running. Calibrate for pointer control.")

    def stop_tracking(self) -> None:
        if not self.running:
            return
        self.running = False
        self.stop_event.set()
        if self.worker and self.worker.is_alive():
            self.worker.join(timeout=2.0)
        if self.tracker:
            self.tracker.stop()
            self.tracker = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.calibrate_button.configure(state="disabled")
        self.status.set("Tracking stopped.")

    def _tracking_loop(self) -> None:
        screen_w, screen_h = pyautogui.size()
        current_x, current_y = screen_w / 2, screen_h / 2
        gesture = self._make_gesture_detector()
        last_status = 0.0

        while not self.stop_event.is_set() and self.tracker is not None:
            try:
                result = self.tracker.read()
            except Exception as exc:
                self.root.after(0, lambda e=str(exc): self._tracking_error(e))
                break

            with self.frame_lock:
                self.latest_frame = result.frame.copy()
                self.latest_features = result.features[:] if result.features else None
                self.latest_eyes_closed = result.eyes_closed
                self.latest_face_detected = result.face_detected

            features = result.features
            if result.face_detected and features is not None and self.calibration.calibrated:
                gx, gy = self.calibration.predict(features)
                target_x = float(np.clip(gx, 0, screen_w - 1))
                target_y = float(np.clip(gy, 0, screen_h - 1))
                smoothing = self.settings.smoothing
                current_x += (target_x - current_x) * smoothing
                current_y += (target_y - current_y) * smoothing
                try:
                    pyautogui.moveTo(int(current_x), int(current_y), _pause=False)
                except pyautogui.FailSafeException:
                    self.root.after(0, self.stop_tracking)
                    break

                event = gesture.update(result.eyes_closed, (current_x, current_y))
                if event:
                    try:
                        if event.action == "left_click":
                            pyautogui.click()
                        elif event.action == "right_click":
                            pyautogui.click(button="right")
                    except pyautogui.FailSafeException:
                        self.root.after(0, self.stop_tracking)
                        break

            now = time.monotonic()
            if now - last_status > 0.5:
                if not result.face_detected:
                    text = result.message
                else:
                    text = "Calibrated — gaze control active" if self.calibration.calibrated else "Face detected — please calibrate"
                self.root.after(0, lambda t=text: self.status.set(t))
                last_status = now
            time.sleep(0.01)

        self.root.after(0, self._worker_finished)

    def _make_gesture_detector(self) -> GestureDetector:
        s = self.settings
        return GestureDetector(
            s.blink_click_enabled,
            s.dwell_click_enabled,
            s.right_blink_enabled,
            s.blink_closed_seconds,
            s.right_blink_seconds,
            s.click_cooldown_seconds,
            s.dwell_seconds,
            s.dwell_radius_px,
        )

    def _tracking_error(self, message: str) -> None:
        self.status.set(f"Tracking error: {message}")
        self.stop_tracking()

    def _worker_finished(self) -> None:
        if self.running:
            self.stop_tracking()

    def _refresh_preview(self) -> None:
        frame = None
        with self.frame_lock:
            if self.latest_frame is not None:
                frame = self.latest_frame.copy()
        if frame is not None:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb)
                image.thumbnail((720, 560))
                photo = ImageTk.PhotoImage(image=image)
                self.preview.configure(image=photo, text="")
                self.preview.image = photo
            except Exception:
                pass
        self.root.after(30, self._refresh_preview)

    def start_calibration(self) -> None:
        if not self.running:
            return
        targets = nine_screen_targets(*pyautogui.size())
        dialog = tk.Toplevel(self.root)
        dialog.title("Eye calibration")
        dialog.geometry("480x230")
        dialog.transient(self.root)
        dialog.grab_set()
        index = 0
        features: list[list[float]] = []
        screen_targets: list[list[float]] = []
        prompt = tk.StringVar(value="Look at the first target shown below, then click Capture.")
        ttk.Label(dialog, text="Nine-point calibration", font=("Helvetica", 16, "bold")).pack(pady=(18, 8))
        ttk.Label(dialog, textvariable=prompt, wraplength=430).pack(pady=5)
        target_label = ttk.Label(dialog, text="", font=("Helvetica", 22, "bold"))
        target_label.pack(pady=8)
        capture = ttk.Button(dialog, text="Capture", state="disabled")
        capture.pack(pady=8)

        def show_target() -> None:
            if index >= len(targets):
                try:
                    self.calibration.fit(features, screen_targets)
                    self.calibration.save()
                    self.status.set("Calibration complete and saved. Gaze mouse control is active.")
                    dialog.destroy()
                except Exception as exc:
                    messagebox.showerror("Calibration failed", str(exc), parent=dialog)
                    dialog.destroy()
                return
            target_label.configure(text=f"Target {index + 1}/9:  {targets[index][0]}, {targets[index][1]}")
            prompt.set("Look at the target on screen, keep your head still, then click Capture.")
            capture.configure(state="normal")

        def do_capture() -> None:
            nonlocal index
            if not self.running:
                messagebox.showwarning("Tracking stopped", "Restart tracking before calibrating.", parent=dialog)
                dialog.destroy()
                return
            with self.frame_lock:
                sample = self.latest_features[:] if self.latest_features else None
                face_detected = self.latest_face_detected
            if not face_detected or sample is None:
                prompt.set("Face/iris not detected. Adjust your position and try Capture again.")
                return
            features.append(sample)
            screen_targets.append([float(targets[index][0]), float(targets[index][1])])
            index += 1
            capture.configure(state="disabled")
            show_target()

        capture.configure(command=do_capture)
        show_target()

    def reset_calibration(self) -> None:
        self.calibration.reset()
        self.status.set("Calibration reset. Run the 9-point calibration again.")

    def close(self) -> None:
        self.stop_tracking()
        self._current_settings()
        save_settings(self.settings)
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    EyeTrackingApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
