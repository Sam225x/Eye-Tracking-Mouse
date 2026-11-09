"""User settings and persistence for Eye Tracking Mouse."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".eye_tracking_mouse.json"


@dataclass
class Settings:
    camera_index: int = 0
    smoothing: float = 0.28
    blink_click_enabled: bool = True
    dwell_click_enabled: bool = False
    right_blink_enabled: bool = False
    blink_closed_seconds: float = 0.12
    right_blink_seconds: float = 0.65
    click_cooldown_seconds: float = 0.8
    dwell_seconds: float = 1.5
    dwell_radius_px: int = 28
    show_preview: bool = True


def load_settings() -> Settings:
    if CONFIG_PATH.exists():
        try:
            raw = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            allowed = Settings.__dataclass_fields__.keys()
            return Settings(**{key: value for key, value in raw.items() if key in allowed})
        except (OSError, ValueError, TypeError):
            pass
    return Settings()


def save_settings(settings: Settings) -> None:
    try:
        CONFIG_PATH.write_text(
            json.dumps(asdict(settings), indent=2),
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"Could not save settings to {CONFIG_PATH}: {exc}")
