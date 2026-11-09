"""Blink and dwell gesture state machines."""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class GestureEvent:
    action: str
    timestamp: float


class GestureDetector:
    def __init__(
        self,
        blink_enabled: bool,
        dwell_enabled: bool,
        right_blink_enabled: bool,
        blink_seconds: float,
        right_blink_seconds: float,
        cooldown: float,
        dwell_seconds: float,
        dwell_radius: int,
    ) -> None:
        self.blink_enabled = blink_enabled
        self.dwell_enabled = dwell_enabled
        self.right_blink_enabled = right_blink_enabled
        self.blink_seconds = blink_seconds
        self.right_blink_seconds = right_blink_seconds
        self.cooldown = cooldown
        self.dwell_seconds = dwell_seconds
        self.dwell_radius = dwell_radius
        self.closed_since: Optional[float] = None
        self.last_click = 0.0
        self.dwell_anchor: Optional[tuple[float, float]] = None
        self.dwell_since: Optional[float] = None
        self.dwell_fired = False

    def update(
        self,
        eyes_closed: bool,
        gaze: Optional[tuple[float, float]],
        now: Optional[float] = None,
    ) -> Optional[GestureEvent]:
        now = time.monotonic() if now is None else now
        event = None

        if eyes_closed:
            if self.closed_since is None:
                self.closed_since = now
        elif self.closed_since is not None:
            duration = now - self.closed_since
            self.closed_since = None
            if now - self.last_click >= self.cooldown:
                if self.right_blink_enabled and duration >= self.right_blink_seconds:
                    event = GestureEvent("right_click", now)
                elif self.blink_enabled and duration >= self.blink_seconds:
                    event = GestureEvent("left_click", now)
                if event:
                    self.last_click = now

        if self.dwell_enabled and gaze is not None and event is None:
            if self.dwell_anchor is None:
                self.dwell_anchor, self.dwell_since = gaze, now
            else:
                distance = math.dist(gaze, self.dwell_anchor)
                if distance > self.dwell_radius:
                    self.dwell_anchor = gaze
                    self.dwell_since = now
                    self.dwell_fired = False
                elif (
                    self.dwell_since is not None
                    and not self.dwell_fired
                    and now - self.dwell_since >= self.dwell_seconds
                    and now - self.last_click >= self.cooldown
                ):
                    event = GestureEvent("left_click", now)
                    self.last_click = now
                    self.dwell_fired = True
        else:
            self.dwell_anchor = None
            self.dwell_since = None
            self.dwell_fired = False

        return event
