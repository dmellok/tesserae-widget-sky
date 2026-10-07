"""sky_moon reads the wall clock in the location's zone, not the
server's: the sun-arc "now" marker, the header time and the next-phase
date all follow the timezone Open-Meteo reports for the coordinates."""

from __future__ import annotations

import importlib.util
import json
import types
from datetime import datetime
from pathlib import Path
from typing import Any
from unittest.mock import patch
from zoneinfo import ZoneInfo

_SERVER = Path(__file__).resolve().parent.parent / "server.py"
MEL = ZoneInfo("Australia/Melbourne")


def _load() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("sky_moon_server_tz", _SERVER)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Resp:
    def __init__(self, body: dict[str, Any]) -> None:
        self._raw = json.dumps(body).encode()

    def read(self) -> bytes:
        return self._raw

    def __enter__(self) -> _Resp:
        return self

    def __exit__(self, *a: object) -> bool:
        return False


def _payload(**extra: Any) -> dict[str, Any]:
    today = datetime.now(MEL).date().isoformat()
    body: dict[str, Any] = {
        "daily": {
            "sunrise": [f"{today}T06:30"],
            "sunset": [f"{today}T19:30"],
            "moonrise": [f"{today}T10:00"],
            "moonset": [f"{today}T23:00"],
        }
    }
    body.update(extra)
    return body


def _mins(dt: datetime) -> int:
    return dt.hour * 60 + dt.minute


def test_now_follows_open_meteo_zone(tmp_path: Path) -> None:
    mod = _load()
    with patch(
        "urllib.request.urlopen", return_value=_Resp(_payload(timezone="Australia/Melbourne"))
    ):
        out = mod.fetch(
            {"latitude": -37.8, "longitude": 144.9, "label": "Melbourne"},
            {},
            ctx={"data_dir": str(tmp_path)},
        )
    now = datetime.now(MEL)
    assert out["tz"] == "Australia/Melbourne"
    assert out["time"] == now.strftime("%H:%M")
    assert abs(out["sun"]["nowMin"] - _mins(now)) <= 1
    # Cache hit re-stamps in the same zone.
    out2 = mod.fetch({"latitude": -37.8, "longitude": 144.9}, {}, ctx={"data_dir": str(tmp_path)})
    assert out2["time"] == datetime.now(MEL).strftime("%H:%M")
    assert out2["nowMin"] == out2["sun"]["nowMin"]


def test_location_option_zone_when_upstream_has_none(tmp_path: Path) -> None:
    mod = _load()
    with patch("urllib.request.urlopen", return_value=_Resp(_payload())):
        out = mod.fetch(
            {
                "latitude": -37.8,
                "longitude": 144.9,
                "location": {"name": "Melbourne", "timezone": "Australia/Melbourne"},
            },
            {},
            ctx={"data_dir": str(tmp_path)},
        )
    assert out["tz"] == "Australia/Melbourne"
    assert out["time"] == datetime.now(MEL).strftime("%H:%M")


def test_offset_only(tmp_path: Path) -> None:
    mod = _load()
    off = int(datetime.now(MEL).utcoffset().total_seconds())  # type: ignore[union-attr]
    with patch("urllib.request.urlopen", return_value=_Resp(_payload(utc_offset_seconds=off))):
        out = mod.fetch(
            {"latitude": -37.8, "longitude": 144.9}, {}, ctx={"data_dir": str(tmp_path)}
        )
    assert out["tz"] is None
    assert out["utcOffsetSec"] == off
    assert out["time"] == datetime.now(MEL).strftime("%H:%M")


def test_no_zone_falls_back_to_server_clock(tmp_path: Path) -> None:
    mod = _load()
    out = mod.fetch({"latitude": 0, "longitude": 0}, {}, ctx={"data_dir": str(tmp_path)})
    assert out["tz"] is None
    assert out["time"] == datetime.now().strftime("%H:%M")
