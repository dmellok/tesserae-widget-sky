"""sky_moon, sun arc + moon phase, with rise/set/length/noon.

Phase + illumination are computed locally from the synodic-month length
anchored to a known new-moon epoch (2000-01-06 18:14 UTC). Accuracy is
better than ±0.5 day for centuries either side of the epoch, plenty
for a "what does the moon look like tonight" widget.

Sunrise / sunset and moonrise / moonset come from Open-Meteo when
lat/lon are set; failure to fetch is non-fatal (the rest of the card
still renders). The new "Sun & Moon" variants paint from a structured
``sun`` + ``moon`` block (see ``SUNMOON`` in the design handoff); the
legacy variant still reads the flat ``phase_name`` / ``moonrise`` etc.
fields, so both shapes are kept in the payload.
"""

from __future__ import annotations

import contextlib
import json
import math
import time
import urllib.request
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

CACHE_TTL_S = 3600  # phase moves slowly; 1h is plenty
HTTP_TIMEOUT_S = 12
USER_AGENT = "tesserae/0.1 (+sky_moon)"

# Known new moon: 2000-01-06 18:14 UTC. Synodic month: ~29.5305881 days.
_NEW_MOON_EPOCH = datetime(2000, 1, 6, 18, 14, tzinfo=UTC)
_SYNODIC_DAYS = 29.5305881


def _phase_age_days(now: datetime) -> float:
    """Age in days since the last new moon (0..synodic)."""
    delta = (now - _NEW_MOON_EPOCH).total_seconds() / 86400.0
    return delta % _SYNODIC_DAYS


def _illumination(age: float) -> float:
    """Fraction of disc illuminated, 0..1.
    Cosine of phase angle, mapped 0=new, 1=full."""
    phase_angle = 2 * math.pi * age / _SYNODIC_DAYS
    return (1 - math.cos(phase_angle)) / 2


def _phase_name(age: float) -> str:
    """Common name for a phase age in days."""
    # Boundaries at ~3.7 day intervals between the 8 standard phases.
    if age < 1.0 or age > _SYNODIC_DAYS - 1.0:
        return "New Moon"
    if age < 6.4:
        return "Waxing Crescent"
    if age < 8.4:
        return "First Quarter"
    if age < 13.8:
        return "Waxing Gibbous"
    if age < 15.8:
        return "Full Moon"
    if age < 21.1:
        return "Waning Gibbous"
    if age < 23.1:
        return "Last Quarter"
    return "Waning Crescent"


def _next_phase(now: datetime, target_fraction: float) -> datetime:
    """Find the next datetime where the moon age / synodic = target_fraction.
    target_fraction ∈ {0, 0.25, 0.5, 0.75}."""
    current = _phase_age_days(now) / _SYNODIC_DAYS
    delta = (target_fraction - current) % 1.0
    if delta == 0:
        delta = 1.0
    return now + timedelta(days=delta * _SYNODIC_DAYS)


def _fetch_sky(lat: float, lon: float) -> tuple[Any, ...]:
    """Return (sunrise_iso, sunset_iso, moonrise_iso, moonset_iso,
    tz_name, utc_offset_seconds) for today at the location, or all-None
    on failure. Open-Meteo's daily endpoint serves these without auth;
    with ``timezone=auto`` the times are the location's wall clock and
    the payload names that zone."""
    if lat == 0 and lon == 0:
        return None, None, None, None, None, None
    url = (
        "https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        "&daily=sunrise,sunset,moonrise,moonset&timezone=auto&forecast_days=1"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None, None, None, None, None, None
    daily = payload.get("daily") or {}
    sunrise = (daily.get("sunrise") or [None])[0]
    sunset = (daily.get("sunset") or [None])[0]
    moonrise = (daily.get("moonrise") or [None])[0]
    moonset = (daily.get("moonset") or [None])[0]
    tz_name = payload.get("timezone")
    offset = payload.get("utc_offset_seconds")
    return (
        sunrise,
        sunset,
        moonrise,
        moonset,
        tz_name if isinstance(tz_name, str) else None,
        int(offset) if isinstance(offset, (int, float)) and not isinstance(offset, bool) else None,
    )


def _tz_from_name(name: Any) -> tzinfo | None:
    if not isinstance(name, str) or not name.strip():
        return None
    try:
        return ZoneInfo(name.strip())
    except (ValueError, KeyError, OSError):
        return None


def _option_tz_name(options: dict[str, Any]) -> str | None:
    """IANA name carried on a saved location option, if any. Newer
    servers add ``timezone`` to the picker's location dict; older saved
    locations and pasted coordinates don't have one."""
    for value in (options.get("timezone"), options.get("location")):
        if isinstance(value, dict):
            value = value.get("timezone")
        if _tz_from_name(value) is not None:
            return str(value).strip()
    return None


def _resolve_tz(tz_name: Any, offset_s: Any) -> tzinfo | None:
    """Location zone from an IANA name, else a fixed UTC offset in
    seconds. ``None`` means the server's local zone."""
    tz = _tz_from_name(tz_name)
    if tz is not None:
        return tz
    if isinstance(offset_s, (int, float)) and not isinstance(offset_s, bool):
        with contextlib.suppress(ValueError, OverflowError):
            return timezone(timedelta(seconds=int(offset_s)))
    return None


def fetch(
    options: dict[str, Any], settings: dict[str, Any], *, ctx: dict[str, Any]
) -> dict[str, Any]:
    del settings
    lat = float(options.get("latitude") or 0.0)
    lon = float(options.get("longitude") or 0.0)
    label = (options.get("label") or "").strip()

    data_dir = Path(ctx["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    cache = data_dir / f"moon_{lat:.2f}_{lon:.2f}.json"
    if cache.exists() and time.time() - cache.stat().st_mtime < CACHE_TTL_S:
        try:
            cached = json.loads(cache.read_text(encoding="utf-8"))
            # "nowMin" and "time" are wall-clock dependent (in the
            # location's zone), so always re-stamp them on a cache hit; everything else
            # (phase, rise/set, etc.) is stable within the TTL.
            loc_tz = _resolve_tz(cached.get("tz"), cached.get("utcOffsetSec"))
            now_min = _now_min(loc_tz)
            cached["nowMin"] = now_min
            cached["time"] = _now_hhmm(loc_tz)
            if isinstance(cached.get("sun"), dict):
                cached["sun"]["nowMin"] = now_min
            # ``label`` and the duplicate ``place`` are UI strings from
            # the cell editor, not part of the upstream API response.
            # Overlay both on cache hit so a rename on the same
            # coordinates shows up on the next preview instead of
            # waiting for the TTL.
            cached["label"] = label
            cached["place"] = label
            return cached
        except (json.JSONDecodeError, OSError):
            pass

    now = datetime.now(UTC)
    age = _phase_age_days(now)
    fraction = age / _SYNODIC_DAYS  # 0..1 around the synodic cycle
    illum = _illumination(age)
    waxing = age < _SYNODIC_DAYS / 2
    name = _phase_name(age)

    sunrise, sunset, moonrise, moonset, tz_name, offset_s = _fetch_sky(lat, lon)
    # Wall-clock fields follow the location, not the server: Open-Meteo's
    # zone first, then one saved on the location option, else server local.
    if _tz_from_name(tz_name) is None:
        tz_name = _option_tz_name(options)
    loc_tz = _resolve_tz(tz_name, offset_s)

    rise_min = _iso_to_min(sunrise)
    set_min = _iso_to_min(sunset)
    now_min = _now_min(loc_tz)
    day_length = _hhmm_delta(rise_min, set_min)
    solar_noon = _hhmm_mid(rise_min, set_min)

    # Pre-compute the "next major phase" name + ISO for the SUNMOON
    # ``moon.next`` field, this is what the design's compact moon
    # block renders. Picks whichever of the four major phases lands
    # soonest from now.
    upcoming = [
        ("New Moon", _next_phase(now, 0.0)),
        ("First Quarter", _next_phase(now, 0.25)),
        ("Full Moon", _next_phase(now, 0.5)),
        ("Last Quarter", _next_phase(now, 0.75)),
    ]
    next_label, next_dt = min(upcoming, key=lambda pair: pair[1])
    # Shown as a calendar date, so read it in the location's zone.
    next_iso = next_dt.astimezone(loc_tz).isoformat()  # consumed by ``moon.next`` below

    result: dict[str, Any] = {
        # Legacy fields the existing client.js render path still uses.
        "label": label,
        "lat": lat,
        "phase_name": name,
        "age_days": round(age, 1),
        "fraction": round(fraction, 4),  # client uses this to draw the disc
        "illumination": round(illum * 100, 1),  # %
        "waxing": waxing,
        "next_new": _next_phase(now, 0).isoformat(),
        "next_first_quarter": _next_phase(now, 0.25).isoformat(),
        "next_full": _next_phase(now, 0.5).isoformat(),
        "next_last_quarter": _next_phase(now, 0.75).isoformat(),
        "sunrise": sunrise,
        "sunset": sunset,
        "moonrise": moonrise,
        "moonset": moonset,
        "fetched_at": int(time.time()),
        # Location zone for the client's date labels and for re-stamping
        # the wall-clock fields on a cache hit. ``tz`` is an IANA name or
        # None; ``utcOffsetSec`` is the fixed offset when only that is known.
        "tz": tz_name if _tz_from_name(tz_name) is not None else None,
        "utcOffsetSec": offset_s,
        # Structured "SUNMOON" block the new variants paint from.
        "place": label,
        "time": _now_hhmm(loc_tz),
        "rise": _hhmm(sunrise),
        "set": _hhmm(sunset),
        "riseMin": rise_min,
        "setMin": set_min,
        "nowMin": now_min,
        "dayLength": day_length,
        "solarNoon": solar_noon,
        "sun": {
            "rise": _hhmm(sunrise),
            "set": _hhmm(sunset),
            "riseMin": rise_min,
            "setMin": set_min,
            "nowMin": now_min,
            "dayLength": day_length,
            "solarNoon": solar_noon,
        },
        "moon": {
            "phase": name,
            "illum": round(illum * 100),
            "age": round(age, 1),
            "fraction": round(fraction, 4),
            "waxing": waxing,
            "rise": _hhmm(moonrise),
            "set": _hhmm(moonset),
            "next": f"Next {next_label.lower()} · {_short_date(next_iso)}",
        },
    }
    with contextlib.suppress(OSError):
        cache.write_text(json.dumps(result), encoding="utf-8")
    return result


# ----------------------------------------------------------------------
# Helpers, same pattern as weather_now's _iso_to_min / _hhmm / _now_min,
# adapted to also produce day-length + solar-noon strings.
# ----------------------------------------------------------------------


def _iso_to_min(iso: Any) -> int | None:
    """Open-Meteo returns ISO timestamps like ``2026-06-03T06:45`` (no
    seconds, no tz, the API treats them as local once we pass
    ``timezone=auto``). Parse just the hour/minute and convert to
    minutes-since-midnight for the sun-arc charts."""
    if not isinstance(iso, str) or "T" not in iso:
        return None
    try:
        _, t = iso.split("T", 1)
        h, m = t.split(":", 2)[:2]
        return int(h) * 60 + int(m)
    except (ValueError, IndexError):
        return None


def _hhmm(iso: Any) -> str:
    """Trim ``2026-06-03T06:45`` to ``06:45`` for display."""
    if not isinstance(iso, str) or "T" not in iso:
        return ""
    try:
        return iso.split("T", 1)[1][:5]
    except (ValueError, IndexError):
        return ""


def _now_min(tz: tzinfo | None = None) -> int:
    """Wall-clock minutes-since-midnight in ``tz`` (server local if None)."""
    n = datetime.now(tz)
    return n.hour * 60 + n.minute


def _now_hhmm(tz: tzinfo | None = None) -> str:
    """Wall-clock HH:MM in ``tz`` for the SUNMOON ``time`` field."""
    n = datetime.now(tz)
    return f"{n.hour:02d}:{n.minute:02d}"


def _hhmm_delta(rise_min: int | None, set_min: int | None) -> str:
    """Format set - rise as ``HHh MMm`` (e.g. ``10h 23m``)."""
    if rise_min is None or set_min is None:
        return ""
    span = max(0, set_min - rise_min)
    h, m = divmod(span, 60)
    return f"{h}h {m:02d}m"


def _hhmm_mid(rise_min: int | None, set_min: int | None) -> str:
    """Format the midpoint of rise..set as HH:MM (solar noon)."""
    if rise_min is None or set_min is None:
        return ""
    mid = (rise_min + set_min) // 2
    h, m = divmod(mid, 60)
    return f"{h:02d}:{m:02d}"


def _short_date(iso: Any) -> str:
    """``2026-06-12T…`` → ``Fri 12 Jun`` (for the moon.next label).
    Built from format codes that work cross-platform (Windows doesn't
    support ``%-d``)."""
    if not isinstance(iso, str):
        return ""
    try:
        d = datetime.fromisoformat(iso)
    except ValueError:
        return ""
    return f"{d.strftime('%a')} {d.day} {d.strftime('%b')}"
