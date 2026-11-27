"""Minimal checker for the seven single-limit rules of rail-standard-shift-bench.

It does two things and nothing else:
  1. compile_condition(): turn the JSON condition written by a language model into one limit value for the
     project's cell, or say why it cannot be used (abstain / invalid);
  2. verdict(): judge a candidate design value against a limit value ('적합' = complies, '위반' = violates).

The published scores were produced with the authors' full verification engine (not released). For the seven
rules of this benchmark the engine's verdict is a single inequality, and this file reproduces it exactly:
540 of 540 gold verdicts and every engine-scored model verdict in results/scores_p3.json
(run `python checker/reproduce.py`).

No dependencies beyond the Python standard library (Python 3.9+).
"""
from __future__ import annotations

import json
import math
import re

COMPLY, VIOLATE, HOLD = "적합", "위반", "보류"

# Rules in scope. dir "min": a design value x complies if x >= limit; "max": complies if x <= limit.
# Articles refer to the Korean notice 「철도의 건설기준에 관한 규정」 (Rules on Railway Construction Standards).
RULES = {
    "R-06-01": dict(label="본선 최소 곡선반경", label_en="minimum curve radius of main track", dir="min", unit="m",
                    article="제6조제1항", track_dep=True),
    "R-07-01": dict(label="최대 설정캔트", label_en="maximum applied cant", dir="max", unit="mm",
                    article="제7조제1항", track_dep=True),
    "R-09-01": dict(label="직선 및 원곡선의 최소 길이", label_en="minimum length of straight and circular curve", dir="min", unit="m",
                    article="제9조", track_dep=False),
    "R-10-01": dict(label="본선 최대 기울기", label_en="maximum gradient of main track", dir="max", unit="‰",
                    article="제10조제1항", track_dep=False),
    "R-11-02": dict(label="최소 종곡선 반경", label_en="minimum vertical curve radius", dir="min", unit="m",
                    article="제11조제2항", track_dep=True),
    "R-11-04": dict(label="종곡선 반경 상한", label_en="upper limit of vertical curve radius", dir="max", unit="m",
                    article="제11조제2항 표 ㈜ 단서", track_dep=True),
    "R-11-05": dict(label="종곡선 최소 연장", label_en="minimum vertical curve length", dir="min", unit="m",
                    article="제11조제4항", track_dep=False),
}

KO_TRACK = {"자갈도상": "ballast", "자갈": "ballast", "ballast": "ballast",
            "콘크리트도상": "slab", "콘크리트": "slab", "slab": "slab"}
ANY_TRACK = {"무관", "any", "공통", "전체", "모두", "both", ""}


def verdict(direction: str, x: float, limit: float) -> str:
    """Verdict for design value x under one limit value."""
    if direction == "min":
        return COMPLY if x >= limit else VIOLATE
    if direction == "max":
        return COMPLY if x <= limit else VIOLATE
    raise ValueError(direction)


# ---------------------------------------------------------------------------------------------------------------
# Compiler: model JSON condition -> one limit value for the project's cell (or a reason why it cannot be used)
# ---------------------------------------------------------------------------------------------------------------
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def extract_json(text):
    """First balanced {...} block of a model output (code fences tolerated). Returns a dict or None."""
    if text is None:
        return None
    s = text.strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    start = s.find("{")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except Exception:
                        break
        start = s.find("{", start + 1)
    return None


def _num(v):
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(float(v)) else None
    if isinstance(v, str):
        m = _NUM.search(v.replace(" ", ""))
        if m:
            return float(m.group(0).replace(",", ""))
    return None


def _op(v):
    s = str(v or "").strip()
    if s in (">=", "≥", "이상", "=>", "min", "최소"):
        return "min"
    if s in ("<=", "≤", "이하", "=<", "max", "최대"):
        return "max"
    return None


def _unit_factor(rule_unit, u):
    u = re.sub(r"\(.*?\)", "", str(u or "")).strip().lower().replace(" ", "")
    if rule_unit == "m":
        return {"m": 1.0, "미터": 1.0, "km": 1000.0, "킬로미터": 1000.0, "mm": 0.001, "밀리미터": 0.001}.get(u)
    if rule_unit == "mm":
        return {"mm": 1.0, "밀리미터": 1.0, "m": 1000.0, "미터": 1000.0}.get(u)
    if rule_unit == "‰":
        return {"‰": 1.0, "퍼밀": 1.0, "permil": 1.0, "permille": 1.0, "‰(천분율)": 1.0, "천분율": 1.0,
                "1/1000": 1.0, "1천분의": 1.0, "%": 10.0}.get(u)
    return None


def compile_condition(item, obj):
    """Returns (status, value, reason); status is 'ok', 'abstain' or 'invalid'.

    'ok'      -> the condition can be used: `value` is the limit for the project's cell, in the rule's unit.
    'abstain' -> the model set "abstain": true.
    'invalid' -> the condition cannot be used (no JSON, no value, unknown unit, wrong direction of the inequality,
                 design speed or track type that is not the project's). In the primary analysis an invalid
                 condition is routed to the designer, i.e. counted as a hold, not as a verdict.
    """
    if obj is None:
        return ("invalid", None, "no_json")
    if not isinstance(obj, dict):
        return ("invalid", None, "not_object")
    if obj.get("abstain") in (True, "true", "True", "예", "yes"):
        return ("abstain", None, "model_abstain")
    R = RULES[item["rule"]]
    val = _num(obj.get("value"))
    if val is None:
        return ("invalid", None, "no_value")
    f = _unit_factor(R["unit"], obj.get("unit"))
    if f is None:
        return ("invalid", None, "unit")
    val = val * f
    if _op(obj.get("op")) != R["dir"]:
        return ("invalid", None, "op")
    ap = obj.get("applies") or {}
    if not isinstance(ap, dict):
        ap = {}
    sp = _num(ap.get("design_speed_kmh"))
    if sp is not None and int(round(sp)) != item["V"]:
        return ("invalid", None, "applies_speed")
    tr = str(ap.get("track", "") or "").strip()
    if R["track_dep"]:
        if tr in ANY_TRACK or KO_TRACK.get(tr) != item["track"]:
            return ("invalid", None, "applies_track")
    elif tr not in ANY_TRACK and KO_TRACK.get(tr) not in (None, item["track"]):
        return ("invalid", None, "applies_track")
    if not (val > 0 and math.isfinite(val)):
        return ("invalid", None, "nonpositive")
    return ("ok", val, "")
