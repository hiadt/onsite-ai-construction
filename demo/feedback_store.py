"""Local, append-only validation feedback storage for the PathGuard demo."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def database_path() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "PathGuard"
    base.mkdir(parents=True, exist_ok=True)
    return base / "validation_history.sqlite3"


def _connect() -> sqlite3.Connection:
    con = sqlite3.connect(database_path(), timeout=10)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""CREATE TABLE IF NOT EXISTS validation_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at_utc TEXT NOT NULL,
        sample_id TEXT NOT NULL,
        route_sha256 TEXT,
        map_id TEXT,
        vehicle_structure TEXT,
        vehicle_config_sha256 TEXT,
        feature_version TEXT,
        schema_sha256 TEXT,
        model_name TEXT,
        model_risk REAL,
        rule_risk REAL,
        risk_level TEXT,
        queue_reason TEXT,
        geometry_state TEXT,
        geometry_min_margin_m REAL,
        validation_result TEXT NOT NULL,
        validation_method TEXT NOT NULL,
        note TEXT,
        extra_json TEXT NOT NULL
    )""")
    return con


def save_feedback(record: dict[str, Any]) -> int:
    """Persist only the submitted review result, never historical training labels."""
    config = record.pop("vehicle_config", {})
    config_json = json.dumps(config, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    config_hash = hashlib.sha256(config_json.encode("utf-8")).hexdigest() if config else ""
    values = (
        datetime.now(timezone.utc).isoformat(timespec="seconds"),
        str(record.get("sample_id", "")), str(record.get("route_sha256", "")),
        str(record.get("map_id", "")), str(record.get("vehicle_structure", "unknown")),
        config_hash, str(record.get("feature_version", "")), str(record.get("schema_sha256", "")),
        str(record.get("model_name", "")), _optional_float(record.get("model_risk")),
        _optional_float(record.get("rule_risk")), str(record.get("risk_level", "")),
        str(record.get("queue_reason", "")), str(record.get("geometry_state", "未评估")),
        _optional_float(record.get("geometry_min_margin_m")),
        str(record.get("validation_result", "")), str(record.get("validation_method", "")),
        str(record.get("note", "")), config_json,
    )
    with closing(_connect()) as con:
        cursor = con.execute("""INSERT INTO validation_feedback (
            created_at_utc,sample_id,route_sha256,map_id,vehicle_structure,
            vehicle_config_sha256,feature_version,schema_sha256,model_name,model_risk,
            rule_risk,risk_level,queue_reason,geometry_state,geometry_min_margin_m,
            validation_result,validation_method,note,extra_json
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""", values)
        con.commit()
        return int(cursor.lastrowid)


def load_feedback(limit: int = 500) -> pd.DataFrame:
    with closing(_connect()) as con:
        return pd.read_sql_query(
            "SELECT * FROM validation_feedback ORDER BY id DESC LIMIT ?", con, params=(int(limit),)
        )


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if pd.notna(result) else None
    except (TypeError, ValueError):
        return None
