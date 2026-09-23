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


def make_condition_key(route_sha256: str, environment_version: str,
                       route_version: str, vehicle_config: dict) -> str:
    config_hash = hashlib.sha256(json.dumps(vehicle_config, sort_keys=True, ensure_ascii=False,
                                            separators=(",", ":")).encode("utf-8")).hexdigest()
    return hashlib.sha256(json.dumps({
        "route_sha256": str(route_sha256), "environment_version": str(environment_version),
        "route_version": str(route_version), "vehicle_config_sha256": config_hash,
    }, sort_keys=True).encode("utf-8")).hexdigest()


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
    existing = {row[1] for row in con.execute("PRAGMA table_info(validation_feedback)")}
    for column in ("task_id", "environment_version", "route_version", "severity",
                   "evidence_ref", "evidence_completeness", "condition_consistency",
                   "review_status", "training_review", "effective_config_json", "condition_key",
                   "evidence_level"):
        if column not in existing:
            con.execute(f"ALTER TABLE validation_feedback ADD COLUMN {column} TEXT")
    return con


def save_feedback(record: dict[str, Any]) -> int:
    """Persist only the submitted review result, never historical training labels."""
    config = record.get("vehicle_config", {})
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
        con.execute("""UPDATE validation_feedback SET
            task_id=?, environment_version=?, route_version=?, severity=?, evidence_ref=?,
            evidence_completeness=?, condition_consistency=?, review_status=?,
            training_review=?, effective_config_json=?, condition_key=?, evidence_level=? WHERE id=?""", (
            str(record.get("task_id", "")), str(record.get("environment_version", "")),
            str(record.get("route_version", "")), str(record.get("severity", "未分级")),
            str(record.get("evidence_ref", "")), str(record.get("evidence_completeness", "未核对")),
            str(record.get("condition_consistency", "未核对")), str(record.get("review_status", "待复核")),
            "待审核", config_json,
            make_condition_key(record.get("route_sha256", ""), record.get("environment_version", ""),
                               record.get("route_version", ""), config),
            evidence_level(record), int(cursor.lastrowid)))
        condition_key = make_condition_key(record.get("route_sha256", ""),
                                           record.get("environment_version", ""),
                                           record.get("route_version", ""), config)
        conflicting = con.execute("""SELECT COUNT(DISTINCT validation_result) FROM validation_feedback
            WHERE condition_key=? AND condition_consistency='与本次评估一致'
            AND validation_result IN ('通过','失败')""", (condition_key,)).fetchone()[0]
        if conflicting > 1:
            con.execute("UPDATE validation_feedback SET training_review='待审核' WHERE condition_key=?",
                        (condition_key,))
        con.commit()
        return int(cursor.lastrowid)


def load_feedback(limit: int = 500) -> pd.DataFrame:
    with closing(_connect()) as con:
        return pd.read_sql_query(
            "SELECT * FROM validation_feedback ORDER BY id DESC LIMIT ?", con, params=(int(limit),)
        )


def classify_disagreement(records: pd.DataFrame, model_risk: float | None,
                          current_condition_key: str) -> str:
    """Keep prediction and observations separate; flag only comparable outcomes."""
    if records.empty:
        return "尚无验证记录"
    valid = records[records["validation_result"].isin(["通过", "失败"])].copy()
    if valid.empty:
        return "验证进行中"
    comparable = valid[(valid["condition_key"].fillna("") == current_condition_key) &
                       (valid["condition_consistency"].fillna("") == "与本次评估一致")]
    if comparable.empty:
        return "已有不同环境条件下的结果，请分别查看"
    if comparable["validation_result"].nunique() > 1:
        return "同环境验证结果冲突：待复核，暂不生成单一训练标签"
    outcome = comparable.iloc[0]["validation_result"]
    if model_risk is not None and pd.notna(model_risk):
        if model_risk < 0.45 and outcome == "失败":
            return "模型低风险但验证失败：优先排查漏判"
        if model_risk >= 0.70 and outcome == "通过":
            return "模型高风险但验证通过：检查条件与模型保守性"
    return "预测与当前验证记录未见明显冲突"


def set_training_review(record_id: int, status: str) -> None:
    if status not in {"待审核", "可用于训练候选", "暂不采用"}:
        raise ValueError("训练审核状态无效。")
    with closing(_connect()) as con:
        row = con.execute("SELECT validation_result,review_status,condition_key,condition_consistency FROM validation_feedback WHERE id=?",
                          (int(record_id),)).fetchone()
        if row is None:
            raise ValueError("验证记录不存在。")
        if status == "可用于训练候选":
            if row[0] not in {"通过", "失败"} or row[1] != "已核对" or row[3] != "与本次评估一致":
                raise ValueError("只有条件一致、已核对且结果明确的记录可进入训练候选。")
            others = con.execute("""SELECT COUNT(DISTINCT validation_result) FROM validation_feedback
                WHERE condition_key=? AND condition_consistency='与本次评估一致'
                AND validation_result IN ('通过','失败')""", (row[2],)).fetchone()[0]
            if others > 1:
                raise ValueError("同条件存在相互矛盾的验证结果，请先复核。")
        con.execute("UPDATE validation_feedback SET training_review=? WHERE id=?", (status, int(record_id)))
        con.commit()


def _optional_float(value: Any) -> float | None:
    try:
        result = float(value)
        return result if pd.notna(result) else None
    except (TypeError, ValueError):
        return None


def evidence_level(record: dict[str, Any]) -> str:
    """Describe reviewability, without inventing a universal confidence percentage."""
    if record.get("condition_consistency") != "与本次评估一致":
        return "条件待核对或不同"
    if record.get("review_status") != "已核对":
        return "提交者结论待复核"
    if record.get("evidence_completeness") == "有报告或日志" and record.get("evidence_ref"):
        return "已核对且证据可复查"
    return "已核对但证据待补"
