"""
记忆库 (Memory Store)：基于 SQLite 的持久化存储，覆盖两类记忆。
1. 长期记忆 —— analysis_runs：每次分析的结论、评分、风控结果、数据质量、完整研报与 trace。
   下次分析同一标的时自动召回，用于“与上次结论对比”，并支持按 run_id 复盘。
2. 会话记忆 —— chat_messages / chat_summaries：追问对话的逐条消息及其压缩摘要，
   服务重启后会话不丢失，客户端只需带 session_id，不必再自己维护/截断历史。
只用标准库 sqlite3；写操作加锁，单进程多线程安全(异步代码通过 asyncio.to_thread 调用)。
"""

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    run_id TEXT PRIMARY KEY,
    session_id TEXT,
    ticker TEXT NOT NULL,
    target_name TEXT,
    created_at REAL NOT NULL,
    strategy_style TEXT,
    verdict TEXT,
    consensus_score REAL,
    position TEXT,
    risk_status TEXT,
    is_vetoed INTEGER,
    data_quality TEXT,
    summary TEXT,
    report_md TEXT,
    trace TEXT
);
CREATE INDEX IF NOT EXISTS idx_runs_ticker_time ON analysis_runs(ticker, created_at DESC);
CREATE TABLE IF NOT EXISTS chat_messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    ticker TEXT,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    ts REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_msg_session ON chat_messages(session_id, id);
CREATE TABLE IF NOT EXISTS chat_summaries (
    session_id TEXT PRIMARY KEY,
    summary TEXT NOT NULL,
    upto_id INTEGER NOT NULL,
    updated REAL NOT NULL
);
"""

_RUN_LIGHT_COLS = ("run_id, session_id, ticker, target_name, created_at, strategy_style, verdict, "
                   "consensus_score, position, risk_status, is_vetoed, data_quality, summary")


class MemoryStore:
    def __init__(self, db_path: str = ":memory:"):
        if db_path != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ------------------------------------------------------------------ 长期记忆
    def save_run(self, run: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO analysis_runs VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    run["run_id"], run.get("session_id"), run["ticker"], run.get("target_name"),
                    run.get("created_at", time.time()), run.get("strategy_style"), run.get("verdict"),
                    run.get("consensus_score"), run.get("position"), run.get("risk_status"),
                    int(bool(run.get("is_vetoed"))), json.dumps(run.get("data_quality", {}), ensure_ascii=False),
                    run.get("summary", ""), run.get("report_md", ""),
                    json.dumps(run.get("trace", {}), ensure_ascii=False),
                ),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
        d = dict(row)
        for k in ("data_quality", "trace"):
            if k in d and isinstance(d[k], str):
                try:
                    d[k] = json.loads(d[k])
                except ValueError:
                    d[k] = {}
        return d

    def recent_runs(self, ticker: str, limit: int = 3, exclude_run_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """同一标的最近的若干次分析(不含研报全文与 trace，供召回/对比使用)"""
        with self._lock:
            rows = self._conn.execute(
                f"SELECT {_RUN_LIGHT_COLS} FROM analysis_runs WHERE ticker=? AND run_id IS NOT ? "
                "ORDER BY created_at DESC LIMIT ?",
                (ticker, exclude_run_id, limit),
            ).fetchall()
        return [self._row_to_dict(r) for r in rows]

    def latest_run(self, ticker: str) -> Optional[Dict[str, Any]]:
        runs = self.recent_runs(ticker, limit=1)
        return runs[0] if runs else None

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM analysis_runs WHERE run_id=?", (run_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_runs(self, ticker: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        sql = f"SELECT {_RUN_LIGHT_COLS} FROM analysis_runs"
        args: List[Any] = []
        if ticker:
            sql += " WHERE ticker=?"
            args.append(ticker)
        sql += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [self._row_to_dict(r) for r in rows]

    # ------------------------------------------------------------------ 会话记忆
    def add_message(self, session_id: str, role: str, content: str, ticker: Optional[str] = None) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO chat_messages(session_id, ticker, role, content, ts) VALUES (?,?,?,?,?)",
                (session_id, ticker, role, content, time.time()),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def get_messages(self, session_id: str, after_id: int = 0) -> List[Dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, role, content, ts FROM chat_messages WHERE session_id=? AND id>? ORDER BY id",
                (session_id, after_id),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute(
                "SELECT summary, upto_id FROM chat_summaries WHERE session_id=?", (session_id,)
            ).fetchone()
        return dict(row) if row else None

    def set_summary(self, session_id: str, summary: str, upto_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO chat_summaries VALUES (?,?,?,?)", (session_id, summary, upto_id, time.time())
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
