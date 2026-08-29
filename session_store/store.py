"""SQLite-backed persistent session store.

Implements the same interface as core.feedback.session.SessionStore
plus a `persist(session)` method that writes the full session state to
SQLite. Returns series are stored as parquet blobs; everything else as
JSON.

Usage:
    from session_store import PersistentSessionStore
    store = PersistentSessionStore(db_path="/data/volmcp.db")
    sid = store.open()
    session = store.get(sid)
    # ... mutate session ...
    store.persist(session)  # write through

The store also maintains an in-memory cache so that repeated `get()`
calls within the same process are fast. On cache miss, the session is
hydrated from SQLite.
"""
from __future__ import annotations

import io
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from core.config import get_settings
from core.feedback.session import Session, SessionStore
from core.feedback.decision_log import DecisionLogEntry

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


class PersistentSessionStore(SessionStore):
    """In-memory cache + SQLite persistence.

    Drop-in replacement for SessionStore. Every `open()` creates a row
    in the database; `persist(session)` writes through; `get(sid)`
    hydrates from cache or database.
    """

    def __init__(self, db_path: str | None = None) -> None:
        super().__init__()  # init the in-memory dict cache
        settings = get_settings()
        self.db_path = db_path or settings.session_db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    # ── Database setup ─────────────────────────────────────────────
    def _init_db(self) -> None:
        sql = (_MIGRATIONS_DIR / "001_init.sql").read_text(encoding="utf-8")
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(sql)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    # ── SessionStore interface ──────────────────────────────────────
    def open(self) -> str:
        sid = str(uuid.uuid4())
        now = datetime.utcnow().isoformat() + "Z"
        session = Session(id=sid)
        session.created_at = now
        self._sessions[sid] = session
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, created_at, updated_at, "
                "current_stage, metadata, is_restored) "
                "VALUES (?, ?, ?, ?, ?, 0)",
                (sid, now, now, "empty", "{}"),
            )
        return sid

    def get(self, sid: str) -> Session:
        if sid in self._sessions:
            return self._sessions[sid]
        session = self._hydrate(sid)
        self._sessions[sid] = session
        return session

    def close(self, sid: str) -> None:
        self._sessions.pop(sid, None)
        with self._conn() as conn:
            conn.execute("DELETE FROM sessions WHERE id = ?", (sid,))

    def list_ids(self) -> list[str]:
        with self._conn() as conn:
            rows = conn.execute("SELECT id FROM sessions ORDER BY created_at DESC").fetchall()
        return [r[0] for r in rows]

    # ── Persistence ────────────────────────────────────────────────
    def persist(self, session: Session) -> None:
        """Write the entire session state to the database."""
        self._sessions[session.id] = session  # update cache
        now = datetime.utcnow().isoformat() + "Z"
        with self._conn() as conn:
            self._upsert_session(conn, session, now)
            self._persist_returns(conn, session)
            self._persist_decisions(conn, session)
            self._persist_fitted_models(conn, session)
            self._persist_optimization(conn, session)
            self._persist_risk(conn, session)
            self._persist_diagnostics(conn, session)
            self._persist_backtest(conn, session)

    # ── Upsert helpers ──────────────────────────────────────────────
    @staticmethod
    def _upsert_session(conn, session: Session, now: str) -> None:
        conn.execute(
            "UPDATE sessions SET updated_at = ?, current_stage = ?, "
            "metadata = ?, last_fit_config = ?, comparison_result = ?, "
            "coverage_scorecard = ?, is_restored = 0 WHERE id = ?",
            (
                now,
                session.current_stage,
                json.dumps(session.metadata, default=str),
                json.dumps(session.last_fit_config, default=str)
                    if session.last_fit_config else None,
                json.dumps(session.comparison_result, default=str)
                    if session.comparison_result else None,
                json.dumps(session.coverage_scorecard, default=str)
                    if session.coverage_scorecard else None,
                session.id,
            ),
        )

    @staticmethod
    def _persist_returns(conn, session: Session) -> None:
        if session.returns is None:
            return
        blob = PersistentSessionStore._to_parquet(session.returns)
        conn.execute(
            "INSERT OR REPLACE INTO returns (session_id, parquet_blob) "
            "VALUES (?, ?)",
            (session.id, blob),
        )

    @staticmethod
    def _persist_decisions(conn, session: Session) -> None:
        conn.execute("DELETE FROM decision_log WHERE session_id = ?",
                     (session.id,))
        for e in session.decision_log.entries:
            conn.execute(
                "INSERT INTO decision_log (session_id, entry_id, timestamp, "
                "action, target, outcome, detail) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session.id, e.id, e.timestamp, e.action, e.target,
                 e.outcome, e.detail),
            )

    @staticmethod
    def _persist_fitted_models(conn, session: Session) -> None:
        conn.execute("DELETE FROM fitted_models WHERE session_id = ?",
                     (session.id,))
        for name, fit in session.fitted_models.items():
            params_json = json.dumps(
                {k: float(v) for k, v in fit.params.items()}, default=str
            )
            cv_blob = PersistentSessionStore._to_parquet(fit.conditional_volatility) \
                if fit.conditional_volatility is not None else None
            sr_blob = PersistentSessionStore._to_parquet(fit.std_resid) \
                if fit.std_resid is not None else None
            conn.execute(
                "INSERT INTO fitted_models (session_id, model_name, name, "
                "params_json, cond_vol_parquet, std_resid_parquet, "
                "loglikelihood, aic, bic, n_params, converged, "
                "convergence_flag, next_vol) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (session.id, name, fit.name, params_json,
                 cv_blob, sr_blob,
                 float(fit.loglikelihood), float(fit.aic), float(fit.bic),
                 int(fit.n_params), int(fit.converged),
                 int(fit.convergence_flag), float(fit.next_vol)),
            )

    @staticmethod
    def _persist_optimization(conn, session: Session) -> None:
        conn.execute("DELETE FROM optimization_results WHERE session_id = ?",
                     (session.id,))
        for kind, opt in session.optimization_results.items():
            optimal_json = json.dumps(opt.optimal, default=str)
            sweep_json = json.dumps(opt.sweep_table, default=str)
            conn.execute(
                "INSERT INTO optimization_results (session_id, kind, "
                "optimal_json, optimal_score, criterion, sweep_json, "
                "recommendation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session.id, kind, optimal_json,
                 float(opt.optimal_score), opt.criterion,
                 sweep_json, opt.recommendation),
            )

    @staticmethod
    def _persist_risk(conn, session: Session) -> None:
        conn.execute("DELETE FROM risk_results WHERE session_id = ?",
                     (session.id,))
        for key, val in session.risk_results.items():
            conn.execute(
                "INSERT INTO risk_results (session_id, result_key, "
                "result_json) VALUES (?, ?, ?)",
                (session.id, key, json.dumps(val, default=str)),
            )

    @staticmethod
    def _persist_diagnostics(conn, session: Session) -> None:
        conn.execute("DELETE FROM diagnostics_results WHERE session_id = ?",
                     (session.id,))
        for name, diag in session.diagnostics_results.items():
            conn.execute(
                "INSERT INTO diagnostics_results (session_id, model_name, "
                "result_json) VALUES (?, ?, ?)",
                (session.id, name, json.dumps(diag, default=str)),
            )

    @staticmethod
    def _persist_backtest(conn, session: Session) -> None:
        conn.execute("DELETE FROM backtest_results WHERE session_id = ?",
                     (session.id,))
        for name, bt in session.backtest_results.items():
            conn.execute(
                "INSERT INTO backtest_results (session_id, model_name, "
                "returns_parquet, variance_parquet, var_parquet, "
                "violations_parquet) VALUES (?, ?, ?, ?, ?, ?)",
                (session.id, name,
                 PersistentSessionStore._to_parquet(bt.returns) if bt.returns is not None else None,
                 PersistentSessionStore._to_parquet(bt.variance) if bt.variance is not None else None,
                 PersistentSessionStore._to_parquet(bt.var) if bt.var is not None else None,
                 PersistentSessionStore._to_parquet(bt.violations) if bt.violations is not None else None),
            )

    # ── Hydration ──────────────────────────────────────────────────
    def _hydrate(self, sid: str) -> Session:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT id, created_at, current_stage, metadata, "
                "last_fit_config, comparison_result, coverage_scorecard "
                "FROM sessions WHERE id = ?",
                (sid,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown session: {sid!r}")
            session = Session(id=row[0])
            session.created_at = row[1]
            session.current_stage = row[2]
            session.metadata = json.loads(row[3] or "{}")
            session.last_fit_config = json.loads(row[4]) if row[4] else None
            session.comparison_result = json.loads(row[5]) if row[5] else None
            session.coverage_scorecard = json.loads(row[6]) if row[6] else None

            # Returns
            ret_row = conn.execute(
                "SELECT parquet_blob FROM returns WHERE session_id = ?",
                (sid,),
            ).fetchone()
            if ret_row and ret_row[0]:
                session.returns = self._from_parquet(ret_row[0])

            # Decision log
            for e in conn.execute(
                "SELECT entry_id, timestamp, action, target, outcome, detail "
                "FROM decision_log WHERE session_id = ? ORDER BY entry_id",
                (sid,),
            ).fetchall():
                session.decision_log.entries.append(DecisionLogEntry(
                    id=e[0], timestamp=e[1], action=e[2],
                    target=e[3], outcome=e[4], detail=e[5],
                ))
                session.decision_log._counter = max(
                    session.decision_log._counter, e[0]
                )

            # Fitted models
            for f in conn.execute(
                "SELECT model_name, name, params_json, cond_vol_parquet, "
                "std_resid_parquet, loglikelihood, aic, bic, n_params, "
                "converged, convergence_flag, next_vol "
                "FROM fitted_models WHERE session_id = ?",
                (sid,),
            ).fetchall():
                from core.models.univariate.arch_family import FitResult
                fit = FitResult(
                    name=f[1] or f[0],
                    params=json.loads(f[2] or "{}"),
                    conditional_volatility=self._from_parquet(f[3])
                        if f[3] else pd.Series(dtype=float),
                    std_resid=self._from_parquet(f[4])
                        if f[4] else pd.Series(dtype=float),
                    loglikelihood=f[5] or 0.0,
                    aic=f[6] or 0.0,
                    bic=f[7] or 0.0,
                    n_params=f[8] or 0,
                    converged=bool(f[9]),
                    convergence_flag=f[10] or 0,
                    next_vol=f[11] or 0.0,
                    extras={},  # _arch_fit not persisted
                )
                session.fitted_models[f[0]] = fit

            # Optimization results
            for o in conn.execute(
                "SELECT kind, optimal_json, optimal_score, criterion, "
                "sweep_json, recommendation FROM optimization_results "
                "WHERE session_id = ?",
                (sid,),
            ).fetchall():
                from core.optimize.optimizer import OptimizationResult
                opt = OptimizationResult(
                    kind=o[0],
                    optimal=json.loads(o[1]) if o[1] else None,
                    optimal_score=o[2] or 0.0,
                    criterion=o[3] or "",
                    sweep_table=json.loads(o[4]) if o[4] else [],
                    recommendation=o[5] or "",
                )
                session.optimization_results[o[0]] = opt

            # Risk results
            for r in conn.execute(
                "SELECT result_key, result_json FROM risk_results "
                "WHERE session_id = ?",
                (sid,),
            ).fetchall():
                session.risk_results[r[0]] = json.loads(r[1])

            # Diagnostics
            for d in conn.execute(
                "SELECT model_name, result_json FROM diagnostics_results "
                "WHERE session_id = ?",
                (sid,),
            ).fetchall():
                session.diagnostics_results[d[0]] = json.loads(d[1])

            # Backtest results
            for b in conn.execute(
                "SELECT model_name, returns_parquet, variance_parquet, "
                "var_parquet, violations_parquet FROM backtest_results "
                "WHERE session_id = ?",
                (sid,),
            ).fetchall():
                from core.backtest.rolling import ModelBacktestResult
                bt = ModelBacktestResult(
                    model=b[0],
                    returns=self._from_parquet(b[1])
                        if b[1] else pd.Series(dtype=float),
                    variance=self._from_parquet(b[2])
                        if b[2] else pd.Series(dtype=float),
                    var=self._from_parquet(b[3])
                        if b[3] else pd.Series(dtype=float),
                    violations=self._from_parquet(b[4])
                        if b[4] else pd.Series(dtype=bool),
                )
                session.backtest_results[b[0]] = bt

        return session

    # ── Parquet helpers ────────────────────────────────────────────
    @staticmethod
    def _to_parquet(series: pd.Series) -> bytes:
        buf = io.BytesIO()
        series.to_frame(name=series.name or "value").to_parquet(
            buf, compression="zstd"
        )
        return buf.getvalue()

    @staticmethod
    def _from_parquet(blob: bytes) -> pd.Series:
        df = pd.read_parquet(io.BytesIO(blob))
        return df.iloc[:, 0]
