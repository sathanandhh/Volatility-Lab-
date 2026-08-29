-- ─────────────────────────────────────────────────────────────────────────────
-- Volatility MCP — Session Store Initial Schema
-- Backend: SQLite (default) or DuckDB (set VOLMCP_SESSION_BACKEND=duckdb)
-- ─────────────────────────────────────────────────────────────────────────────

-- Sessions table: one row per analytics session
CREATE TABLE IF NOT EXISTS sessions (
    id                 TEXT PRIMARY KEY,
    created_at         TEXT NOT NULL,
    updated_at         TEXT NOT NULL,
    current_stage      TEXT NOT NULL DEFAULT 'empty',
    metadata           TEXT NOT NULL DEFAULT '{}',
    last_fit_config    TEXT,
    comparison_result  TEXT,
    coverage_scorecard TEXT,
    is_restored        INTEGER NOT NULL DEFAULT 0
);

-- Returns table: parquet-encoded returns series (one per session)
CREATE TABLE IF NOT EXISTS returns (
    session_id    TEXT PRIMARY KEY,
    parquet_blob  BLOB,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Decision log: append-only audit trail
CREATE TABLE IF NOT EXISTS decision_log (
    session_id  TEXT NOT NULL,
    entry_id    INTEGER NOT NULL,
    timestamp   TEXT NOT NULL,
    action      TEXT NOT NULL,
    target      TEXT NOT NULL DEFAULT '',
    outcome     TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (session_id, entry_id),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Fitted models: one row per (session, model_name)
-- Note: the underlying arch fit object is NOT persisted — only the
-- public attributes (params, conditional_volatility, std_resid, etc.).
-- Forecasting from a restored fit requires re-fitting.
CREATE TABLE IF NOT EXISTS fitted_models (
    session_id        TEXT NOT NULL,
    model_name        TEXT NOT NULL,
    name              TEXT,
    params_json       TEXT NOT NULL DEFAULT '{}',
    cond_vol_parquet  BLOB,
    std_resid_parquet BLOB,
    loglikelihood     REAL,
    aic               REAL,
    bic               REAL,
    n_params          INTEGER,
    converged         INTEGER,
    convergence_flag  INTEGER,
    next_vol          REAL,
    PRIMARY KEY (session_id, model_name),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Optimization results: one row per (session, kind)
CREATE TABLE IF NOT EXISTS optimization_results (
    session_id    TEXT NOT NULL,
    kind          TEXT NOT NULL,
    optimal_json  TEXT NOT NULL,
    optimal_score REAL,
    criterion     TEXT,
    sweep_json    TEXT NOT NULL DEFAULT '[]',
    recommendation TEXT,
    PRIMARY KEY (session_id, kind),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Risk results: one row per (session, result_key)
CREATE TABLE IF NOT EXISTS risk_results (
    session_id  TEXT NOT NULL,
    result_key  TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (session_id, result_key),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Diagnostics results: one row per (session, model_name)
CREATE TABLE IF NOT EXISTS diagnostics_results (
    session_id  TEXT NOT NULL,
    model_name  TEXT NOT NULL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (session_id, model_name),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Backtest results: one row per (session, model_name)
-- Series stored as parquet blobs for efficiency
CREATE TABLE IF NOT EXISTS backtest_results (
    session_id        TEXT NOT NULL,
    model_name        TEXT NOT NULL,
    returns_parquet   BLOB,
    variance_parquet  BLOB,
    var_parquet       BLOB,
    violations_parquet BLOB,
    PRIMARY KEY (session_id, model_name),
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sessions_stage ON sessions(current_stage);
CREATE INDEX IF NOT EXISTS idx_decisions_session ON decision_log(session_id);
CREATE INDEX IF NOT EXISTS idx_fitted_session ON fitted_models(session_id);
CREATE INDEX IF NOT EXISTS idx_optimization_session ON optimization_results(session_id);
CREATE INDEX IF NOT EXISTS idx_risk_session ON risk_results(session_id);
CREATE INDEX IF NOT EXISTS idx_diagnostics_session ON diagnostics_results(session_id);
CREATE INDEX IF NOT EXISTS idx_backtest_session ON backtest_results(session_id);
