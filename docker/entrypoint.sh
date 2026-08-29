#!/bin/bash
# ── Volatility MCP — Docker Entrypoint ──────────────────────────────
#
# Runs database migrations (if using SQLite/DuckDB backend) and then
# starts the MCP server with the specified transport.
set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " Volatility MCP Server"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo " Python:        $(python --version 2>&1)"
echo " Working dir:   $(pwd)"
echo " Session backend: ${VOLMCP_SESSION_BACKEND:-memory}"
echo " Report dir:    ${VOLMCP_REPORT_DIR:-/tmp/volmcp_reports}"
echo ""

# Ensure directories exist
mkdir -p "${VOLMCP_REPORT_DIR:-/tmp/volmcp_reports}"
mkdir -p "$(dirname "${VOLMCP_SESSION_DB:-/tmp/volmcp_sessions.db}")"

# If using SQLite backend, initialize the database
if [ "${VOLMCP_SESSION_BACKEND:-memory}" = "sqlite" ]; then
    echo "Initializing SQLite session store at ${VOLMCP_SESSION_DB}..."
    python -c "
from session_store.store import PersistentSessionStore
store = PersistentSessionStore(db_path='${VOLMCP_SESSION_DB:-/tmp/volmcp_sessions.db}')
print('  ✓ Database initialized')
store.close(store.open())  # create + immediately close a test session
" || echo "  ⚠ Database initialization skipped (will create on first use)"
fi

echo ""
echo "Starting MCP server..."
echo ""

# Execute the main command (passed CMD args)
exec python -m mcp_server.server "$@"
