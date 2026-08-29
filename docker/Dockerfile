# ── Volatility MCP — Multi-stage Dockerfile ─────────────────────────
# Build:  docker build -t volatility-mcp:latest -f docker/Dockerfile .
# Run:    docker run -p 8765:8765 volatility-mcp:latest --transport sse

# ── Stage 1: Build ──────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libffi-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml README.md ./
COPY mcp_server/ ./mcp_server/
COPY core/ ./core/
COPY session_store/ ./session_store/

# Install dependencies to a separate prefix for later copy
RUN pip install --no-cache-dir --prefix=/install -e ".[pdf]"

# ── Stage 2: Runtime ────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

# Install runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY --from=builder /build/mcp_server /app/mcp_server
COPY --from=builder /build/core /app/core
COPY --from=builder /build/session_store /app/session_store
COPY --from=builder /build/pyproject.toml /app/pyproject.toml

WORKDIR /app

# Create directories for reports and session DB
RUN mkdir -p /tmp/volmcp_reports /tmp/volmcp_sessions

# Copy entrypoint
COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose SSE port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Default environment
ENV VOLMCP_DEFAULT_PROVIDER=yfinance \
    VOLMCP_SESSION_BACKEND=sqlite \
    VOLMCP_SESSION_DB=/data/volmcp_sessions.db \
    VOLMCP_REPORT_DIR=/tmp/volmcp_reports \
    PYTHONPATH=/app

# Volume for persistent session DB
VOLUME ["/data"]

ENTRYPOINT ["/entrypoint.sh"]
CMD ["--transport", "sse", "--host", "0.0.0.0", "--port", "8765"]
