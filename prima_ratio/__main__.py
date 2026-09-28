"""`python -m prima_ratio` — start the HTTP server."""

from __future__ import annotations

import os


def main() -> None:
    import uvicorn

    from .server import app

    uvicorn.run(
        app,
        host=os.environ.get("PRIMA_HOST", "0.0.0.0"),
        port=int(os.environ.get("PRIMA_PORT", "8000")),
        log_level=os.environ.get("PRIMA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
