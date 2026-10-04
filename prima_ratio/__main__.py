"""Entry point: `python -m prima_ratio` runs the HTTP service."""
import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "prima_ratio.server:app",
        host=os.environ.get("PRIMA_HOST", "0.0.0.0"),
        port=int(os.environ.get("PRIMA_PORT", "8000")),
        log_level=os.environ.get("PRIMA_LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
