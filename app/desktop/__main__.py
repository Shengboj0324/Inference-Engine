"""``python -m app.desktop`` entry point.

Defers all work to :func:`app.desktop.launcher.run_sidecar` so the module
import is side-effect free and ``run_sidecar`` can be unit-tested
independently of the ``-m`` invocation.
"""

from __future__ import annotations

import argparse
import logging

from app.desktop.launcher import run_sidecar


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m app.desktop",
        description="Launch the Social Media Radar FastAPI sidecar bound to "
                    "a loopback-only TCP port.",
    )
    parser.add_argument("--port", type=int, default=0,
                        help="Force a specific port (0 = auto-assign).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Bind host.  Defaults to 127.0.0.1; do NOT change "
                             "this on a multi-user box.")
    parser.add_argument("--log-level", default="info",
                        choices=["critical", "error", "warning", "info", "debug"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run_sidecar(
        port=args.port or None,
        host=args.host,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
