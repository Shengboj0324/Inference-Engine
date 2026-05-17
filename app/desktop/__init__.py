"""Desktop sidecar entry point.

Invoke with ``python -m app.desktop`` to launch the FastAPI app on a
random free 127.0.0.1 port.  The Tauri shell discovers the port via the
``sidecar.port`` file inside the user data directory.
"""

from app.desktop.launcher import (
    SidecarConfig,
    find_free_port,
    run_sidecar,
    write_port_file,
)

__all__ = [
    "SidecarConfig",
    "find_free_port",
    "run_sidecar",
    "write_port_file",
]
