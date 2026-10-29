"""Run the local API and Vite together; terminate only the children we created."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    root = Path(__file__).resolve().parents[1]
    frontend = root / "web" / "frontend"
    backend_port = int(os.environ.get("USAMI_BACKEND_PORT", "8001"))
    frontend_port = int(os.environ.get("USAMI_FRONTEND_PORT", "5173"))
    vite = frontend / "node_modules" / "vite" / "bin" / "vite.js"
    if not vite.is_file():
        sys.exit("Frontend dependencies missing: cd web/frontend && npm ci")
    try:
        import fastapi  # noqa: F401
        import uvicorn  # noqa: F401
        for port in (backend_port, frontend_port):
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", port))
    except ImportError:
        sys.exit("Backend dependencies missing: python -m pip install -e '.[web]'")
    except OSError as exc:
        sys.exit(f"Local port unavailable: {exc}. Set USAMI_BACKEND_PORT / USAMI_FRONTEND_PORT.")

    children = []

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        children.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "web.backend.app:app",
                                         "--host", "127.0.0.1", "--port", str(backend_port)], cwd=root))
        origin_values = ",".join(sorted({
            f"http://127.0.0.1:{frontend_port}", f"http://localhost:{frontend_port}",
            f"http://127.0.0.1:{backend_port}", f"http://localhost:{backend_port}",
        }))
        environment = dict(os.environ, USAMI_BACKEND_URL=f"http://127.0.0.1:{backend_port}",
                           USAMI_HUMAN_ALLOWED_ORIGINS=os.environ.get(
                               "USAMI_HUMAN_ALLOWED_ORIGINS", origin_values))
        children.append(subprocess.Popen(["node", str(vite), "--host", "127.0.0.1", "--port", str(frontend_port)],
                                         cwd=frontend, env=environment))
        print(f"\nUsami World Explorer: http://127.0.0.1:{frontend_port}\nCtrl+C stops both servers.\n", flush=True)
        while all(child.poll() is None for child in children):
            time.sleep(0.25)
        return next(child.returncode for child in children if child.returncode is not None)
    except KeyboardInterrupt:
        return 0
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    sys.exit(main())
