import os
import sys
import socket
import subprocess

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable


def get_local_ip() -> str:
    try:
        # Connect to an external address to discover the outbound interface IP.
        # No data is actually sent.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        return socket.gethostbyname(socket.gethostname())


def main():
    port   = int(os.environ.get("PORT", 8080))
    domain = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")

    print("=" * 56)
    print("         TRADING AGENT — API SERVER")
    print("=" * 56)

    # ── Step 1: Sync account data ──────────────────────────────────────────────
    print("\nSyncing account data from Alpaca...")
    result = subprocess.run(
        [PYTHON, os.path.join(BASE_DIR, "account_sync.py")],
        cwd=BASE_DIR,
    )
    if result.returncode != 0:
        print("WARNING: account_sync.py failed — API will start without fresh account data.")

    # ── Step 2: Show connection info ───────────────────────────────────────────
    local_ip = get_local_ip()
    hostname = socket.gethostname()

    print("\n" + "=" * 56)
    print("  API is starting...")
    if domain:
        print(f"  Public URL    : https://{domain}")
        print(f"  API docs      : https://{domain}/docs")
    print(f"  Local machine : http://localhost:{port}")
    print(f"  On your network: http://{local_ip}:{port}")
    print(f"  Hostname      : {hostname}")
    print("=" * 56 + "\n")

    # ── Step 3: Start FastAPI server ───────────────────────────────────────────
    import uvicorn
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    main()
