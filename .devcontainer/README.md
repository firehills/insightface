# Development Container

This CPU-only environment provides Python 3.11, an editable InsightFace SDK
with PrivateFrame dependencies, Python test/lint tools, C/C++ build tools,
FFmpeg, and Node.js 22 for the server frontend tests. GPU training frameworks,
desktop GUI dependencies, datasets, and pretrained models are not installed.

## Open in VS Code

1. Install Docker and the VS Code Dev Containers extension.
2. Open the repository and run **Dev Containers: Reopen in Container**.
3. Wait for the image build and post-create dependency installation to finish.

Python dependencies live in `/opt/insightface-venv`, outside the bind-mounted
checkout. Commands run as the non-root `vscode` user. The checkout remains
shared with the host; this is a development environment, not a security sandbox
for untrusted code. No privileged mode or host Docker socket is required.

The InsightFace cache at `/home/vscode/.insightface` uses a named Docker volume
and survives container rebuilds. Models must be supplied or downloaded explicitly
and remain subject to their own licenses. No licenses are accepted automatically.

Run SDK tests from their project directory:

```sh
cd python-package
python -m pytest
```

Some tests require optional dependencies, models, hardware, or external services.
Older training and research subprojects have independent dependencies; install
those separately rather than assuming this environment supports every framework.

## Server Development

Keep the server's locked CPU dependencies in a separate environment to avoid
mixing its headless OpenCV build with the SDK's OpenCV distribution:

```sh
cd server
python -m venv /home/vscode/server-venv
. /home/vscode/server-venv/bin/activate
python -m pip install -r requirements.dev.lock
python -m pip install --no-deps -e .
make test
make test-frontend
make test-native-cpu
```

Server data and models default to writable directories inside the persistent
cache volume via `INSIGHTFACE_DATA_DIR` and `INSIGHTFACE_MODELS_DIR`.
Follow the [server model setup instructions](../server/README.md) before running
real inference; when using the model installer, use the models directory above.
From `server/`, start the development server with:

```sh
PYTHONPATH=backend:../python-package \
INSIGHTFACE_CONFIG_FILE="$PWD/config/server.toml" \
python -m uvicorn insightface_server.app:app --host 0.0.0.0 --port 8080 --reload
```

Port 8080 is forwarded to the host by VS Code. The server is not started
automatically. For production deployment or NVIDIA GPU inference, use the
existing [server Docker configurations](../server/docker/).