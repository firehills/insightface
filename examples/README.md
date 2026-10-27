# InsightFace Examples

Install or upgrade InsightFace before running the examples:

```bash
python -m pip install -U insightface
```

See the [Python package README](../python-package/README.md) for documentation.

## Liveness detection

[demo_liveness.py](demo_liveness.py) runs the optional RGB liveness addon on
one or more images. Use an InsightFace installation that supports
`addons=["liveness"]`; see the [addon documentation](../python-package/docs/liveness.md)
for model downloads, offline installation, and licensing notices.

From the repository root:

```bash
python examples/demo_liveness.py input.jpg --out-dir liveness_output
python examples/demo_liveness.py input.jpg another.jpg --mode normal --threshold 0.8
```

The demo defaults to CPU; use `--ctx 0` for the first CUDA GPU with a compatible
ONNX Runtime GPU installation. The first run downloads any missing models.
It prints each face's live/fake score or input-rejection reason, reports whether
a recognition embedding was computed, and saves annotated images with a
`liveness_` prefix. Green means live, red means fake, and yellow means input
rejected. No detected faces is reported as `0 face(s)`, not as fake.

The default `--mode observe` computes recognition embeddings regardless of the
liveness result. Use `--mode normal` to compute embeddings only for live faces.
The threshold is a model-score cutoff in `[0, 1]`, not a guarantee of authenticity;
validate it with your deployment data and execution provider.

### Windows webcam with WSL2

[liveness_demo.py](liveness_demo.py) also supports a Windows browser as its
webcam client. No Python installation or camera forwarding is needed on Windows.
Run this command **inside WSL**, from the repository root:

```bash
python examples/liveness_demo.py --webcam --ctx -1
```

Once the models finish loading, open **http://localhost:8765** in **Windows Edge
or Chrome**, allow camera access, and click **Start camera**. The page shows the
local camera alongside the annotated frames returned by WSL, with liveness,
gender/age estimates, embedding availability, and input-rejection reasons.
Click **Stop camera** to release the webcam; press **Ctrl+C** in WSL to stop the
server. Use `--debug` for model and HTTP debug logs.

For GPU inference or a different port, for example:

```bash
python examples/liveness_demo.py --webcam --ctx 0 --port 8766 --mode observe --threshold 0.8
```

Open http://localhost:8766 for that command. This demo defaults to `normal`
liveness mode: non-live or input-rejected faces skip recognition, but still get
gender/age estimates. Image mode remains available:

```bash
python examples/liveness_demo.py friends.jpg --ctx -1 --output liveness_result.jpg
```

The browser sends JPEG frames (at most 960 pixels on the longest edge) over local
HTTP, one frame per response, targeting at most 5 FPS. CPU inference may be
slower; frames are not queued while inference is busy. Frames and results are
kept in memory, not saved in webcam mode. The server binds to `127.0.0.1` only:
it is a local development demo, not an authenticated network service. Close
other camera tabs if the page reports that another frame is being processed.

WSL2 normally forwards Windows localhost connections to Linux. If the page does
not load, check the WSL terminal for startup errors and try
`curl http://localhost:8765/` inside WSL. If running in a VS Code dev container,
forward port **8765** (or your chosen port) in VS Code's **Ports** panel to
Windows localhost. Keep the Windows browser URL on `localhost`: ordinary HTTP
URLs using the WSL IP are not secure contexts and browsers may deny webcam
access. For permission errors, allow camera use in Windows privacy settings and
the browser's site settings, and close applications using the camera.

Liveness, gender and age are fallible model estimates. Use consenting subjects;
these labels do not establish identity, authenticity, or a person's gender.
