"""Showcase InsightFace's optional RGB face-liveness addon on one image,
together with gender and age estimation.

Usage:
    python examples/liveness_demo.py input.jpg
    python examples/liveness_demo.py input.jpg --mode observe --threshold 0.9 --output out.jpg
    python examples/liveness_demo.py --webcam --ctx -1
    python examples/liveness_demo.py --webcam --mirror --ctx -1

For webcam mode, open http://localhost:8765 in a Windows browser. The browser
captures the Windows camera; Python in WSL receives and annotates its frames.

The liveness score is the model's estimated probability that a detected face is
live, not an identity match or a guarantee against spoofing. Calibrate the
threshold for the model and execution provider used in your application.
"""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
from pathlib import Path
import secrets
import threading
import time

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def main():
    """Run on a saved image or serve the browser webcam client from WSL.

    The command-line options control the input image, liveness policy, decision
    threshold, output image path, and execution device:

    * ``image``: input image path. The image must be readable by OpenCV.
    * ``--mode``: ``normal`` skips recognition for faces not classified as live;
      ``observe`` reports liveness but does not use it to block recognition.
    * ``--threshold``: live-score cutoff from 0 to 1; scores equal to the cutoff
      count as live. The addon's default is 0.8.
    * ``--output``: destination for the image annotated with face boxes and labels.
    * ``--ctx``: execution provider context; use -1 for CPU or a GPU id for GPU.
    * ``--debug``: show InsightFace debug logs, such as which model files were
      found and their input shapes and normalization values.
    * ``--webcam``: accept frames from a Windows browser instead of an image.
    * ``--port``: localhost HTTP port for webcam mode (default: 8765).
    * ``--mirror``: flip webcam frames horizontally, like a mirror.
    """
    parser = argparse.ArgumentParser(description="InsightFace liveness demo")
    parser.add_argument(
        "image",
        nargs="?",
        help="Input image path (for example, friends.jpg)",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Serve a browser webcam client at http://localhost:8765",
    )
    parser.add_argument("--port", type=int, default=8765, help="Webcam server port (default: 8765)")
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Flip webcam frames horizontally so movement matches a mirror",
    )
    parser.add_argument("--mode", choices=["normal", "observe"], default="normal",
                        help="normal blocks recognition for non-live/rejected faces; observe does not")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.8,
        help="Minimum live score in [0, 1]; equality passes (default: 0.8)",
    )
    parser.add_argument(
        "--output",
        default="liveness_result.jpg",
        help="Path for the annotated output image (default: liveness_result.jpg)",
    )
    parser.add_argument(
        "--ctx",
        type=int,
        default=0,
        help="GPU id for inference, or -1 to use the CPU (default: 0)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print InsightFace debug log messages",
    )
    args = parser.parse_args()
    if bool(args.image) == args.webcam:
        parser.error("Choose either an input image or --webcam, but not both.")
    if args.mirror and not args.webcam:
        parser.error("--mirror can only be used with --webcam.")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")

    # InsightFace logs through the standard logging module under the
    # "insightface" logger hierarchy but does not attach handlers itself.
    # basicConfig adds a console handler; only InsightFace is raised to DEBUG
    # so other libraries stay quiet.
    logging.basicConfig(format="%(levelname)s %(name)s: %(message)s")
    if args.debug:
        logging.getLogger("insightface").setLevel(logging.DEBUG)

    # The liveness addon is optional and must be selected explicitly. In normal
    # mode it gates recognition; observe mode is useful for inspecting scores
    # while still computing recognition embeddings for every detected face.
    app = FaceAnalysis(
        name="buffalo_l",
        addons=["liveness"],
        liveness_mode=args.mode,
        liveness_threshold=args.threshold,
    )
    # prepare() initializes the selected models and chooses CPU/GPU execution.
    # det_size controls the detector's input resolution.
    app.prepare(ctx_id=args.ctx, det_size=(640, 640))

    if args.webcam:
        server = create_webcam_server(app, args.port, mirror=args.mirror)
        print(f"Open http://localhost:{args.port} in Windows Edge or Chrome.", flush=True)
        print("Allow camera access, then click Start camera. Press Ctrl+C here to stop.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping webcam server.")
        finally:
            server.server_close()
        return

    # OpenCV reads color images in BGR order, which is the format expected by
    # FaceAnalysis. Fail explicitly rather than passing an invalid image onward.
    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Could not read {args.image}")

    # get() detects faces, evaluates liveness, and (subject to mode) runs the
    # other selected tasks. The face list includes faces even when liveness
    # rejects recognition, so their locations can still be displayed.
    faces = app.get(image)
    print(f"Detected {len(faces)} face(s) in {args.image} (mode={args.mode}, threshold={args.threshold})")
    for line in annotate_faces(image, faces):
        print(line)

    if not cv2.imwrite(args.output, image):
        raise OSError(f"Could not write {args.output}")
    print(f"Annotated image saved to {args.output}")


def annotate_faces(image, faces):
    """Draw model results in place and return human-readable lines for each face.

    Args:
        image: Original BGR image as a NumPy array; annotations modify it.
        faces: Face objects returned by ``FaceAnalysis.get(image)``.

    Returns:
        Display lines including liveness, gender/age, embedding availability,
        and any input-rejection reasons. Both CLI and webcam mode use these.
    """
    lines = []
    for i, face in enumerate(faces):
        result = face.liveness
        # A missing result means liveness did not run. An input_rejected result
        # means the aligned face crop lacked enough source-image area; it is not
        # a "fake" classification. For status "ok", is_live applies threshold.
        if result is None:
            label, color = "N/A", (128, 128, 128)
        elif result.status == "input_rejected":
            label, color = "REJECTED", (0, 165, 255)
        elif result.is_live:
            label, color = f"LIVE {result.live_score:.2f}", (0, 200, 0)
        else:
            label, color = f"FAKE {result.live_score:.2f}", (0, 0, 255)

        # buffalo_l includes a gender/age model (genderage.onnx). It runs for
        # every detected face, because normal liveness mode only skips the
        # recognition task. face.sex is "M"/"F" and face.age is an integer
        # estimate in years; both are None if the model was not loaded.
        if face.sex is not None and face.age is not None:
            demographics = f"{face.sex}, {face.age}"
        else:
            demographics = "gender/age N/A"

        # Recognition can be absent, for example when normal mode blocks it for
        # a fake or rejected face. This is separate from the liveness result.
        has_embedding = face.embedding is not None
        lines.append(f"  face {i}: {label}  {demographics}  (embedding computed: {has_embedding})")
        if result is not None and result.status == "input_rejected":
            lines.append(f"    reason: {result.reason}")

        # Draw the detector's bounding box with the liveness outcome above it
        # and gender/age below it. OpenCV color tuples are BGR: green, red, orange.
        x1, y1, x2, y2 = face.bbox.astype(int)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.putText(image, demographics, (x1, min(y2 + 18, image.shape[0] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return lines


def create_webcam_server(app, port, mirror=False):
    """Create a localhost-only receiver; the caller starts and closes the server.

    Args:
        app: Prepared FaceAnalysis instance, reused for all incoming frames.
        port: Local TCP port. Tests may use 0 to ask the OS for a free port.
        mirror: Flip frames horizontally before inference and mirror the
            browser preview.

    The page sends JPEG frames with a per-server token. Processing is serialized:
    a second in-flight frame gets HTTP 429 rather than queuing stale video.
    Frames and results stay in memory and are not written to disk.
    """
    token = secrets.token_urlsafe(32)
    page = Path(__file__).with_name("liveness_webcam.html").read_text(encoding="utf-8")
    page = page.replace("__FRAME_TOKEN__", token)
    page = page.replace("__MIRROR__", "true" if mirror else "false").encode("utf-8")
    inference_lock = threading.Lock()
    logger = logging.getLogger("insightface.liveness_demo")

    class WebcamHandler(BaseHTTPRequestHandler):
        # A client that stops sending a frame should not retain a worker forever.
        timeout = 15

        def respond(self, status, body, content_type="application/json"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                logger.debug("Browser disconnected before receiving a frame.")

        def error(self, status, message):
            self.respond(status, json.dumps({"error": message}).encode("utf-8"))

        def do_GET(self):
            if self.path != "/":
                self.error(404, "Not found.")
                return
            self.respond(200, page, "text/html; charset=utf-8")

        def do_POST(self):
            if self.path != "/frame":
                self.error(404, "Not found.")
                return
            # Only our local page knows this token. No CORS permissions are
            # granted to unrelated websites that might try to call localhost.
            if not secrets.compare_digest(self.headers.get("X-Frame-Token", ""), token):
                self.error(403, "Invalid client token. Reload the webcam page.")
                return
            if self.headers.get("Content-Type") != "image/jpeg":
                self.error(415, "Send a JPEG frame.")
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                self.error(400, "Invalid Content-Length.")
                return
            if not 0 < length <= 2 * 1024 * 1024:
                self.error(413, "Frame must be between 1 byte and 2 MiB.")
                return
            if not inference_lock.acquire(blocking=False):
                self.error(429, "Another frame is being processed. Use one camera tab.")
                return
            try:
                try:
                    body = self.rfile.read(length)
                except (TimeoutError, ConnectionError):
                    self.error(408, "Timed out or disconnected while receiving the frame.")
                    return
                if len(body) != length:
                    self.error(400, "Incomplete frame.")
                    return
                try:
                    image = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
                except cv2.error:
                    self.error(400, "Could not decode the JPEG frame.")
                    return
                if image is None:
                    self.error(400, "Could not decode the JPEG frame.")
                    return
                if max(image.shape[:2]) > 1920:
                    self.error(413, "Resize the frame to at most 1920 pixels on either edge.")
                    return
                # Flip before inference so boxes match the mirrored image and
                # annotation text remains readable.
                if mirror:
                    image = cv2.flip(image, 1)
                started = time.perf_counter()
                try:
                    faces = app.get(image)
                    lines = annotate_faces(image, faces)
                    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if not ok:
                        raise RuntimeError("Could not encode the annotated frame.")
                except Exception:
                    # Inference errors must not masquerade as fake faces or
                    # successful frames. Report HTTP failure and log the cause.
                    logger.exception("Webcam frame processing failed")
                    self.error(500, "Frame processing failed. See the WSL terminal for details.")
                    return
                payload = {
                    "image": "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode("ascii"),
                    "face_count": len(faces),
                    "results": lines,
                    "elapsed_ms": round((time.perf_counter() - started) * 1000),
                }
                self.respond(200, json.dumps(payload).encode("utf-8"))
            finally:
                inference_lock.release()

        def log_message(self, format, *args):
            # Avoid printing one successful HTTP access log per video frame.
            logger.debug(format, *args)

    return ThreadingHTTPServer(("127.0.0.1", port), WebcamHandler)


if __name__ == "__main__":
    main()
