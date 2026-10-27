"""Focused tests; run with python -m unittest discover -s examples/tests -v."""
import base64
from concurrent.futures import ThreadPoolExecutor
import http.client
import importlib.util
import json
from pathlib import Path
import re
import threading
import unittest
from unittest.mock import Mock

import cv2
import numpy as np
from insightface.app.common import Face


spec = importlib.util.spec_from_file_location(
    "liveness_demo", Path(__file__).resolve().parents[1] / "liveness_demo.py"
)
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class WebcamServerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = Mock()
        cls.server = demo.create_webcam_server(cls.app, 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever)
        cls.thread.start()
        _, cls.page = cls.request("GET", "/")
        cls.token = re.search(rb'const token = "([^"]+)"', cls.page).group(1).decode()
        ok, jpeg = cv2.imencode(".jpg", np.zeros((120, 160, 3), dtype=np.uint8))
        assert ok
        cls.jpeg = jpeg.tobytes()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.app.reset_mock()
        self.app.get.side_effect = None
        self.app.get.return_value = []

    @classmethod
    def request(cls, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection(*cls.server.server_address, timeout=10)
        try:
            connection.request(method, path, body=body, headers=headers or {})
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def frame(self, body=None, headers=None):
        return self.request(
            "POST", "/frame", self.jpeg if body is None else body,
            headers if headers is not None else {
                "Content-Type": "image/jpeg",
                "X-Frame-Token": self.token,
            },
        )

    def test_serves_browser_page_and_local_binding(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")
        self.assertIn(b"getUserMedia", self.page)
        self.assertNotIn(b"__FRAME_TOKEN__", self.page)
        self.assertIn(b'data-mirror="false"', self.page)

    def test_frames_are_not_mirrored_by_default(self):
        image = np.zeros((16, 16, 3), dtype=np.uint8)
        image[:, :8] = 255
        _, jpeg = cv2.imencode(".jpg", image)
        status, _ = self.frame(body=jpeg.tobytes())
        self.assertEqual(status, 200)
        received = self.app.get.call_args.args[0]
        self.assertGreater(received[:, :8].mean(), received[:, 8:].mean())

    def test_frame_returns_jpeg_and_no_face_summary(self):
        status, body = self.frame()
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["face_count"], 0)
        self.assertEqual(payload["results"], [])
        self.assertGreaterEqual(payload["elapsed_ms"], 0)
        encoded = base64.b64decode(payload["image"].split(",", 1)[1])
        image = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(image.shape, (120, 160, 3))
        self.assertEqual(self.app.get.call_args.args[0].shape, (120, 160, 3))

    def test_annotations_include_all_outcomes_and_demographics(self):
        self.app.get.return_value = [
            Face(bbox=np.array([5, 20, 50, 80]), liveness=Face(
                status="ok", is_live=True, live_score=0.95),
                gender=1, age=31, embedding=np.ones(4)),
            Face(bbox=np.array([55, 20, 100, 80]), liveness=Face(
                status="ok", is_live=False, live_score=0.1), gender=0, age=30),
            Face(bbox=np.array([5, 20, 50, 80]), liveness=Face(
                status="input_rejected", is_live=None, live_score=None,
                reason="Insufficient image area."), gender=1, age=40),
            Face(bbox=np.array([5, 20, 50, 80])),
        ]
        status, body = self.frame()
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["face_count"], 4)
        lines = "\n".join(payload["results"])
        self.assertIn("LIVE 0.95  M, 31  (embedding computed: True)", lines)
        self.assertIn("FAKE 0.10  F, 30  (embedding computed: False)", lines)
        self.assertIn("REJECTED  M, 40", lines)
        self.assertIn("reason: Insufficient image area.", lines)
        self.assertIn("N/A  gender/age N/A", lines)

    def test_rejects_missing_token(self):
        status, _ = self.frame(headers={"Content-Type": "image/jpeg"})
        self.assertEqual(status, 403)
        self.app.get.assert_not_called()

    def test_rejects_bad_content_type(self):
        status, _ = self.frame(headers={"X-Frame-Token": self.token})
        self.assertEqual(status, 415)
        self.app.get.assert_not_called()

    def test_rejects_bad_sizes_and_jpeg(self):
        for value, expected in [("0", 413), ("2097153", 413), ("invalid", 400)]:
            with self.subTest(length=value):
                status, _ = self.frame(body=b"", headers={
                    "Content-Type": "image/jpeg",
                    "X-Frame-Token": self.token,
                    "Content-Length": value,
                })
                self.assertEqual(status, expected)
        status, _ = self.frame(body=b"not a jpeg")
        self.assertEqual(status, 400)
        self.app.get.assert_not_called()

    def test_rejects_oversized_dimensions(self):
        _, jpeg = cv2.imencode(".jpg", np.zeros((2, 1921, 3), dtype=np.uint8))
        status, _ = self.frame(body=jpeg.tobytes())
        self.assertEqual(status, 413)
        self.app.get.assert_not_called()

    def test_unknown_routes(self):
        for method in ["GET", "POST"]:
            status, _ = self.request(method, "/missing")
            self.assertEqual(status, 404)

    def test_inference_failure_is_an_error_and_releases_lock(self):
        self.app.get.side_effect = RuntimeError("model inference failure")
        with self.assertLogs("insightface.liveness_demo", level="ERROR") as logged:
            status, body = self.frame()
        self.assertEqual(status, 500)
        self.assertIn("model inference failure", "\n".join(logged.output))
        self.assertIn("See the WSL terminal", json.loads(body)["error"])
        self.app.get.side_effect = None
        self.assertEqual(self.frame()[0], 200)

    def test_concurrent_frames_do_not_queue(self):
        entered = threading.Event()
        release = threading.Event()

        def inference(image):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("Test did not release inference.")
            return []

        self.app.get.side_effect = inference
        with ThreadPoolExecutor(max_workers=1) as executor:
            pending = executor.submit(self.frame)
            try:
                self.assertTrue(entered.wait(5))
                self.assertEqual(self.frame()[0], 429)
            finally:
                release.set()
            self.assertEqual(pending.result()[0], 200)


class MirroredWebcamServerTests(unittest.TestCase):
    def test_mirror_flips_frames_and_browser_preview(self):
        app = Mock()
        app.get.return_value = []
        server = demo.create_webcam_server(app, 0, mirror=True)
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            connection = http.client.HTTPConnection(*server.server_address, timeout=10)
            try:
                connection.request("GET", "/")
                page = connection.getresponse().read()
            finally:
                connection.close()
            self.assertIn(b'data-mirror="true"', page)
            token = re.search(rb'const token = "([^"]+)"', page).group(1).decode()

            image = np.zeros((16, 16, 3), dtype=np.uint8)
            image[:, :8] = 255
            _, jpeg = cv2.imencode(".jpg", image)
            connection = http.client.HTTPConnection(*server.server_address, timeout=10)
            try:
                connection.request("POST", "/frame", body=jpeg.tobytes(), headers={
                    "Content-Type": "image/jpeg",
                    "X-Frame-Token": token,
                })
                self.assertEqual(connection.getresponse().status, 200)
            finally:
                connection.close()
            received = app.get.call_args.args[0]
            self.assertGreater(received[:, 8:].mean(), received[:, :8].mean())
        finally:
            server.shutdown()
            server.server_close()
            thread.join()


if __name__ == "__main__":
    unittest.main()
