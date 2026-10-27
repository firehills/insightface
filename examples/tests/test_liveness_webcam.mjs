// Run with: node --test examples/tests/test_liveness_webcam.mjs
// Exercise the real page script with a simulated camera and HTTP responses.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const html = readFileSync(new URL("../liveness_webcam.html", import.meta.url), "utf8");
const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];

function client({ getUserMedia, fetchFrame } = {}) {
  const elements = new Map();
  for (const id of ["video", "capture", "processed", "status", "results", "start", "stop"]) {
    elements.set(id, {
      disabled: id === "stop",
      hidden: id === "processed",
      textContent: "",
      listeners: {},
      addEventListener(event, callback) { this.listeners[event] = callback; }
    });
  }
  const video = elements.get("video");
  Object.assign(video, { videoWidth: 1920, videoHeight: 1080, play: async () => {} });
  const canvas = elements.get("capture");
  canvas.getContext = () => ({ drawImage() {} });
  canvas.toBlob = callback => callback(new Blob(["jpeg"], { type: "image/jpeg" }));
  const track = { stopped: false, stop() { this.stopped = true; } };
  const camera = { getTracks: () => [track] };
  const requests = [];
  const context = vm.createContext({
    document: { getElementById: id => elements.get(id) },
    navigator: { mediaDevices: { getUserMedia: getUserMedia ?? (async () => camera) } },
    window: { addEventListener() {} },
    AbortController, setTimeout, performance,
    fetch: async (url, options) => {
      requests.push({ url, options });
      if (fetchFrame) return fetchFrame(url, options);
      return {
        ok: true,
        json: async () => ({
          image: "data:image/jpeg;base64,annotated",
          results: ["LIVE 0.95 M, 31"],
          face_count: 1,
          elapsed_ms: 20
        })
      };
    }
  });
  vm.runInContext(script, context);
  return {
    elements, camera, track, requests,
    start: () => elements.get("start").listeners.click(),
    stop: () => elements.get("stop").listeners.click()
  };
}

async function until(predicate) {
  for (let attempt = 0; attempt < 100; attempt++) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  assert.fail("Timed out waiting for browser state.");
}

test("captures resized JPEG frames, displays results, and stops the camera", async () => {
  const page = client();
  const running = page.start();
  await until(() => page.elements.get("results").textContent.includes("LIVE"));
  const { url, options } = page.requests[0];
  assert.equal(url, "/frame");
  assert.equal(options.headers["Content-Type"], "image/jpeg");
  assert.equal(options.body.type, "image/jpeg");
  assert.ok(options.headers["X-Frame-Token"]);
  assert.equal(page.elements.get("capture").width, 960);
  assert.equal(page.elements.get("capture").height, 540);
  assert.equal(page.elements.get("processed").hidden, false);
  assert.match(page.elements.get("status").textContent, /1 face\(s\), 20 ms/);
  page.stop();
  await running;
  assert.equal(page.track.stopped, true);
  assert.equal(page.elements.get("start").disabled, false);
  assert.equal(page.elements.get("stop").disabled, true);
});

test("does not capture a second frame while the first request is pending", async () => {
  let release;
  const page = client({
    fetchFrame: () => new Promise(resolve => { release = resolve; })
  });
  const running = page.start();
  await until(() => page.requests.length === 1);
  await new Promise(resolve => setTimeout(resolve, 250));
  assert.equal(page.requests.length, 1);
  page.stop();
  release({ ok: true, json: async () => ({ image: "stale", results: ["stale"] }) });
  await running;
  assert.notEqual(page.elements.get("processed").src, "stale");
});

test("shows receiver errors and releases the camera", async () => {
  const page = client({
    fetchFrame: async () => ({
      ok: false,
      json: async () => ({ error: "Frame processing failed. See the WSL terminal for details." })
    })
  });
  await page.start();
  assert.match(page.elements.get("status").textContent, /Stopped: Frame processing failed/);
  assert.equal(page.track.stopped, true);
  assert.equal(page.elements.get("start").disabled, false);
});

test("permission denial is visible and permits retry", async () => {
  const page = client({ getUserMedia: async () => { throw new Error("Permission denied"); } });
  await page.start();
  assert.match(page.elements.get("status").textContent, /Permission denied/);
  assert.equal(page.elements.get("start").disabled, false);
  assert.equal(page.requests.length, 0);
});

test("retries a busy receiver after a quick Stop/Start", async () => {
  let count = 0;
  const page = client({
    fetchFrame: async () => ++count === 1 ? {
      ok: false, status: 429, json: async () => ({ error: "Busy" })
    } : {
      ok: true, status: 200,
      json: async () => ({ image: "ready", results: [], face_count: 0, elapsed_ms: 10 })
    }
  });
  const running = page.start();
  await until(() => page.elements.get("status").textContent.startsWith("Waiting"));
  await until(() => page.elements.get("results").textContent === "No faces detected.");
  page.stop();
  await running;
  assert.ok(page.requests.length >= 2);
});

test("stopping while camera permission is pending releases the late camera", async () => {
  let grant;
  const page = client({
    getUserMedia: () => new Promise(resolve => { grant = resolve; })
  });
  const running = page.start();
  page.stop();
  grant(page.camera);
  await running;
  assert.equal(page.track.stopped, true);
  assert.equal(page.requests.length, 0);
  assert.equal(page.elements.get("video").srcObject, null);
});
