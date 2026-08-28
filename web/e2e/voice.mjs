/* End-to-end: a real browser, a real socket, a real pipeline — fake models.
 *
 * See e2e/README.md for how to run it. The three things this covers that
 * nothing else does:
 *
 *   1. the AudioWorklet actually resamples to 16 kHz and emits whole frames
 *   2. those frames reach the server's VAD and close an utterance
 *   3. reply audio decodes and plays, and the browser's report of that
 *      reaches the server before its start timeout fires
 */
import { chromium } from "playwright";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

const APP = process.env.E2E_APP_URL || "http://127.0.0.1:3100";
const SERVER = process.env.E2E_SERVER_URL || "http://127.0.0.1:8123";
const SP = mkdtempSync(join(tmpdir(), "kural-e2e-"));

/* Chrome loops this file as the microphone. One cycle is silence long enough
 * for the noise floor to be calibrated from it, then a burst long enough to
 * clear min_utterance_ms, then silence long enough to close the utterance.
 * Chrome's own fake device is a short beep and satisfies none of those. */
function writeFakeMic(path) {
  const rate = 48000;
  const parts = [[2.0, 0], [1.2, 0.8], [2.5, 0]];
  const total = parts.reduce((n, [s]) => n + Math.round(rate * s), 0);
  const pcm = Buffer.alloc(total * 2);
  let seed = 7;
  const rand = () => ((seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff) * 2 - 1;
  let i = 0;
  for (const [secs, amp] of parts) {
    const n = Math.round(rate * secs);
    for (let k = 0; k < n; k++, i++) {
      // Voiced-ish: a low fundamental with noise on top, so it clears an
      // energy gate the way speech does rather than as a pure tone.
      const v = amp === 0 ? 0
        : Math.max(-1, Math.min(1, (Math.sin((2 * Math.PI * 140 * k) / rate) * 0.5 + rand() * 0.35) * amp));
      pcm.writeInt16LE((v * 32767) | 0, i * 2);
    }
  }
  const head = Buffer.alloc(44);
  head.write("RIFF", 0); head.writeUInt32LE(36 + pcm.length, 4); head.write("WAVE", 8);
  head.write("fmt ", 12); head.writeUInt32LE(16, 16); head.writeUInt16LE(1, 20);
  head.writeUInt16LE(1, 22); head.writeUInt32LE(rate, 24);
  head.writeUInt32LE(rate * 2, 28); head.writeUInt16LE(2, 32); head.writeUInt16LE(16, 34);
  head.write("data", 36); head.writeUInt32LE(pcm.length, 40);
  writeFileSync(path, Buffer.concat([head, pcm]));
  return path;
}

const fakeMic = writeFakeMic(join(SP, "fakemic.wav"));

const failures = [];
const check = (name, cond) => {
  console.log(`  ${cond ? "PASS" : "FAIL"}  ${name}`);
  if (!cond) failures.push(name);
};

const browser = await chromium.launch({
  args: [
    "--use-fake-ui-for-media-stream",
    "--use-fake-device-for-media-stream",
    // A known signal beats Chrome's beep pattern: silence long enough to
    // calibrate, then a burst long enough to clear min_utterance_ms.
    `--use-file-for-fake-audio-capture=${fakeMic}`,
    "--autoplay-policy=no-user-gesture-required",
  ],
});
const ctx = await browser.newContext({ permissions: ["microphone"] });
const page = await ctx.newPage();

const consoleErrors = [];
page.on("console", (m) => {
  if (m.type() === "error") consoleErrors.push(m.text());
});
page.on("pageerror", (e) => consoleErrors.push(String(e)));

// Watch the socket from the browser's side: this is what proves audio frames
// actually leave the worklet and reply audio actually arrives.
const wire = { sentFrames: 0, sentBytes: 0, recvJson: [], recvBinary: 0 };
page.on("websocket", (ws) => {
  ws.on("framesent", (f) => {
    if (typeof f.payload === "string") return;
    wire.sentFrames++;
    wire.sentBytes += f.payload.length;
  });
  ws.on("framereceived", (f) => {
    if (typeof f.payload === "string") {
      try { wire.recvJson.push(JSON.parse(f.payload)); } catch {}
    } else wire.recvBinary++;
  });
});

await page.goto(APP, { waitUntil: "networkidle" });

console.log("page loads");
check("brand renders", (await page.textContent("main")).includes("Kural"));
const startBtn = page.locator("button", { hasText: /Start talking|Loading|offline/ });
await startBtn.waitFor({ timeout: 15000 });
check("start button reflects a ready server",
  (await startBtn.textContent()).includes("Start talking"));

console.log("\ntheme");
const bgLight = await page.evaluate(() =>
  getComputedStyle(document.body).backgroundColor);
await page.click('button[aria-label*="theme"]');
await page.waitForTimeout(200);
const bgDark = await page.evaluate(() =>
  getComputedStyle(document.body).backgroundColor);
check("light theme is pure white", bgLight === "rgb(255, 255, 255)");
check("dark theme is pure black", bgDark === "rgb(0, 0, 0)");
check("theme persists to localStorage",
  (await page.evaluate(() => localStorage.getItem("kural-theme"))) === "dark");
await page.click('button[aria-label*="theme"]');

console.log("\nstarting a session");
await startBtn.click();
await page.waitForFunction(
  () => document.querySelector("[data-state]")?.getAttribute("data-state") !== "idle",
  { timeout: 15000 },
);
const stateAfterStart = await page.getAttribute("[data-state]", "data-state");
check("orb leaves idle", stateAfterStart !== "idle");

// Frames must start flowing before anything else can work.
await page.waitForFunction(() => true, { timeout: 100 });
await page.waitForTimeout(2500);
check("microphone frames are being sent", wire.sentFrames > 20);
check("frames are exactly 320 samples (640 bytes)",
  wire.sentBytes > 0 && wire.sentBytes % 640 === 0);

console.log("\na full turn");
// The fake mic loops silence -> burst -> silence, so a turn happens on its own.
await page.waitForFunction(
  () => document.querySelectorAll('[class*="bubble"]').length >= 2,
  { timeout: 30000 },
).catch(() => {});

const bubbles = await page.locator('[class*="bubble"]').allTextContents();
check("the user's transcript is shown",
  bubbles.some((b) => b.includes("how are you")));
check("the reply is shown",
  bubbles.some((b) => b.includes("I am doing well")));

const kinds = wire.recvJson.map((m) => m.type);
check("hello was received", kinds.includes("hello"));
check("noise floor was calibrated", kinds.includes("calibrated"));
check("speech was detected", kinds.includes("speech_start"));
check("an utterance closed", kinds.includes("utterance"));
check("transcript event received", kinds.includes("stt"));
check("reply event received", kinds.includes("llm"));
check("audio was announced", kinds.includes("audio"));
check("audio bytes arrived", wire.recvBinary >= 1);

// The browser must confirm it actually played, or the server's playback stage
// falls through its start timeout and reports a stall.
const played = wire.recvJson.filter((m) => m.type === "notice" &&
  m.event === "playback_never_started");
check("the reply actually played (no stall reported)", played.length === 0);
check("latency was reported", kinds.includes("latency"));

console.log("\nteardown");
await page.click('button[aria-label="End session"]');
await page.waitForTimeout(800);
check("orb returns to idle",
  (await page.getAttribute("[data-state]", "data-state")) === "idle");

const health = await (await fetch(`${SERVER}/health`)).json();
check("server released the pipeline", health.busy === false);

check("no console errors", consoleErrors.length === 0);
if (consoleErrors.length) console.log("   ", consoleErrors.slice(0, 4));

// Kept for eyeballing the two themes after a run.
await page.screenshot({ path: join(SP, "shot-light.png") });
await page.click('button[aria-label*="theme"]');
await page.waitForTimeout(300);
await page.screenshot({ path: join(SP, "shot-dark.png") });
console.log(`\nscreenshots: ${SP}`);

await browser.close();
console.log("");
if (failures.length) {
  console.log(`${failures.length} FAILED: ${JSON.stringify(failures)}`);
  process.exit(1);
}
console.log("all browser e2e checks passed");
