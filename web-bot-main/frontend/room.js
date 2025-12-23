// frontend/room.js（修正版）
// 目的：
// - Three.js で「3Dの部屋」を描画（WebGL）
// - 従来UI(#ui3d)を CSS3DRenderer で「奥の壁一面」に貼る
// - 手前に VRM(ずんだもん) を配置（無ければダミー）
// - busイベント（assistant/user）に連動して身振り手振り
// - Webカメラ + 顔検出（任意）でモーションパララックス（擬似3D）

import * as THREE from "three";
import { CSS3DRenderer, CSS3DObject } from "three/examples/jsm/renderers/CSS3DRenderer.js";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader.js";

// three-vrm（three@0.160付近で安定しやすい v2 系）
import { VRMLoaderPlugin, VRMUtils } from "https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.1.0/lib/three-vrm.module.js";

import { on } from "./bus.js";

const AVATAR_URL = "./assets/zundamon.vrm"; // ここに配置したVRMを読みます

// アバターが常にこちら（カメラ）を向くようにするか。
// ※「固定向きが良い」なら false にしてください。
const AVATAR_FACE_VIEWER = true;

// モデルの「正面」が +Z ではなく -Z の場合は Math.PI にすると合うことが多いです。
const AVATAR_YAW_OFFSET = 0;

// --- ちょっとしたデバッグ表示（致命的エラー時のみ） ---
const debug = (() => {
  const el = document.createElement("pre");
  el.style.cssText = [
    "position:fixed",
    "left:12px",
    "bottom:12px",
    "max-width: min(720px, 96vw)",
    "max-height: 40vh",
    "overflow:auto",
    "padding:10px 12px",
    "border-radius:10px",
    "background: rgba(0,0,0,0.75)",
    "color:#fff",
    "font: 12px/1.4 ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
    "z-index:999999",
    "display:none",
  ].join(";");
  document.body.appendChild(el);
  return {
    show(msg) {
      el.textContent = String(msg);
      el.style.display = "block";
    },
    append(msg) {
      el.textContent += `\n${msg}`;
      el.style.display = "block";
    },
    hide() {
      el.style.display = "none";
    },
  };
})();

function requireEl(id) {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing element: #${id} (index.html を確認してください)`);
  return el;
}

function lerpAngle(a, b, t) {
  // shortest-path lerp for radians
  const TWO_PI = Math.PI * 2;
  let d = (b - a) % TWO_PI;
  if (d < -Math.PI) d += TWO_PI;
  else if (d > Math.PI) d -= TWO_PI;
  return a + d * t;
}

// 状態（busイベント）
let assistantSpeaking = false;
let micRms = 0;

on("assistant:speakingStart", () => (assistantSpeaking = true));
on("assistant:speakingEnd", () => (assistantSpeaking = false));
on("user:micRms", (e) => {
  const v = Number(e?.detail?.rms ?? 0);
  // ちょい平滑化（ガタつき防止）
  micRms = micRms * 0.85 + v * 0.15;
});

// 顔トラッキング（任意）
// 目的：
// - なるべく「動かない」原因を潰すため、複数の実装をフォールバックする
//   1) ブラウザ内蔵 FaceDetector（対応ブラウザなら最優先・追加DL不要）
//   2) tfjs + face-landmarks-detection（既存の設計）
let faceTrackerKind = "none"; // "none" | "native" | "tfjs"
let faceDetector = null; // FaceDetector or tfjs detector
let faceVideo = null;
let faceFov = null; // { hfov, vfov }
let faceLastMs = 0;
let faceInFlight = false;
let faceCanvas = null;
let faceCtx = null;
let faceLastSize = { w: 0, h: 0 };
let faceLastOkMs = 0;
let faceLoggedFirstOk = false;
let faceLastNoFaceLogMs = 0;
let faceFallbackDetector = null; // tfjs fallback (e.g., blazeface)
let faceFallbackLoading = false;
let faceFallbackLogged = false;

const FACE_SAMPLE_INTERVAL_MS = 80; // 10〜15fps 程度
const WEBCAM_DIAGONAL_FOV_RAD = Math.PI / 3; // ざっくり60deg（参考zipのtrompeloeilと同系統）

function sleep(ms) {
  return new Promise((r) => setTimeout(r, ms));
}

function loadScriptOnce(src) {
  return new Promise((resolve, reject) => {
    const existing = Array.from(document.scripts).find((s) => s.src === src);
    if (existing) {
      if (existing.dataset.loaded === "1") return resolve(true);
      existing.addEventListener("load", () => resolve(true), { once: true });
      existing.addEventListener("error", (e) => reject(e), { once: true });
      return;
    }

    const s = document.createElement("script");
    s.src = src;
    s.defer = true;
    s.dataset.loaded = "0";
    s.onload = () => {
      s.dataset.loaded = "1";
      resolve(true);
    };
    s.onerror = (e) => reject(e);
    document.head.appendChild(s);
  });
}

function fovFromDiagonal(dfov, w, h) {
  const hyp = Math.sqrt(w * w + h * h);
  const t = Math.tan(dfov / 2);
  return {
    hfov: 2 * Math.atan((w * t) / hyp),
    vfov: 2 * Math.atan((h * t) / hyp),
  };
}

function waitForVideoReady(videoEl, { timeoutMs = 3000 } = {}) {
  // videoWidth/videoHeight が 0 のままだと detector が常に空振りするケースがある
  if ((videoEl.videoWidth || 0) > 0 && (videoEl.videoHeight || 0) > 0) return Promise.resolve(true);
  return new Promise((resolve) => {
    let done = false;
    const finish = (ok) => {
      if (done) return;
      done = true;
      cleanup();
      resolve(ok);
    };
    const onMeta = () => finish(true);
    const onCanPlay = () => finish(true);
    const cleanup = () => {
      videoEl.removeEventListener("loadedmetadata", onMeta);
      videoEl.removeEventListener("canplay", onCanPlay);
    };
    videoEl.addEventListener("loadedmetadata", onMeta, { once: true });
    videoEl.addEventListener("canplay", onCanPlay, { once: true });
    setTimeout(() => finish(false), timeoutMs);
  });
}

async function waitForGlobals({ timeoutMs = 8000 } = {}) {
  const start = performance.now();
  while (performance.now() - start < timeoutMs) {
    if (window.tf && window.faceLandmarksDetection) return true;
    await sleep(50);
  }
  return false;
}

async function initFaceTracking(webcamEl) {
  if (!navigator.mediaDevices?.getUserMedia) {
    console.info("[room] Face tracking disabled (getUserMedia not supported)");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    webcamEl.srcObject = stream;
    try {
      await webcamEl.play();
    } catch {}
    await waitForVideoReady(webcamEl, { timeoutMs: 4000 });
    faceVideo = webcamEl;

    const vw = webcamEl.videoWidth || stream.getVideoTracks?.()?.[0]?.getSettings?.()?.width || 640;
    const vh = webcamEl.videoHeight || stream.getVideoTracks?.()?.[0]?.getSettings?.()?.height || 480;
    faceFov = fovFromDiagonal(WEBCAM_DIAGONAL_FOV_RAD, vw, vh);
    faceLastSize = { w: vw, h: vh };

    // 1) Native FaceDetector (best-effort)
    if ("FaceDetector" in window) {
      try {
        faceDetector = new window.FaceDetector({ fastMode: true, maxDetectedFaces: 1 });
        faceTrackerKind = "native";
        console.info("[room] Face tracking enabled (native FaceDetector)");
        return;
      } catch (e) {
        console.info("[room] Native FaceDetector unavailable:", e);
      }
    }

    // 2) tfjs fallback
    if (!window.tf || !window.faceLandmarksDetection) {
      await waitForGlobals({ timeoutMs: 8000 });
    }

    const tf = window.tf;
    let fld = window.faceLandmarksDetection;

    if (!tf) {
      console.info("[room] Face tracking disabled (tfjs not loaded)");
      debug.append(
        "Face tracking: OFF (tfjs not loaded)\n" +
          "※Webカメラのパララックスを使うには、ブラウザのコンソール/Networkで tfjs の読み込みエラーが無いか確認してください。"
      );
      return;
    }

    if (!fld?.createDetector && !fld?.load) {
      try {
        // UMDが読み込めていない環境向けの保険（CDNのESMビルド）
        fld = await import(
          "https://cdn.jsdelivr.net/npm/@tensorflow-models/face-landmarks-detection@1.0.5/dist/face-landmarks-detection.esm.js"
        );
        window.faceLandmarksDetection = fld;
      } catch (e) {
        console.info("[room] Face tracking disabled (face-landmarks-detection not loaded)");
        debug.append(
          "Face tracking: OFF (face-landmarks-detection not loaded)\n" +
            "※index.html の CDN スクリプト読み込みを確認してください。"
        );
        return;
      }
    }

    // できるだけ滑らかにするため WebGL backend を優先（失敗してもデフォルトで続行）
    try {
      await tf.setBackend("webgl");
    } catch {}
    await tf.ready();

    // v1系: createDetector / 旧API: load
    if (fld.createDetector) {
      const model = fld.SupportedModels?.MediaPipeFaceMesh ?? fld.SupportedModels?.MediaPipeFaceMesh;
      faceDetector = await fld.createDetector(model, {
        runtime: "tfjs",
        refineLandmarks: false,
        maxFaces: 1,
      });
    } else {
      // older API shape (参考zipと同じ系)
      faceDetector = await fld.load(fld.SupportedPackages?.mediapipeFacemesh, {
        maxFaces: 1,
        shouldLoadIrisModel: false,
      });
    }

    faceTrackerKind = "tfjs";
    console.info("[room] Face tracking enabled (tfjs)");
  } catch (e) {
    console.warn("[room] Face tracking init failed:", e);
    debug.append(
      "Face tracking: OFF (camera permission denied or init failed)\n" +
        "※ブラウザのサイト設定でカメラ許可をONにするとモーションパララックスが有効になります。\n" +
        String(e)
    );
  }
}

async function ensureFaceFallbackTfjs() {
  if (faceFallbackDetector || faceFallbackLoading) return;
  if (!window.tf) return;
  faceFallbackLoading = true;
  try {
    // FaceMesh が取れない環境向けの最小フォールバック（軽量なbbox検出）
    await loadScriptOnce("https://cdn.jsdelivr.net/npm/@tensorflow-models/blazeface@0.0.7/dist/blazeface.min.js");
    const blaze = window.blazeface;
    if (!blaze?.load) throw new Error("blazeface global not available after script load");
    // blazeface は tfjs backend をそのまま使える
    faceFallbackDetector = await blaze.load();
    if (!faceFallbackLogged) {
      faceFallbackLogged = true;
      console.info("[room] Face tracking fallback enabled (blazeface)");
    }
  } catch (e) {
    console.warn("[room] Failed to load face fallback (blazeface):", e);
  } finally {
    faceFallbackLoading = false;
  }
}

async function sampleFaceBox() {
  if (!faceDetector || !faceVideo) return null;

  const vw = faceVideo.videoWidth || 640;
  const vh = faceVideo.videoHeight || 480;
  if (vw !== faceLastSize.w || vh !== faceLastSize.h || !faceFov) {
    faceFov = fovFromDiagonal(WEBCAM_DIAGONAL_FOV_RAD, vw, vh);
    faceLastSize = { w: vw, h: vh };
  }

  if (faceVideo.readyState < 2) return null;

  const asXY = (p) => {
    if (!p) return null;
    if (Array.isArray(p)) return { x: Number(p[0]), y: Number(p[1]) };
    if (typeof p === "object") return { x: Number(p.x), y: Number(p.y) };
    return null;
  };

  const normalizeIfNeeded = (n, scale) => {
    const v = Number(n);
    if (!Number.isFinite(v)) return NaN;
    // 0..1（または-0.5..1.5程度）で返ってくる実装があるため、ざっくりで判定して px へ寄せる
    if (Math.abs(v) <= 2 && scale > 2) return v * scale;
    return v;
  };

  if (faceTrackerKind === "native") {
    const faces = await faceDetector.detect(faceVideo);
    if (!faces?.length) return null;
    const bb = faces[0].boundingBox;
    if (!bb) return null;
    const cx = bb.x + bb.width * 0.5;
    const cy = bb.y + bb.height * 0.5;
    const w = Math.max(1, bb.width);
    return { cx, cy, w, vw, vh };
  }

  // tfjs path: face-landmarks-detection (various versions return slightly different shapes)
  const estimateFaces = faceDetector.estimateFaces?.bind(faceDetector) || null;
  if (!estimateFaces) return null;

  // trompeloeil と同じく flipHorizontal:false（カメラ視点のまま）
  let faces = null;
  try {
    faces = await estimateFaces(faceVideo, { flipHorizontal: false });
  } catch {
    // legacy API: estimateFaces({input: ImageData, flipHorizontal, predictIrises})
    try {
      faceCanvas ||= document.createElement("canvas");
      faceCtx ||= faceCanvas.getContext("2d", { willReadFrequently: true });
      if (!faceCtx) return null;
      faceCanvas.width = vw;
      faceCanvas.height = vh;
      faceCtx.drawImage(faceVideo, 0, 0, vw, vh);
      const img = faceCtx.getImageData(0, 0, vw, vh);
      faces = await estimateFaces({ input: img, flipHorizontal: false, predictIrises: false });
    } catch {
      return null;
    }
  }
  if (!faces?.length) {
    // FaceMesh が取れない場合の保険：一定時間ヒットしないなら bbox 検出へ切替
    const now = performance.now();
    if (now - faceLastOkMs > 2500) ensureFaceFallbackTfjs().catch(() => {});
    if (faceFallbackDetector?.estimateFaces) {
      try {
        // blazeface: estimateFaces(input, returnTensors, flipHorizontal, annotateBoxes)
        const preds = await faceFallbackDetector.estimateFaces(faceVideo, false, false, false);
        if (preds?.length) {
          const p = preds[0];
          const tl = asXY(p.topLeft);
          const br = asXY(p.bottomRight);
          if (tl && br) {
            const cx = (tl.x + br.x) * 0.5;
            const cy = (tl.y + br.y) * 0.5;
            const w = Math.max(1, br.x - tl.x);
            if (Number.isFinite(cx) && Number.isFinite(cy) && Number.isFinite(w)) return { cx, cy, w, vw, vh };
          }
        }
      } catch {
        // ignore
      }
    }
    return null;
  }

  const f = faces[0];

  // new API: { box: {xMin,xMax,yMin,yMax} } / legacy: { boundingBox: { topLeft, bottomRight } }
  let xMin, xMax, yMin, yMax;
  if (f.box) {
    const box = f.box;
    xMin = box.xMin ?? box.left ?? 0;
    xMax = box.xMax ?? (box.left ?? 0) + (box.width ?? 0);
    yMin = box.yMin ?? box.top ?? 0;
    yMax = box.yMax ?? (box.top ?? 0) + (box.height ?? 0);
  } else if (f.boundingBox?.topLeft && f.boundingBox?.bottomRight) {
    const p1 = asXY(f.boundingBox.topLeft);
    const p2 = asXY(f.boundingBox.bottomRight);
    if (p1 && p2) {
      xMin = p1.x;
      yMin = p1.y;
      xMax = p2.x;
      yMax = p2.y;
    }
  } else if (f.boundingBox) {
    const bb = f.boundingBox;
    xMin = bb.xMin ?? bb.left ?? 0;
    yMin = bb.yMin ?? bb.top ?? 0;
    xMax = bb.xMax ?? (bb.left ?? 0) + (bb.width ?? 0);
    yMax = bb.yMax ?? (bb.top ?? 0) + (bb.height ?? 0);
  } else {
    // 最終手段：keypoints から bbox を作る
    const kps = f.keypoints || f.scaledMesh || f.mesh || null;
    const arr = Array.isArray(kps) ? kps : null;
    if (!arr?.length) return null;
    let minX = Infinity;
    let minY = Infinity;
    let maxX = -Infinity;
    let maxY = -Infinity;
    for (const kp of arr) {
      const p = asXY(kp);
      if (!p) continue;
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y)) continue;
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    if (!Number.isFinite(minX) || !Number.isFinite(minY) || !Number.isFinite(maxX) || !Number.isFinite(maxY))
      return null;
    xMin = minX;
    yMin = minY;
    xMax = maxX;
    yMax = maxY;
  }

  xMin = normalizeIfNeeded(xMin, vw);
  xMax = normalizeIfNeeded(xMax, vw);
  yMin = normalizeIfNeeded(yMin, vh);
  yMax = normalizeIfNeeded(yMax, vh);

  if (![xMin, xMax, yMin, yMax].every((v) => Number.isFinite(v))) return null;

  const cx = (xMin + xMax) * 0.5;
  const cy = (yMin + yMax) * 0.5;
  const w = Math.max(1, xMax - xMin);
  if (!Number.isFinite(cx) || !Number.isFinite(cy) || !Number.isFinite(w)) return null;
  return { cx, cy, w, vw, vh };
}

// 3D初期化
async function init() {
  const stage = requireEl("stage");
  const uiDiv = requireEl("ui3d");
  const webcamEl = requireEl("webcam");

  // stage はフルスクリーン想定（room.cssが無い場合でも最低限効くように）
  stage.style.position = stage.style.position || "fixed";
  stage.style.inset = stage.style.inset || "0";

  // UIの見た目（最低限）
  uiDiv.style.background = uiDiv.style.background || "rgba(15, 15, 15, 0.92)";
  uiDiv.style.color = uiDiv.style.color || "#fff";
  uiDiv.style.borderRadius = uiDiv.style.borderRadius || "14px";
  uiDiv.style.padding = uiDiv.style.padding || "16px";
  uiDiv.style.width = uiDiv.style.width || "920px";
  uiDiv.style.maxWidth = uiDiv.style.maxWidth || "92vw";

  // --- WebGL scene ---
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x050505);

  const camera = new THREE.PerspectiveCamera(52, window.innerWidth / window.innerHeight, 0.1, 200);
  const camBase = new THREE.Vector3(0, 1.65, 6.2);
  const camTarget = camBase.clone();
  camera.position.copy(camBase);
  const camFocus = new THREE.Vector3(0, 1.55, 0);
  camera.lookAt(camFocus);

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.domElement.style.position = "absolute";
  renderer.domElement.style.inset = "0";
  renderer.domElement.style.zIndex = "0";
  // UI操作を優先（必要なら後で切替）
  renderer.domElement.style.pointerEvents = "none";
  stage.appendChild(renderer.domElement);

  // --- CSS3D scene ---
  const cssScene = new THREE.Scene();
  const cssRenderer = new CSS3DRenderer();
  cssRenderer.setSize(window.innerWidth, window.innerHeight);
  cssRenderer.domElement.style.position = "absolute";
  cssRenderer.domElement.style.inset = "0";
  cssRenderer.domElement.style.zIndex = "1";
  // CSS3DRenderer 自体はポインターを奪わず、オブジェクト要素だけ pointerEvents:auto
  cssRenderer.domElement.style.pointerEvents = "none";
  stage.appendChild(cssRenderer.domElement);

  // --- WebGL overlay (avatar only) ---
  // CSS3D(DOM) はブラウザの重なり順の都合で WebGL より上に描画されます。
  // 「アバターがUIより手前」を成立させるため、アバターだけを透明キャンバスで上描きします。
  const avatarRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  avatarRenderer.setSize(window.innerWidth, window.innerHeight);
  avatarRenderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  avatarRenderer.setClearColor(0x000000, 0);
  avatarRenderer.domElement.style.position = "absolute";
  avatarRenderer.domElement.style.inset = "0";
  avatarRenderer.domElement.style.zIndex = "2";
  avatarRenderer.domElement.style.pointerEvents = "none";
  stage.appendChild(avatarRenderer.domElement);

  // --- Lights ---
  const ambient = new THREE.AmbientLight(0xffffff, 0.55);
  // レイヤー分割レンダリング（背景=0 / アバター=1）でも照明が効くように両方へ
  ambient.layers.enable(1);
  scene.add(ambient);

  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(3.5, 4.5, 2.5);
  key.layers.enable(1);
  scene.add(key);
  const rim = new THREE.DirectionalLight(0x88aaff, 0.35);
  rim.position.set(-4, 2.5, -3);
  rim.layers.enable(1);
  scene.add(rim);

  // --- Room geometry ---
  const roomW = 9.0;
  const roomH = 4.8;
  const roomD = 12.0;

  const roomWallDark = new THREE.MeshStandardMaterial({
    color: 0x1f1f24,
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.BackSide,
  });
  const roomWallOrange = new THREE.MeshStandardMaterial({
    color: 0xffa500, // bright orange
    roughness: 0.9,
    metalness: 0.02,
    side: THREE.BackSide,
  });
  const roomWallGray = new THREE.MeshStandardMaterial({
    color: 0x7a7a7a,
    roughness: 0.95,
    metalness: 0.0,
    side: THREE.BackSide,
  });

  const room = new THREE.Mesh(
    new THREE.BoxGeometry(roomW, roomH, roomD),
    // BoxGeometry material order: +x, -x, +y, -y, +z, -z
    [roomWallOrange, roomWallOrange, roomWallGray, roomWallGray, roomWallDark, roomWallDark]
  );
  room.position.set(0, roomH / 2, 0);
  scene.add(room);

  // floor accent
  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(roomW, roomD),
    new THREE.MeshStandardMaterial({ color: 0x6f6f6f, roughness: 1.0, metalness: 0.0 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.set(0, 0.001, 0);
  scene.add(floor);

  // --- Back wall frame (WebGL plane) ---
  const screenW = roomW * 0.86;
  const screenH = roomH * 0.62;
  const backZ = -roomD / 2 + 0.04;

  const screenFrame = new THREE.Mesh(
    new THREE.PlaneGeometry(screenW + 0.2, screenH + 0.2),
    new THREE.MeshStandardMaterial({ color: 0x09090b, roughness: 0.9, metalness: 0.1 })
  );
  screenFrame.position.set(0, roomH * 0.56, backZ + 0.005);
  scene.add(screenFrame);

  // --- UI as CSS3DObject (back wall) ---
  const uiObj = new CSS3DObject(uiDiv);
  // cssRenderer.domElement は pointer-events:none のため、UI本体は明示的に有効化
  uiDiv.style.pointerEvents = "auto";
  uiObj.position.set(0, roomH * 0.56, backZ);
  uiObj.rotation.y = 0; // 反転させると見えなくなることが多いのでまずは0
  cssScene.add(uiObj);

  function fitUI() {
    // elementがCSS3Dに移されるタイミングによって rect が 0 になる場合があるのでガード
    const rect = uiDiv.getBoundingClientRect();
    const pxW = Math.max(1, rect.width);
    const pxH = Math.max(1, rect.height);

    // UIの横幅を screenW に合わせて縮尺を決める（px → world units）
    const s = screenW / pxW;
    uiObj.scale.set(s, s, 1);

    // UIの縦をスクリーン枠に収める（縦がはみ出る場合はさらに縮める）
    const worldH = pxH * s;
    if (worldH > screenH) {
      const s2 = screenH / pxH;
      uiObj.scale.set(s2, s2, 1);
    }
  }

  // CSS3DRenderer が uiDiv を自分のDOMに移してから計測したいので2フレーム遅らせる
  requestAnimationFrame(() => requestAnimationFrame(fitUI));

  // --- Avatar (VRM) ---
  const avatarGroup = new THREE.Group();
  scene.add(avatarGroup);
  const setLayerRecursive = (root, layer) => {
    root.traverse((o) => o.layers.set(layer));
  };
  // アバターは前面キャンバスで描画するため layer=1 へ
  avatarGroup.layers.set(1);

  // まずは必ず見えるダミーを置く（VRMロード失敗でも「3Dが動いてる」ことが分かる）
  const dummy = new THREE.Mesh(
    new THREE.BoxGeometry(0.6, 1.4, 0.4),
    new THREE.MeshStandardMaterial({ color: 0x2ee59d, roughness: 0.6 })
  );
  dummy.position.set(0, 0.7, 2.8);
  avatarGroup.add(dummy);
  setLayerRecursive(dummy, 1);

  let vrm = null;
  const _tmpPos = new THREE.Vector3();
  const _tmpDir = new THREE.Vector3();
  try {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));

    vrm = await new Promise((resolve, reject) => {
      loader.load(
        AVATAR_URL,
        (gltf) => {
          const v = gltf.userData.vrm;
          if (!v) return reject(new Error("VRM parse failed (gltf.userData.vrm is null)"));
          resolve(v);
        },
        undefined,
        reject
      );
    });

    VRMUtils.removeUnnecessaryJoints(vrm.scene);

    // ダミーを消してVRMを置く
    avatarGroup.remove(dummy);

    // NOTE: VRM/VRoid系は「モデルの正面方向」が環境によって +Z / -Z どちらのこともあります。
    // ここでは「こちら（カメラ）を向いていない」問題を解消するため、まず 0 をデフォルトにします。
    // もし逆向きになった場合は、次の1行を Math.PI に戻してください。
    vrm.scene.rotation.y = 0;
    vrm.scene.position.set(0, 0, 2.8);

    // ちょいスケール調整（モデルによっては大き過ぎ/小さ過ぎる）
    vrm.scene.scale.setScalar(1.0);

    avatarGroup.add(vrm.scene);
    setLayerRecursive(vrm.scene, 1);

    console.info("[room] VRM loaded:", AVATAR_URL);
  } catch (e) {
    console.warn("[room] VRM load failed (fallback to dummy):", e);
    debug.append("VRM load failed. Using dummy avatar.\n" + String(e));
  }

  // カメラの方向へアバターのY軸だけ向ける（必要なら無効化可能）
  function updateAvatarFacing() {
    if (!vrm || !AVATAR_FACE_VIEWER) return;
    vrm.scene.getWorldPosition(_tmpPos);
    _tmpDir.subVectors(camera.position, _tmpPos);
    _tmpDir.y = 0;
    if (_tmpDir.lengthSq() < 1e-6) return;
    const desiredYaw = Math.atan2(_tmpDir.x, _tmpDir.z) + AVATAR_YAW_OFFSET;
    vrm.scene.rotation.y = lerpAngle(vrm.scene.rotation.y, desiredYaw, 0.12);
  }

  // --- Face tracking (optional) ---
  initFaceTracking(webcamEl);

  // --- Animation loop ---
  const clock = new THREE.Clock();

  function applyAvatarMotion(t, dt) {
    // ざっくり状態
    const speak = assistantSpeaking ? 1 : 0;
    const listen = Math.min(1, micRms * 10);

    // ダミー用（VRMがない時）
    if (!vrm) {
      dummy.position.y = 0.7 + 0.05 * listen + 0.04 * speak * Math.sin(t * 16);
      dummy.rotation.y = 0.15 * Math.sin(t * 0.8);
      return;
    }

    // VRMボーンで軽い身振り手振り
    const head = vrm.humanoid?.getNormalizedBoneNode("head");
    const jaw = vrm.humanoid?.getNormalizedBoneNode("jaw");
    const lArm = vrm.humanoid?.getNormalizedBoneNode("leftUpperArm");
    const rArm = vrm.humanoid?.getNormalizedBoneNode("rightUpperArm");

    if (head) {
      head.rotation.x = -0.10 * listen + 0.04 * speak * Math.sin(t * 10);
      head.rotation.y = 0.08 * Math.sin(t * 0.6);
    }

    // 口パク（超簡易）：喋ってる間だけ顎を動かす
    if (jaw) {
      const mouth = speak * (0.25 + 0.15 * Math.sin(t * 22));
      jaw.rotation.x = -mouth;
    }

    if (lArm) lArm.rotation.z = 0.35 * speak * Math.sin(t * 5.5);
    if (rArm) rArm.rotation.z = -0.35 * speak * Math.sin(t * 5.5);

    // VRM内部更新
    vrm.update(dt);
  }

  async function updateCameraByFace() {
    const now = performance.now();
    if (now - faceLastMs < FACE_SAMPLE_INTERVAL_MS) return;
    if (faceInFlight) return;
    faceLastMs = now;
    faceInFlight = true;

    try {
      const sample = await sampleFaceBox();
      if (!sample) {
        if (faceTrackerKind === "tfjs" && now - faceLastOkMs > 5000 && now - faceLastNoFaceLogMs > 5000) {
          faceLastNoFaceLogMs = now;
          console.warn(
            "[room] Face tracking active (tfjs) but no face detected. Check lighting, camera framing, and DevTools Network for model download errors."
          );
        }
        return;
      }

      faceLastOkMs = now;
      if (!faceLoggedFirstOk) {
        faceLoggedFirstOk = true;
        console.info("[room] Face detected; motion parallax should respond to head movement.");
      }

      const { cx, cy, w, vw, vh } = sample;

      // trompeloeil の式に寄せて「角度→カメラ位置」を計算する
      const kx = (2 * (cx - vw / 2)) / vw;
      const ky = (2 * (cy - vh / 2)) / vh;

      const hfov = faceFov?.hfov ?? THREE.MathUtils.degToRad(50);
      const vfov = faceFov?.vfov ?? THREE.MathUtils.degToRad(35);

      const ax = Math.atan(kx * Math.tan(hfov / 2));
      const ay = Math.atan(ky * Math.tan(vfov / 2));

      // 目安として ±30deg までに制限（画面外に大きく出た時の暴れ防止）
      const axC = THREE.MathUtils.clamp(ax, -0.52, 0.52);
      const ayC = THREE.MathUtils.clamp(ay, -0.52, 0.52);

      const tan1 = -Math.tan(axC);
      const tan2 = -Math.tan(ayC);

      // 距離は一定（PDFでも「距離推定はノイズが多いので固定」が説明されている）
      const d = camBase.z - camFocus.z;
      const z = Math.sqrt((d * d) / (1 + tan1 * tan1 + tan2 * tan2));

      // 顔サイズは「検出できているか」程度にのみ使用（急な跳ねの抑制）
      const faceOk = w > 20;
      const strength = faceOk ? 1 : 0.4;

      camTarget.set(
        camFocus.x + z * tan1 * strength,
        camFocus.y + (camBase.y - camFocus.y) + z * tan2 * strength,
        camFocus.z + z
      );
    } finally {
      faceInFlight = false;
    }
  }

	  function animate() {
	    requestAnimationFrame(animate);

    const dt = clock.getDelta();
    const t = clock.elapsedTime;

    // アバター
    applyAvatarMotion(t, dt);

    // 顔追従カメラ（任意）
    if (faceDetector) {
      // 非同期（重いので await しない）
      updateCameraByFace().catch(() => {});
    }

    // なめらかに追従
    camera.position.lerp(camTarget, 0.12);
    camera.lookAt(camFocus);

    // アバターをこちらへ向ける（視点移動時も追従）
    updateAvatarFacing();

    // 背景（部屋）: layer=0
    camera.layers.set(0);
    renderer.render(scene, camera);

	    // UI（DOM/CSS3D）: layer=0
	    cssRenderer.render(cssScene, camera);

	    // 前景（アバター）: layer=1 を透明キャンバスで上描き
	    // NOTE: overlay renderer で同じ scene を render すると scene.background が「不透明に」クリアされ、
	    // 下の WebGL(部屋) / CSS3D(UI) を覆ってしまう。ここでは overlay パスだけ background を無効化して透明に保つ。
	    const _bg = scene.background;
	    scene.background = null;
	    try {
	      camera.layers.set(1);
	      avatarRenderer.render(scene, camera);
	    } finally {
	      scene.background = _bg;
	    }

	    // 次フレームへ戻す
	    camera.layers.set(0);
	  }

  function onResize() {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    avatarRenderer.setSize(window.innerWidth, window.innerHeight);
    cssRenderer.setSize(window.innerWidth, window.innerHeight);
    fitUI();
  }

  window.addEventListener("resize", onResize);

  animate();
}

// 初期化タイミング：DOM構築後
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", () => {
    init().catch((e) => {
      console.error("[room] init failed:", e);
      debug.show("room.js init failed:\n" + String(e));
    });
  });
} else {
  init().catch((e) => {
    console.error("[room] init failed:", e);
    debug.show("room.js init failed:\n" + String(e));
  });
}
