import * as THREE from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
import { CSS3DRenderer, CSS3DObject } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/renderers/CSS3DRenderer.js";
import { GLTFLoader } from "https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js";

import { on } from "./bus.js";

const ROOM = { w: 10, h: 6, d: 14 };
const AVATAR_URL = "./assets/zundamon.vrm";

const state = {
  assistantSpeaking: false,
  micRms: 0,
  micRmsSmoothed: 0,
};

function clamp(v, min, max) {
  return Math.min(max, Math.max(min, v));
}

function safeLog(...args) {
  // eslint-disable-next-line no-console
  console.log("[room]", ...args);
}

init().catch((e) => safeLog("init failed:", e));

async function init() {
  const stage = document.getElementById("stage");
  if (!stage) return;

  const ui = document.getElementById("ui3d");
  const webcam = document.getElementById("webcam");

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(0x05070a);

  const cssScene = new THREE.Scene();

  const camera = new THREE.PerspectiveCamera(55, window.innerWidth / window.innerHeight, 0.1, 200);
  const baseCamPos = new THREE.Vector3(0, 1.6, 7.4);
  const camTargetPos = baseCamPos.clone();
  camera.position.copy(baseCamPos);

  const lookAt = new THREE.Vector3(0, 1.45, -2.2);
  camera.lookAt(lookAt);

  const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.domElement.style.position = "absolute";
  renderer.domElement.style.inset = "0";
  renderer.domElement.style.pointerEvents = "none";
  stage.appendChild(renderer.domElement);

  const cssRenderer = new CSS3DRenderer();
  cssRenderer.setSize(window.innerWidth, window.innerHeight);
  cssRenderer.domElement.style.position = "absolute";
  cssRenderer.domElement.style.inset = "0";
  cssRenderer.domElement.style.zIndex = "1";
  cssRenderer.domElement.style.pointerEvents = "auto";
  stage.appendChild(cssRenderer.domElement);

  // ---- Room (simple box) ----
  const roomGeom = new THREE.BoxGeometry(ROOM.w, ROOM.h, ROOM.d);
  const roomMat = new THREE.MeshStandardMaterial({ color: 0x0a0f16, side: THREE.BackSide, roughness: 1, metalness: 0 });
  const roomMesh = new THREE.Mesh(roomGeom, roomMat);
  roomMesh.position.set(0, ROOM.h / 2 - 0.2, 0);
  scene.add(roomMesh);

  const floor = new THREE.Mesh(
    new THREE.PlaneGeometry(ROOM.w * 1.05, ROOM.d * 1.05),
    new THREE.MeshStandardMaterial({ color: 0x070a0f, roughness: 1, metalness: 0 })
  );
  floor.rotation.x = -Math.PI / 2;
  floor.position.y = 0;
  scene.add(floor);

  const hemi = new THREE.HemisphereLight(0xaad3ff, 0x0b0c12, 0.65);
  scene.add(hemi);
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(3, 6, 6);
  scene.add(key);

  // ---- UI on the back wall (CSS3D) ----
  if (ui) {
    const uiObj = new CSS3DObject(ui);
    cssScene.add(uiObj);
    uiObj.position.set(0, 1.55, -ROOM.d / 2 + 0.02);
    uiObj.rotation.y = Math.PI;
    uiObj.rotation.x = 0;
    uiObj.rotation.z = 0;

    const fitUiToWall = () => {
      const rect = ui.getBoundingClientRect();
      if (!rect.width || !rect.height) return;
      const wallW = ROOM.w * 0.92;
      const wallH = ROOM.h * 0.72;
      const sx = wallW / rect.width;
      const sy = wallH / rect.height;
      const s = Math.min(sx, sy) * 0.98;
      uiObj.scale.set(s, s, s);
    };
    requestAnimationFrame(fitUiToWall);
    window.addEventListener("resize", fitUiToWall, { passive: true });
  }

  // ---- Avatar (VRM or dummy) ----
  const avatarRoot = new THREE.Group();
  avatarRoot.position.set(0, 0, 2.45);
  scene.add(avatarRoot);

  const { updateAvatar, vrm } = await loadAvatar(avatarRoot);

  // ---- Bus events ----
  on("assistant:speakingStart", () => {
    state.assistantSpeaking = true;
  });
  on("assistant:speakingEnd", () => {
    state.assistantSpeaking = false;
  });
  on("user:micRms", ({ rms }) => {
    state.micRms = typeof rms === "number" ? rms : 0;
  });

  // ---- Face tracking (optional; falls back automatically) ----
  let face = null;
  if (webcam) {
    face = await tryStartFaceTracking(webcam, {
      baseCamPos,
      camTargetPos,
    });
  }

  // ---- Render loop ----
  const clock = new THREE.Clock();

  const onResize = () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
    cssRenderer.setSize(window.innerWidth, window.innerHeight);
  };
  window.addEventListener("resize", onResize, { passive: true });

  function tick() {
    const dt = clock.getDelta();
    const t = clock.elapsedTime;

    state.micRmsSmoothed = state.micRmsSmoothed * 0.88 + state.micRms * 0.12;
    const userTalking = state.micRmsSmoothed > 0.055;
    const mode = state.assistantSpeaking ? "speaking" : (userTalking ? "listening" : "idle");

    if (face) face.update();
    camera.position.lerp(camTargetPos, 0.08);
    camera.lookAt(lookAt);

    updateAvatar({ dt, t, mode, mic: state.micRmsSmoothed, vrm });

    renderer.render(scene, camera);
    cssRenderer.render(cssScene, camera);

    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

async function loadAvatar(parent) {
  const avatarRig = new THREE.Group();
  parent.add(avatarRig);

  const dummy = createDummyAvatar();
  avatarRig.add(dummy);

  let vrm = null;
  let bones = null;

  try {
    const { VRMLoaderPlugin, VRMUtils } = await import("https://cdn.jsdelivr.net/npm/@pixiv/three-vrm@2.0.0/+esm");
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));

    vrm = await new Promise((resolve, reject) => {
      loader.load(
        AVATAR_URL,
        (gltf) => resolve(gltf.userData.vrm || null),
        undefined,
        reject
      );
    });

    if (!vrm) throw new Error("VRM plugin did not produce vrm");

    try { VRMUtils.removeUnnecessaryVertices(vrm.scene); } catch {}
    try { VRMUtils.removeUnnecessaryJoints(vrm.scene); } catch {}
    try { VRMUtils.rotateVRM0(vrm); } catch {}

    dummy.visible = false;
    avatarRig.add(vrm.scene);
    vrm.scene.position.set(0, 0, 0);
    vrm.scene.rotation.y = Math.PI;
    vrm.scene.scale.setScalar(1.0);

    bones = captureVrmBones(vrm);
    safeLog("VRM loaded:", AVATAR_URL);
  } catch (e) {
    safeLog("VRM not available, using dummy:", e?.message || e);
  }

  const updateAvatar = ({ dt, t, mode, mic, vrm }) => {
    if (vrm) vrm.update(dt);
    animateAvatar({ t, mode, mic, dummy, vrm, bones, rig: avatarRig });
  };

  return { updateAvatar, vrm };
}

function createDummyAvatar() {
  const group = new THREE.Group();
  group.position.set(0, 0, 0);

  const body = new THREE.Mesh(
    new THREE.SphereGeometry(0.42, 32, 24),
    new THREE.MeshStandardMaterial({ color: 0x38e5b6, roughness: 0.35, metalness: 0.05 })
  );
  body.position.y = 1.05;
  group.add(body);

  const face = new THREE.Mesh(
    new THREE.SphereGeometry(0.28, 28, 20),
    new THREE.MeshStandardMaterial({ color: 0x7ff3d3, roughness: 0.45, metalness: 0.02 })
  );
  face.position.y = 1.6;
  face.position.z = 0.08;
  group.add(face);

  const armL = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.07, 0.52, 8, 18),
    new THREE.MeshStandardMaterial({ color: 0x2bcaa2, roughness: 0.35, metalness: 0.05 })
  );
  armL.position.set(-0.42, 1.12, 0.05);
  armL.rotation.z = 0.9;
  group.add(armL);

  const armR = armL.clone();
  armR.position.x = 0.42;
  armR.rotation.z = -0.9;
  group.add(armR);

  group.userData = { body, face, armL, armR };
  return group;
}

function captureVrmBones(vrm) {
  const get = (name) => {
    try { return vrm.humanoid?.getNormalizedBoneNode(name) || null; } catch { return null; }
  };
  const nodes = {
    head: get("head"),
    neck: get("neck"),
    chest: get("chest"),
    spine: get("spine"),
    leftUpperArm: get("leftUpperArm"),
    rightUpperArm: get("rightUpperArm"),
  };

  const base = {};
  for (const [k, node] of Object.entries(nodes)) {
    if (node) base[k] = node.quaternion.clone();
  }
  return { nodes, base };
}

function animateAvatar({ t, mode, mic, dummy, vrm, bones, rig }) {
  const breath = Math.sin(t * 1.2) * 0.03;
  const sway = Math.sin(t * 0.7) * 0.12;

  let nod = 0;
  let talk = 0;
  let wave = 0;

  if (mode === "idle") {
    nod = Math.sin(t * 1.4) * 0.05;
    talk = 0;
    wave = 0;
  } else if (mode === "listening") {
    nod = Math.sin(t * 3.0) * 0.18 + clamp(mic * 0.9, 0, 0.22);
    talk = 0;
    wave = 0.1 + Math.sin(t * 1.1) * 0.08;
  } else if (mode === "speaking") {
    nod = Math.sin(t * 4.2) * 0.11;
    talk = 0.2 + Math.sin(t * 6.0) * 0.25;
    wave = 0.45 + Math.sin(t * 2.7) * 0.25;
  }

  rig.position.y = 0 + breath * (mode === "speaking" ? 1.3 : 1.0);
  rig.rotation.y = sway * (mode === "speaking" ? 1.2 : 0.8);

  if (dummy?.visible) {
    const { body, face, armL, armR } = dummy.userData || {};
    if (body) body.position.y = 1.05 + breath * 0.8;
    if (face) face.rotation.x = -nod;
    if (armL) armL.rotation.x = -wave;
    if (armR) armR.rotation.x = -wave;
    return;
  }

  if (!vrm || !bones) return;
  const { nodes, base } = bones;

  const apply = (key, euler) => {
    const node = nodes[key];
    if (!node || !base[key]) return;
    node.quaternion.copy(base[key]);
    const q = new THREE.Quaternion().setFromEuler(euler);
    node.quaternion.multiply(q);
  };

  apply("spine", new THREE.Euler(-nod * 0.22, 0, 0));
  apply("chest", new THREE.Euler(-nod * 0.35, 0, 0));
  apply("neck", new THREE.Euler(-nod * 0.45, 0, 0));
  apply("head", new THREE.Euler(-nod * 0.85, Math.sin(t * 0.8) * 0.08, 0));

  apply("leftUpperArm", new THREE.Euler(-wave * 0.7, 0, 0.38 + talk * 0.08));
  apply("rightUpperArm", new THREE.Euler(-wave * 0.7, 0, -0.38 - talk * 0.08));
}

async function tryStartFaceTracking(video, { baseCamPos, camTargetPos }) {
  if (!navigator.mediaDevices?.getUserMedia) return null;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
    video.srcObject = stream;
    await video.play().catch(() => {});
  } catch (e) {
    safeLog("face tracking: getUserMedia rejected:", e?.message || e);
    return null;
  }

  const fld = window.faceLandmarksDetection;
  if (!fld?.createDetector) {
    safeLog("face tracking: face-landmarks-detection not loaded; fixed camera");
    return null;
  }

  let detector = null;
  try {
    const model = fld.SupportedModels?.MediaPipeFaceMesh || "MediaPipeFaceMesh";
    detector = await fld.createDetector(model, {
      runtime: "mediapipe",
      maxFaces: 1,
      refineLandmarks: false,
      solutionPath: "https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh",
    });
  } catch (e) {
    safeLog("face tracking: detector init failed:", e?.message || e);
    return null;
  }

  const target = baseCamPos.clone();
  let last = 0;
  let hasFace = false;
  let running = false;

  const update = async () => {
    const now = performance.now();
    if (running) return;
    if (now - last < 120) return;
    last = now;
    running = true;

    let faces = [];
    try {
      faces = await detector.estimateFaces(video, { flipHorizontal: true });
    } catch {
      running = false;
      return;
    }
    running = false;
    if (!faces?.length) {
      hasFace = false;
      target.copy(baseCamPos);
      camTargetPos.copy(target);
      return;
    }
    hasFace = true;

    const { cx, cy, w } = faceBoxFromEstimate(faces[0], video.videoWidth || 640, video.videoHeight || 480);
    const dx = (cx - 0.5);
    const dy = (cy - 0.5);

    const offsetX = clamp(-dx * 0.85, -0.6, 0.6);
    const offsetY = clamp(-dy * 0.45, -0.35, 0.35);
    const offsetZ = clamp((0.22 - w) * 3.2, -0.55, 0.55);

    target.set(baseCamPos.x + offsetX, baseCamPos.y + offsetY, baseCamPos.z + offsetZ);
    camTargetPos.lerp(target, 0.35);
  };

  // Wrap async update to avoid unhandled rejections inside RAF loop
  const safeUpdate = () => { update().catch(() => {}); };

  safeLog("face tracking: active", hasFace ? "(face detected)" : "");
  return { update: safeUpdate };
}

function faceBoxFromEstimate(face, vw, vh) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  const b = face?.box;
  if (b && typeof b === "object") {
    const xMin = b.xMin ?? b.left ?? b.x ?? b[0];
    const yMin = b.yMin ?? b.top ?? b.y ?? b[1];
    const xMax = b.xMax ?? b.right ?? ((typeof xMin === "number" && typeof b.width === "number") ? (xMin + b.width) : undefined);
    const yMax = b.yMax ?? b.bottom ?? ((typeof yMin === "number" && typeof b.height === "number") ? (yMin + b.height) : undefined);
    if ([xMin, yMin, xMax, yMax].every((v) => typeof v === "number")) {
      minX = xMin; minY = yMin; maxX = xMax; maxY = yMax;
    }
  }

  if (!Number.isFinite(minX) && Array.isArray(face?.keypoints)) {
    for (const kp of face.keypoints) {
      const x = kp.x ?? kp[0];
      const y = kp.y ?? kp[1];
      if (typeof x !== "number" || typeof y !== "number") continue;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x);
      maxY = Math.max(maxY, y);
    }
  }

  if (!Number.isFinite(minX)) {
    minX = vw * 0.35; maxX = vw * 0.65;
    minY = vh * 0.25; maxY = vh * 0.75;
  }

  const cx = ((minX + maxX) / 2) / Math.max(vw, 1);
  const cy = ((minY + maxY) / 2) / Math.max(vh, 1);
  const w = (maxX - minX) / Math.max(vw, 1);
  return { cx, cy, w };
}
