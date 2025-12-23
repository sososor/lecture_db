// frontend/app.js（認証・多ユーザー対応）
// - ログイン必須：最初に /me を確認し、未ログインなら /auth.html へ遷移
// - /auth/ws-token で短命トークンを取得して WebSocket に接続（?t=...）
// - 互換のため、WS 接続直後に {user_id, display_name} も送る（旧サーバ実装でも動く）
// - /chat は同一オリジン・ログインセッション前提
// - マイクは AudioWorklet で 48kHz Int16 PCM を生成し、サーバ側で 16kHz へダウンサンプル

import { emit } from "./bus.js";

let ws, audioCtx, micSource, workletNode, isConnecting = false;
let isMicMuted = false;
let playbackQueue = [];
let isProcessingQueue = false;
let currentPlaybackSource = null;
let currentUser = null; // { id, display_name }
let isAssistantSpeaking = false;

const $ = (id) => document.getElementById(id);
const log = (t) => { const el = $("log"); if(!el) return; el.textContent += t + "\n"; el.scrollTop = el.scrollHeight; };

// ===== 認証チェック（未ログインなら /auth.html へ） =====
(async function ensureAuthed(){
  try {
    const r = await fetch("/me", { credentials: "same-origin" });
    if (!r.ok) throw new Error("not authed");
    currentUser = await r.json(); // {id, display_name}
  } catch {
    location.replace("/auth.html");
    return; // 以降は実行されない
  }
})();

// ====== UI 取得 ======
const btnConnect = $("btnConnect");
const btnDisconnect = $("btnDisconnect");
const btnMute = $("btnMute");
updateMuteButtonUI();

btnConnect?.addEventListener("click", async () => {
  if (isConnecting || (ws && ws.readyState === WebSocket.OPEN)) return;
  isConnecting = true; btnConnect.disabled = true; btnDisconnect && (btnDisconnect.disabled = false);
  btnMute && (btnMute.disabled = true);
  try {
    await startAudio();
    await openWS();
    updateMuteButtonUI();
  } catch (e) {
    log("❌ 接続エラー: " + (e?.message || e));
    stopAudio();
  } finally {
    isConnecting = false; btnConnect.disabled = false;
  }
});

btnDisconnect?.addEventListener("click", () => {
  try { if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) ws.close(); } catch {}
  stopAudio();
});

btnMute?.addEventListener("click", () => {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  setMuteState(!isMicMuted);
});

// ====== WebSocket 接続（/auth/ws-token → /ws?t=...） ======
async function openWS(){
  // 1) 短命トークンを取得
  const tokRes = await fetch("/auth/ws-token", { method: "POST", credentials: "same-origin" });
  if (!tokRes.ok) throw new Error("/auth/ws-token 失敗");
  const tok = await tokRes.json(); // { token, id, display_name }

  // 2) WS を開く（同一オリジン）
  const WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws?t=" + encodeURIComponent(tok.token);
  await new Promise((resolve, reject) => {
    ws = new WebSocket(WS_URL);
    ws.binaryType = "arraybuffer";

    ws.onopen = () => {
      // 旧サーバ実装との互換用：最初に user 情報を送る
      try {
        ws.send(JSON.stringify({
          user_id: (currentUser?.id || tok.id || "user"),
          display_name: (currentUser?.display_name || tok.display_name || "WebUser")
        }));
      } catch {}
      log("WS connected");
      emit("ws:open");
      resolve();
    };

    ws.onerror = () => reject(new Error("WebSocket error"));

    ws.onclose = () => {
      log("WS closed");
      emit("ws:close");
      stopAudio();
    };

    ws.onmessage = async (ev) => {
      try {
        if (typeof ev.data === "string") {
          const msg = JSON.parse(ev.data);
          if (msg.type === "partialText") {
            log("Gemini ▶ " + msg.text);
            emit("assistant:partialText", { text: msg.text });
          }
          else if (msg.type === "error") {
            log("❌ " + (msg.message || "Gemini エラー"));
            stopAudio();
          }
        } else {
          // WAV バイナリ（Blob or ArrayBuffer）を受信 → 再生キューへ積む
          const arrBuf = ev.data instanceof Blob ? await ev.data.arrayBuffer() : ev.data;
          enqueueAudioChunk(arrBuf);
        }
      } catch (e) {
        log("再生エラー: " + (e?.message || e));
      }
    };
  });
}

async function decodeWavToAudioBuffer(arrayBuffer){
  // Safari 対策：ArrayBuffer をコピーして渡す
  const copy = arrayBuffer.slice(0);
  try {
    return await audioCtx.decodeAudioData(copy);
  } catch (e) {
    // 一部環境で decodeAudioData が Promise でない実装の後方互換処理
    return new Promise((resolve, reject) => {
      audioCtx.decodeAudioData(copy, resolve, reject);
    });
  }
}

// ====== マイク開始（AudioWorklet → Int16 PCM を main-thread へ） ======
async function startAudio(){
  if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
  if (audioCtx.state === "suspended") await audioCtx.resume();

  await audioCtx.audioWorklet.addModule("./pcm-worklet.js");
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  micSource = audioCtx.createMediaStreamSource(stream);
  workletNode = new AudioWorkletNode(audioCtx, "pcm-worklet");

  workletNode.port.onmessage = (ev) => {
    const pcm16 = ev.data; // Int16Array (48kHz, mono)
    try {
      const n = pcm16?.length || 0;
      if (n > 0) {
        let sum = 0;
        for (let i = 0; i < n; i++) {
          const v = pcm16[i] / 32768;
          sum += v * v;
        }
        const rms = Math.sqrt(sum / n);
        emit("user:micRms", { rms });
      }
    } catch {}
    if (!isMicMuted && ws && ws.readyState === WebSocket.OPEN) ws.send(pcm16.buffer);
  };

  // ハウリング防止：録音のみ。出力にはつながない
  micSource.connect(workletNode);
}

function stopAudio(){
  try { if (micSource) micSource.disconnect(); } catch {}
  try { if (workletNode) workletNode.disconnect(); } catch {}
  micSource = null; workletNode = null;
  setMuteState(false);
  if (isAssistantSpeaking) {
    isAssistantSpeaking = false;
    emit("assistant:speakingEnd");
  }
  if (currentPlaybackSource) {
    try { currentPlaybackSource.stop(); } catch {}
    currentPlaybackSource = null;
  }
  playbackQueue.length = 0;
  isProcessingQueue = false;
  const ctx = audioCtx;
  audioCtx = null;
  if (ctx) { try { ctx.close(); } catch {} }
  updateMuteButtonUI();
}

// ====== テキスト送信（ログインセッションからユーザー決定） ======
$("chatForm")?.addEventListener("submit", async (e) => {
  e.preventDefault();
  const input = $("chatInput");
  const text = input.value.trim();
  if (!text) return;

  try {
    const r = await fetch("/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }) // user_id はサーバ側セッションで判断
    });
    const data = await r.json().catch(()=> ({}));
    if(!r.ok){
      const err = data.error || data.detail || data.reply || `${r.status} ${r.statusText}`;
      log("❌ /chat エラー: " + err);
      const div = $("chatArea");
      div.innerHTML += `<p><b>あなた:</b> ${escapeHtml(text)}</p>` +
                       `<p><b>Bot:</b> ${escapeHtml(err)}</p>`;
      return;
    }
    const div = $("chatArea");
    div.innerHTML += `<p><b>あなた:</b> ${escapeHtml(text)}</p>` +
                     `<p><b>Bot:</b> ${escapeHtml(data.reply || "")}</p>`;
  } catch (e) {
    log("❌ /chat エラー: " + (e?.message || e));
  } finally {
    input.value = "";
  }
});

// ====== ユーティリティ ======
function escapeHtml(s){
  return String(s).replace(/[&<>"']/g, (c) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  })[c]);
}

function setMuteState(muted){
  isMicMuted = muted;
  updateMuteButtonUI();
  log(muted ? "🔇 マイクをミュートしました" : "🔈 マイクのミュートを解除しました");
}

function updateMuteButtonUI(){
  if (!btnMute) return;
  const isWsOpen = ws && ws.readyState === WebSocket.OPEN;
  btnMute.disabled = !isWsOpen;
  btnMute.textContent = isMicMuted ? "🔈 ミュート解除" : "🔇 ミュート";
  btnMute.setAttribute("aria-pressed", isMicMuted ? "true" : "false");
}

// ページ離脱時のクリーンアップ
window.addEventListener("beforeunload", () => {
  try { if (ws) ws.close(); } catch {}
  stopAudio();
});

function enqueueAudioChunk(arrayBuffer){
  if (!audioCtx) return;
  playbackQueue.push(arrayBuffer.slice(0)); // keep a copy per chunk
  if (!isProcessingQueue) processPlaybackQueue().catch((e) => {
    log("再生キューエラー: " + (e?.message || e));
  });
}

async function processPlaybackQueue(){
  if (!audioCtx) {
    playbackQueue.length = 0;
    return;
  }
  isProcessingQueue = true;
  if (!isAssistantSpeaking) {
    isAssistantSpeaking = true;
    emit("assistant:speakingStart");
  }
  try {
    while (audioCtx && playbackQueue.length > 0) {
      const chunk = playbackQueue.shift();
      const buf = await decodeWavToAudioBuffer(chunk);
      try {
        await playAudioBufferSequentially(buf);
      } finally {
        notifyTtsPlaybackFinished();
      }
    }
  } finally {
    isProcessingQueue = false;
    if (isAssistantSpeaking) {
      isAssistantSpeaking = false;
      emit("assistant:speakingEnd");
    }
  }
}

function playAudioBufferSequentially(buffer){
  return new Promise((resolve, reject) => {
    if (!audioCtx) {
      resolve();
      return;
    }
    try {
      const src = audioCtx.createBufferSource();
      currentPlaybackSource = src;
      src.buffer = buffer;
      src.connect(audioCtx.destination);
      src.onended = () => {
        if (currentPlaybackSource === src) currentPlaybackSource = null;
        resolve();
      };
      src.start();
    } catch (err) {
      if (currentPlaybackSource) currentPlaybackSource.disconnect?.();
      currentPlaybackSource = null;
      reject(err);
    }
  });
}

function notifyTtsPlaybackFinished(){
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  try {
    ws.send(JSON.stringify({ type: "ttsAck" }));
  } catch (e) {
    log("TTS ACK 送信エラー: " + (e?.message || e));
  }
}
