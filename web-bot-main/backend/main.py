# backend/main.py
import os, asyncio, json, time, io, wave, contextlib, re, random
from datetime import datetime, timezone, timedelta
from typing import Optional, Deque
from collections import deque

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import aiohttp
import psycopg
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool
# --- 重要: 72バイト制限を回避するため bcrypt_sha256 を使用 ---
from passlib.hash import bcrypt_sha256 as bcrypt
from itsdangerous import TimestampSigner, BadSignature, SignatureExpired

# ====== ここから bot.py 由来の共通設定を “ほぼそのまま” ======
# Gemini API key must be provided via GEMINI_API_KEY. GOOGLE_API_KEY is deliberately ignored.
_api_key_env = os.getenv("GEMINI_API_KEY")
_placeholder_markers = (
    "PUT_YOUR_GEMINI_API_KEY_HERE",
    "YOUR_GEMINI_API_KEY_HERE",
    "YOUR_API_KEY_HERE",
)
if not _api_key_env or any(m in _api_key_env for m in _placeholder_markers):
    suffix = " GOOGLE_API_KEY is no longer supported." if os.getenv("GOOGLE_API_KEY") else ""
    raise RuntimeError(
        "Gemini API key is not set. Please export GEMINI_API_KEY before starting the backend."
        + suffix
    )
GEMINI_API_KEY = _api_key_env
# NOTE: 'gemini-live-2.5-flash-preview' was deprecated (ended 2025-12-09).
# We keep the response modality as TEXT and synthesize audio via VOICEVOX.
MODEL_ID = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-native-audio-preview-12-2025")
# テキストモデルは環境変数優先で、利用可能なものに自動フォールバックする
TEXT_MODEL_ID = os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash")
_TEXT_MODEL_CANDIDATES = []
if TEXT_MODEL_ID:
    _TEXT_MODEL_CANDIDATES.append(TEXT_MODEL_ID)
# 新しい順にいくつか候補を並べておく（重複は除外）
for _cand in [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash-latest",
]:
    if _cand not in _TEXT_MODEL_CANDIDATES:
        _TEXT_MODEL_CANDIDATES.append(_cand)
VOICEVOX_URL = os.getenv("VOICEVOX_URL") or os.getenv("VOICEVOX_HOST") or "http://127.0.0.1:50021"
VOICEVOX_SPEAKER = int(os.getenv("VOICEVOX_SPEAKER", "3"))

DISCORD_PCM_RATE = 48000
LIVE_PCM_RATE = 16000
LIVE_PCM_MIME = f"audio/pcm;rate={LIVE_PCM_RATE}"

VAD_SILENCE_MS = int(os.getenv("VAD_SILENCE_MS", "200"))
CHUNK_TARGET_MS = int(os.getenv("CHUNK_TARGET_MS", "160"))
EOS_IDLE_SEC = float(os.getenv("EOS_IDLE_SEC", "1.0"))

SENT_END_CHARS = os.getenv("SENT_END_CHARS", "。．！？!?\n")
TRAILING_CLOSERS = os.getenv("TRAILING_CLOSERS", "』」】）》］）】』”’\"")
SENTENCE_IDLE_MS = int(os.getenv("SENTENCE_IDLE_MS", "150"))
TAIL_FORCE_FLUSH_MS = int(os.getenv("TAIL_FORCE_FLUSH_MS", "900"))

SYSTEM_PERSONALITY = os.getenv(
    "SYSTEM_PERSONALITY",
    "どんな質問にもわかりやすく、まとまった考えで100文字以内で答えてください。"
)

SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me")
WS_TOKEN_TTL = int(os.getenv("WS_TOKEN_TTL", "60"))  # WebSocket 接続トークンの寿命（秒）
_signer = TimestampSigner(SESSION_SECRET)
LIVE_MODEL_AVAILABLE = True  # Gemini Live を利用できるかどうか（起動時チェックで更新）
TEXT_RETRY_MAX = int(os.getenv("TEXT_RETRY_MAX", "3"))  # テキスト生成時のリトライ回数（モデルごと）
TEXT_RETRY_BASE_DELAY = float(os.getenv("TEXT_RETRY_BASE_DELAY", "0.6"))  # リトライ時の初期待機（秒）

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
log = logging.getLogger("web-bot")

def _jst_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat()

# ====== RAG（data/ 配下のテキストをシンプルに注入） ======
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAG_MAX_TOTAL_CHARS = int(os.getenv("RAG_MAX_TOTAL_CHARS", "7000"))   # システムプロンプトに載せる総文字数上限
RAG_PER_FILE_CHARS = int(os.getenv("RAG_PER_FILE_CHARS", "1800"))     # 1ファイルあたりの抜粋上限
RAG_FILE_SIZE_LIMIT = int(os.getenv("RAG_FILE_SIZE_LIMIT", str(2 * 1024 * 1024)))  # 既定: 2MB まで読み込み
RAG_ALLOWED_EXTS = {".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".yaml", ".yml", ".log"}


def _load_rag_context(max_total_chars: int = RAG_MAX_TOTAL_CHARS) -> str:
    """
    data/ 直下に置いたテキストファイルを読み、システムプロンプトへ渡すための
    簡易RAGコンテキスト文字列を返す。
    - UTF-8 で読み、読めない場合はスキップ
    - 1ファイルあたり RAG_PER_FILE_CHARS まで抜粋
    - 総量が max_total_chars を超えない範囲で連結
    """
    if max_total_chars <= 0:
        return ""
    if not os.path.isdir(DATA_DIR):
        return ""

    parts: list[str] = []
    budget = max_total_chars
    per_file_limit = max(300, min(RAG_PER_FILE_CHARS, max_total_chars))

    for name in sorted(os.listdir(DATA_DIR)):
        path = os.path.join(DATA_DIR, name)
        if not os.path.isfile(path):
            continue
        ext = os.path.splitext(name)[1].lower()
        if ext and ext not in RAG_ALLOWED_EXTS:
            # 拡張子不明でもテキストなら読めることがあるので完全には弾かない
            pass
        try:
            if os.path.getsize(path) > RAG_FILE_SIZE_LIMIT:
                log.warning("RAG: %s is larger than limit (%d bytes), truncating head.", name, RAG_FILE_SIZE_LIMIT)
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                snippet = f.read(per_file_limit)
        except Exception as e:
            log.warning("RAG: failed to read %s: %s", name, e)
            continue
        snippet = (snippet or "").strip()
        if not snippet:
            continue
        block = f"### {name}\n{snippet}"
        if budget - len(block) < -200:  # 残りがほぼ無い場合は打ち切り
            break
        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            break
    return "\n\n".join(parts)


def _compose_system_instruction(
    base_personality: str,
    history_excerpt: str | None,
    display_name: str | None = None,
) -> str:
    """
    SYSTEM_PERSONALITY に RAG コンテキストと会話履歴を重ねたシステムプロンプトを生成。
    """
    parts = [base_personality]
    # Streaming時に思考過程を口頭で読み上げるのを防ぐため、
    # モデルへ「最終的な返答だけを出力する」ことを強く指示する。
    parts.append(
        """
ユーザーに渡す出力は最終回答の本文だけにしてください。分析・方針・推論メモ・段階的な説明・見出しや箇条書きの「考え中の内容」を一切出力しないでください。回答は簡潔な自然な文章のみで、余計な接頭語やナンバリング、メタコメント（例: '考えています', '方針', 'Exploring ...', 'Refining ...' など）を含めてはいけません。
""".strip()
    )
    rag = _load_rag_context()
    if rag:
        parts.append(
            f"""以下は data/ ディレクトリに配置された外部知識ベースの抜粋です。
この内容を根拠として優先的に活用し、不足時はその旨を伝えてください。引用は要約で構いません。
{rag}"""
        )
    if history_excerpt:
        name = display_name or "ユーザー"
        parts.append(
            f"""あなたは{name}との継続会話を担当します。以下はこれまでの会話ログの抜粋（JSONL）：
{history_excerpt}

この文脈を踏まえて、一貫性のある返答を心がけてください。"""
        )
    return "\n\n".join(parts)


def _looks_like_meta_sentence(text: str) -> bool:
    """
    Heuristic filter: drop model self-talk / analysis headings that should not be
    spoken. Returns True if the sentence looks like meta commentary.
    """
    t = text.strip()
    if not t:
        return True
    # Strip simple markdown-style emphasis or numbering
    t = re.sub(r"^[#>*\\s]+", "", t)
    t = re.sub(r"^[0-9]+\\s*[\\.．、]\\s*", "", t)
    t = t.strip("*_`~ ")
    lowered = t.lower()
    meta_keywords = [
        "exploring", "analysis", "analyzing", "reasoning", "thoughts",
        "refining", "plan", "planning", "approach", "strategy",
        "summary of reasoning", "deliberation", "responding to",
        "response strategy", "bridging", "re-engage", "reengage",
        "preference", "context", "greeting", "user's", "users",
        "will provide", "i will", "i'll", "going to", "aim to", "focus on",
        "next, i", "now i", "intent", "meta", "note:", "analysis:",
    ]
    jp_meta = ["考察", "分析", "方針", "計画", "思考", "理由", "戦略"]
    if any(k in lowered for k in meta_keywords):
        return True
    if any(k in t for k in jp_meta):
        return True
    # Short title-like headings without sentence-ending punctuation are likely meta
    if len(t) <= 80 and not re.search(r"[。．！？!?…]", t) and not re.search(r"[。！？!?]$", t):
        if not re.search(r"[。！？!?]", t) and " " in t:
            return True
    # If the text references "user" and "response/request/topic", treat as meta
    if ("user" in lowered or "ユーザー" in t) and any(w in lowered for w in ["response", "request", "topic", "preference", "greeting"]):
        return True
    # Parenthesized snippets (often inline translations) without CJK are likely meta
    if t.startswith("(") and not re.search(r"[ぁ-んァ-ン一-龠]", t):
        return True
    # If the text is very short and non-CJK, likely not the final answer.
    if len(t) <= 4 and not re.search(r"[ぁ-んァ-ン一-龠]", t):
        return True
    return False

# --- transient error helper (for text chat retries) ---
_TRANSIENT_STATUS_NAMES = {"UNAVAILABLE", "RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED", "ABORTED"}
_TRANSIENT_HTTP_CODES = {429, 500, 503, 504}

def _is_retryable_genai_error(err: Exception) -> bool:
    """
    Heuristic判定で「しばらく待てば成功する」タイプのエラーを拾う。
    google-genai 1.x の例外はコードやステータスの持ち方が一定でないため、
    属性とメッセージ文字列の両方をゆるくチェックする。
    """
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if hasattr(code, "name"):  # grpc.StatusCode など
        if code.name in _TRANSIENT_STATUS_NAMES:
            return True
    if isinstance(code, str) and code.upper() in _TRANSIENT_STATUS_NAMES:
        return True
    if isinstance(code, int) and code in _TRANSIENT_HTTP_CODES:
        return True
    status = getattr(err, "status", None)
    if isinstance(status, str) and status.upper() in _TRANSIENT_STATUS_NAMES:
        return True
    msg = str(err).upper()
    keywords = ["UNAVAILABLE", "OVERLOADED", "RESOURCE_EXHAUSTED", "TRY AGAIN LATER", "BACKEND ERROR"]
    if any(k in msg for k in keywords):
        return True
    return False

# ---- VOICEVOX HTTP TTS（同じ） ----
async def voicevox_tts_async(text: str, speaker: int = VOICEVOX_SPEAKER) -> bytes:
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(f"{VOICEVOX_URL}/audio_query",
                                params={"text": text, "speaker": speaker}) as r:
            r.raise_for_status(); query = await r.json()
        async with session.post(f"{VOICEVOX_URL}/synthesis",
                                params={"speaker": speaker}, json=query) as r2:
            r2.raise_for_status(); return await r2.read()

# ---- 48k → 16k mono（同じ） ----
def _downsample_48k_to_16k_mono(pcm48: bytes) -> bytes:
    if not pcm48: return b""
    a = np.frombuffer(pcm48, dtype="<i2")
    if a.size >= 2 and (a.size % 2 == 0):
        stereo = a.reshape(-1, 2); mono = stereo.mean(axis=1).astype(np.int16)
    else:
        mono = a
    mono16 = mono[::3].astype(np.int16)
    return mono16.tobytes()

def _estimate_wav_duration(wav_bytes: bytes) -> float:
    """Approximate playback duration (seconds) for a WAV blob."""
    if not wav_bytes:
        return 0.0
    try:
        with contextlib.closing(wave.open(io.BytesIO(wav_bytes))) as wf:
            frames = wf.getnframes()
            rate = wf.getframerate() or 1
            return frames / max(rate, 1)
    except Exception:
        return 0.0


# ====== Gemini API 事前チェック ======
def _probe_gemini_models() -> bool:
    """
    Gemini APIキー・モデルが有効か確認する。
    - TEXT_MODEL_ID は候補リストから順に試し、利用可能な最初のものを採用する。
    - 戻り値は Live モデル利用可否。
    """
    global TEXT_MODEL_ID
    client = genai.Client(api_key=GEMINI_API_KEY)

    def _get(model_id: str):
        # models.get は無料で、キー・モデルの有効性チェックに使える
        client.models.get(model=model_id)

    # テキストモデル：候補から探す
    text_errors: list[str] = []
    selected = None
    for cand in _TEXT_MODEL_CANDIDATES:
        try:
            _get(cand)
            selected = cand
            break
        except Exception as e:
            text_errors.append(f"{cand}: {e}")
    if not selected:
        raise RuntimeError(
            "No usable Gemini text model found. Tried: " + ", ".join(text_errors)
        )
    if selected != TEXT_MODEL_ID:
        log.warning("TEXT_MODEL_ID '%s' unavailable. Falling back to '%s'.", TEXT_MODEL_ID, selected)
    TEXT_MODEL_ID = selected

    # Live モデル
    live_ok = True
    try:
        _get(MODEL_ID)
    except Exception as e:
        log.warning("Gemini Live model '%s' is not available: %s", MODEL_ID, e)
        live_ok = False
    return live_ok


async def _ensure_gemini_ready():
    global LIVE_MODEL_AVAILABLE
    LIVE_MODEL_AVAILABLE = await asyncio.to_thread(_probe_gemini_models)

# ====== データベース（PostgreSQL） ======
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://webbot:webbot@db:5432/webbot")
DB_POOL: ConnectionPool | None = None
LOG_TAIL_ROW_LIMIT = int(os.getenv("LOG_TAIL_ROW_LIMIT", "400"))


def _get_pool() -> ConnectionPool:
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = ConnectionPool(
            DATABASE_URL,
            min_size=int(os.getenv("DB_MIN_CONN", "1")),
            max_size=int(os.getenv("DB_MAX_CONN", "5")),
            kwargs={"autocommit": True},
        )
    return DB_POOL


def _init_db():
    pool = _get_pool()
    with pool.connection() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS users(
                id TEXT PRIMARY KEY,
                pw_hash TEXT NOT NULL,
                display_name TEXT NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            )
            """
        )
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS chat_logs(
                id BIGSERIAL PRIMARY KEY,
                user_id TEXT NOT NULL,
                ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                role TEXT,
                modality TEXT,
                text TEXT,
                data JSONB NOT NULL
            )
            """
        )
        con.execute(
            """
            CREATE INDEX IF NOT EXISTS chat_logs_user_ts_idx
                ON chat_logs(user_id, ts DESC, id DESC)
            """
        )


async def _ensure_db_ready(retries: int = 5, delay: float = 1.5):
    for attempt in range(retries):
        try:
            await asyncio.to_thread(_init_db)
            return
        except Exception as e:
            log.warning("DB init failed (attempt %s/%s): %s", attempt + 1, retries, e)
            if attempt + 1 == retries:
                raise
            await asyncio.sleep(delay)


async def _get_display_name(uid: str) -> str:
    def _fetch():
        pool = _get_pool()
        with pool.connection() as con:
            cur = con.execute("SELECT display_name FROM users WHERE id=%s", (uid,))
            row = cur.fetchone()
            return row[0] if row else "ユーザー"

    return await asyncio.to_thread(_fetch)


# ====== 会話ログ（PostgreSQL に保存） ======
async def append_log(uid: str, rec: dict) -> None:
    rec = dict(rec)
    rec.setdefault("ts", _jst_now_iso())
    ts_val = rec.get("ts")
    try:
        ts = datetime.fromisoformat(ts_val)
    except Exception:
        ts = datetime.now(timezone.utc)

    def _insert():
        pool = _get_pool()
        with pool.connection() as con:
            con.execute(
                """
                INSERT INTO chat_logs(user_id, ts, role, modality, text, data)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    uid,
                    ts,
                    rec.get("role"),
                    rec.get("modality"),
                    rec.get("text"),
                    Json(rec),
                ),
            )

    await asyncio.to_thread(_insert)


async def read_tail(uid: str, n_bytes: int = 6000) -> str:
    def _fetch() -> str:
        pool = _get_pool()
        with pool.connection() as con:
            cur = con.execute(
                """
                SELECT data
                FROM chat_logs
                WHERE user_id=%s
                ORDER BY ts DESC, id DESC
                LIMIT %s
                """,
                (uid, LOG_TAIL_ROW_LIMIT),
            )
            rows = cur.fetchall()

        lines: list[str] = []
        for (data,) in reversed(rows):
            try:
                obj = data if isinstance(data, dict) else json.loads(data)
                lines.append(json.dumps(obj, ensure_ascii=False))
            except Exception:
                continue

        joined = "\n".join(lines)
        encoded = joined.encode("utf-8")
        if len(encoded) <= n_bytes:
            return joined
        return encoded[-n_bytes:].decode("utf-8", errors="ignore")

    return await asyncio.to_thread(_fetch)

# ====== WebSocket ブリッジ（Discord代替） ======
class WSBridge:
    def __init__(self, ws: WebSocket, user_id: str):
        self.ws = ws
        self.user_id = user_id
        self.audio_q: asyncio.Queue[bytes] = asyncio.Queue(maxsize=256)
        self._tts_ack_event = asyncio.Event()
        self._pending_tts_acks = 0

    def get_queue(self): return self.audio_q

    async def send_text_partial(self, text: str):
        await self.ws.send_json({"type":"partialText", "text": text})

    async def send_tts_wav(self, wav_bytes: bytes):
        # 文ごとにWAVバイナリを送信（ブラウザ側で decodeAudioData して再生）
        await self.ws.send_bytes(wav_bytes)

    def notify_tts_ack(self):
        self._pending_tts_acks += 1
        self._tts_ack_event.set()

    async def wait_tts_ack(self, timeout: float | None = None) -> bool:
        async def _wait():
            while self._pending_tts_acks == 0:
                await self._tts_ack_event.wait()

        try:
            if self._pending_tts_acks == 0:
                if timeout is None:
                    await _wait()
                else:
                    await asyncio.wait_for(_wait(), timeout)
            if self._pending_tts_acks == 0:
                return False
            self._pending_tts_acks -= 1
            if self._pending_tts_acks == 0:
                self._tts_ack_event.clear()
            return True
        except asyncio.TimeoutError:
            self._pending_tts_acks = 0
            self._tts_ack_event.clear()
            return False

# ====== Gemini Live worker（bot.py をWeb向けに調整） ======
class GeminiLiveWorker:
    def __init__(self, bridge: WSBridge):
        self.bridge = bridge
        self._audio_q = bridge.get_queue()
        self._stop = asyncio.Event()
        self._task: Optional[asyncio.Task] = None

        # 文セグメンター
        self._out_buf = ""
        self._sent_lock = asyncio.Lock()
        self._last_text_ts = 0.0
        self._idle_task: Optional[asyncio.Task] = None
        self._tts_q: asyncio.Queue[str] = asyncio.Queue(maxsize=256)
        self._tts_task: Optional[asyncio.Task] = None
        self._end_chars = set(SENT_END_CHARS)
        self._closers = set(TRAILING_CLOSERS)

        # 履歴注入
        self._history_user_id: Optional[str] = None
        self._history_display_name: Optional[str] = None
        self._history_excerpt: Optional[str] = None

    def running(self): return self._task is not None and not self._task.done()
    def get_queue(self): return self._audio_q

    def set_context_from_history(self, user_id: str, display_name: str, excerpt: str | None):
        self._history_user_id = user_id
        self._history_display_name = display_name
        self._history_excerpt = (excerpt or "").strip()

    async def start(self):
        if self.running(): return
        client = genai.Client(api_key=GEMINI_API_KEY)
        self._stop.clear()
        self._task = asyncio.create_task(self._run(client))
        self._tts_task = asyncio.create_task(self._run_tts())

    async def stop(self):
        self._stop.set()
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
            try: await self._idle_task
            except Exception: pass
        async with self._sent_lock:
            self._out_buf = ""
        if self._tts_task and not self._tts_task.done():
            self._tts_task.cancel()
            try: await self._tts_task
            except Exception: pass
        try:
            while True:
                self._tts_q.get_nowait(); self._tts_q.task_done()
        except Exception: pass
        t = self._task; self._task = None
        if t:
            try: await asyncio.wait_for(t, timeout=2.0)
            except asyncio.TimeoutError: pass

    # --- 文セグメント ---
    async def _on_text(self, delta: str):
        if not delta: return
        async with self._sent_lock:
            self._out_buf += delta
            self._last_text_ts = time.time()
            await self._emit_ready_sentences_locked()
            self._schedule_idle_flush_locked()

    async def _emit_ready_sentences_locked(self):
        buf = self._out_buf
        if not buf: return
        i = 0; last = 0; n = len(buf); emitted = False
        while i < n:
            ch = buf[i]
            if ch in self._end_chars:
                j = i + 1
                while j < n and buf[j] in self._end_chars: j += 1
                while j < n and buf[j] in self._closers: j += 1
                sent = buf[last:j].strip()
                if sent:
                    await self._enqueue_sentence(sent); emitted = True
                last = j; i = j; continue
            i += 1
        if emitted: self._out_buf = buf[last:]

    def _schedule_idle_flush_locked(self):
        if self._idle_task and not self._idle_task.done():
            self._idle_task.cancel()
        self._idle_task = asyncio.create_task(self._idle_flush())

    async def _idle_flush(self):
        try:
            await asyncio.sleep(SENTENCE_IDLE_MS / 1000.0)
            async with self._sent_lock:
                await self._emit_ready_sentences_locked()
                if self._out_buf.strip() and (time.time() - self._last_text_ts) >= (TAIL_FORCE_FLUSH_MS / 1000.0):
                    last_tail = self._out_buf.strip(); self._out_buf = ""
                    await self._enqueue_sentence(last_tail)
        except asyncio.CancelledError:
            pass

    async def _enqueue_sentence(self, sent: str):
        clean = (sent or "").strip()
        if _looks_like_meta_sentence(clean):
            log.info("Gemini ▶ (filtered meta) %s", clean)
            return
        log.info("Gemini ▶ %s", clean)
        # ログ（assistant/voice）
        try:
            await append_log(self._history_user_id or self.bridge.user_id, {
                "user_id": self._history_user_id or self.bridge.user_id,
                "role": "assistant", "modality": "voice", "text": clean,
            })
        except Exception: pass
        await self._tts_q.put(clean)
        await self.bridge.send_text_partial(clean)  # 画面にも流し込み

    async def _run_tts(self):
        while not self._stop.is_set():
            try:
                text = await self._tts_q.get()
            except asyncio.CancelledError:
                break
            try:
                wav = await voicevox_tts_async(text)
                await self.bridge.send_tts_wav(wav)
                await self._wait_for_playback_ack(wav)
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error("TTS error: %s", e)
            finally:
                self._tts_q.task_done()

    async def _wait_for_playback_ack(self, wav: bytes):
        duration = _estimate_wav_duration(wav)
        timeout = max(1.5, duration + 0.8)
        try:
            acked = await self.bridge.wait_tts_ack(timeout=timeout)
            if not acked:
                log.warning("TTS ack timeout (waited %.2fs); continuing", timeout)
        except asyncio.CancelledError:
            raise

    async def _run(self, client: "genai.Client"):
        last_audio_sent_ts = 0.0
        last_eos_ts = 0.0

        async def sender(sess):
            nonlocal last_audio_sent_ts, last_eos_ts
            chunk_buf: Deque[bytes] = deque()
            BYTES_PER_SAMPLE = 2
            target_bytes_48k = int(DISCORD_PCM_RATE * (CHUNK_TARGET_MS / 1000.0)) * BYTES_PER_SAMPLE
            eos_sent = False
            # Guard against sending EOS before any audio was ever streamed.
            have_sent_audio = False
            async def flush_now():
                nonlocal last_audio_sent_ts, last_eos_ts, eos_sent, have_sent_audio
                if not chunk_buf: return
                payload48 = b"".join(chunk_buf); chunk_buf.clear()
                if not payload48: return
                out_bytes = _downsample_48k_to_16k_mono(payload48)
                await sess.send_realtime_input(audio=types.Blob(data=out_bytes, mime_type=LIVE_PCM_MIME))
                last_audio_sent_ts = time.time()
                have_sent_audio = True
                eos_sent = False

            while not self._stop.is_set():
                try:
                    pcm = await asyncio.wait_for(self._audio_q.get(), timeout=0.8)
                    chunk_buf.append(pcm)
                    if sum(len(c) for c in chunk_buf) >= target_bytes_48k:
                        await flush_now()
                except asyncio.TimeoutError:
                    await flush_now()
                    now = time.time()
                    if have_sent_audio and (now - last_audio_sent_ts) > EOS_IDLE_SEC and (now - last_eos_ts) > 1.0 and not eos_sent:
                        try:
                            await sess.send_realtime_input(audio_stream_end=True)
                            last_eos_ts = now; eos_sent = True
                            log.info("⬛ EOS (idle)")
                        except Exception:
                            pass
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    log.warning("sender error: %s", e)
                    break

        async def receiver(sess):
            while not self._stop.is_set():
                try:
                    async for msg in sess.receive():
                        text = getattr(msg, "text", None)
                        sc = getattr(msg, "server_content", None)

                        # 1) Native-audio path: prefer output transcription
                        if not text and sc and getattr(sc, "output_transcription", None):
                            try:
                                text = sc.output_transcription.text or ""
                            except Exception:
                                pass

                        # 2) Legacy TEXT modality path
                        if not text and sc and getattr(sc, "model_turn", None):
                            try:
                                parts = sc.model_turn.parts or []
                                for p in parts:
                                    if getattr(p, "text", None):
                                        text = p.text
                                        break
                            except Exception:
                                pass

                        # Log user's speech transcription (input side)
                        if sc and getattr(sc, "input_transcription", None):
                            try:
                                tr = sc.input_transcription.text or ""
                                if tr:
                                    await append_log(self.bridge.user_id, {
                                        "user_id": self.bridge.user_id,
                                        "role": "user", "modality": "voice", "text": tr,
                                    })
                            except Exception:
                                pass

                        if text:
                            await self._on_text(text)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    if self._stop.is_set():
                        break
                    log.warning("receiver error: %s", e)
                    await asyncio.sleep(0.3)
                else:
                    if getattr(sess, "closed", False):
                        break

        sys_inst = _compose_system_instruction(
            SYSTEM_PERSONALITY,
            self._history_excerpt,
            self._history_display_name,
        )
        # NOTE: The latest native-audio preview models only accept AUDIO response modality.
        # Requesting TEXT triggers `Cannot extract voices from a non-audio request (1007)`
        # from the Live endpoint. For non-native models we keep TEXT to reuse VOICEVOX TTS.
        native_audio_model = "native-audio" in (MODEL_ID or "")
        response_modalities = ["AUDIO"] if native_audio_model else ["TEXT"]

        config = {
            "response_modalities": response_modalities,
            "input_audio_transcription": {},
            "realtime_input_config": {
                "automatic_activity_detection": {
                    "disabled": False,
                    "silence_duration_ms": VAD_SILENCE_MS,
                    "prefix_padding_ms": 50,
                }
            },
            "system_instruction": sys_inst,


            # ★ thinking を完全にオフ + 思考サマリも出さない
            "thinking_config": {
                "thinking_budget": 0,      # thinkingBudget: 0 → 思考を無効化
                "include_thoughts": False, # 思考サマリをレスポンスに含めない
            },
        }

        # When using audio modality, also request text via output transcription so the
        # downstream VOICEVOX pipeline can continue working without consuming model audio.
        if native_audio_model:
            config["output_audio_transcription"] = {}

        log.info("Connecting Gemini Live...")
        try:
            async with client.aio.live.connect(model=MODEL_ID, config=config) as sess:
                await asyncio.gather(sender(sess), receiver(sess))
        except Exception as e:
            log.exception("Gemini Live failed: %s", e)
            try:
                await self.bridge.ws.send_json({
                    "type": "error",
                    "message": f"Gemini Live 接続に失敗しました: {e}",
                })
            except Exception:
                pass

# ====== FastAPI アプリ ======
app = FastAPI()

# セッション（署名付きCookie）
app.add_middleware(
    SessionMiddleware,
    secret_key=SESSION_SECRET,
    same_site="lax",
    https_only=False,  # 本番は True（HTTPS 前提）
)


@app.on_event("startup")
async def on_startup():
    await _ensure_db_ready()
    await _ensure_gemini_ready()


@app.on_event("shutdown")
async def on_shutdown():
    pool = _get_pool()
    pool.close()
    pool.wait_close()


@app.get("/health")
async def health():
    return {"ok": True, "log_storage": "postgres"}

# ====== 認証API（登録／ログイン／ログアウト／自分） ======
@app.post("/auth/register")
async def register(req: Request, payload: dict):
    uid = (payload.get("id") or "").strip()
    pw  = (payload.get("password") or "")
    name= (payload.get("display_name") or uid)[:40] or "ユーザー"
    if not uid or not pw:
        raise HTTPException(400, "id と password は必須です。")

    def _create():
        pool = _get_pool()
        with pool.connection() as con:
            cur = con.execute("SELECT 1 FROM users WHERE id=%s", (uid,))
            if cur.fetchone():
                return False
            # bcrypt_sha256 でハッシュ化（72バイト制限の影響を避ける）
            con.execute(
                "INSERT INTO users(id, pw_hash, display_name, created_at) VALUES(%s,%s,%s,NOW())",
                (uid, bcrypt.hash(pw), name)
            )
            return True
    ok = await asyncio.to_thread(_create)
    if not ok:
        raise HTTPException(409, "そのIDは登録済みです。")

    req.session["user_id"] = uid
    req.session["display_name"] = name
    return {"ok": True, "id": uid, "display_name": name}

@app.post("/auth/login")
async def login(req: Request, payload: dict):
    uid = (payload.get("id") or "").strip()
    pw  = (payload.get("password") or "")

    def _check():
        pool = _get_pool()
        with pool.connection() as con:
            cur = con.execute("SELECT pw_hash, display_name FROM users WHERE id=%s", (uid,))
            row = cur.fetchone()
            if not row: return None
            pw_hash, name = row
            # bcrypt_sha256 で検証
            return (bcrypt.verify(pw, pw_hash), name)

    res = await asyncio.to_thread(_check)
    if not res:
        raise HTTPException(401, "IDまたはパスワードが違います。")
    ok, name = res
    if not ok:
        raise HTTPException(401, "IDまたはパスワードが違います。")

    req.session["user_id"] = uid
    req.session["display_name"] = name
    return {"ok": True, "id": uid, "display_name": name}

@app.post("/auth/logout")
async def logout(req: Request):
    req.session.clear()
    return {"ok": True}

@app.get("/me")
async def me(req: Request):
    uid = req.session.get("user_id")
    if not uid:
        raise HTTPException(401, "not authenticated")
    return {"id": uid, "display_name": req.session.get("display_name","ユーザー")}

# WebSocket 接続用の短命トークンを発行（ログイン済ユーザーのみ）
@app.post("/auth/ws-token")
async def ws_token(req: Request):
    uid = req.session.get("user_id"); name = req.session.get("display_name")
    if not uid:
        raise HTTPException(401)
    token = _signer.sign(uid.encode()).decode()
    return {"token": token, "id": uid, "display_name": name}

# ====== テキストチャット（ログイン必須） ======
@app.post("/chat")
async def chat(req: Request, payload: dict):
    uid = req.session.get("user_id")
    if not uid:
        raise HTTPException(401)
    user_text = (payload.get("text") or "").strip() or "こんにちは。"
    excerpt = (await read_tail(uid, 6000)).strip()
    display_name = req.session.get("display_name", "ユーザー")

    sys_inst = _compose_system_instruction(SYSTEM_PERSONALITY, excerpt, display_name)
    client = genai.Client(api_key=GEMINI_API_KEY)

    async def _call_with_fallback():
        global TEXT_MODEL_ID
        last_err = None
        # _TEXT_MODEL_CANDIDATES は環境変数の指定を先頭に、利用可能性順に並んでいる
        for model_id in _TEXT_MODEL_CANDIDATES:
            for attempt in range(TEXT_RETRY_MAX):
                try:
                    resp = await asyncio.to_thread(
                        client.models.generate_content,
                        model=model_id,
                        contents=user_text,
                        config=types.GenerateContentConfig(system_instruction=sys_inst),
                    )
                    # 成功したら以降もこのモデルを使う
                    if model_id != TEXT_MODEL_ID:
                        log.warning("chat: fallback text model in use -> %s (was %s)", model_id, TEXT_MODEL_ID)
                        TEXT_MODEL_ID = model_id
                    return resp
                except Exception as e:
                    last_err = e
                    retryable = _is_retryable_genai_error(e)
                    remaining = TEXT_RETRY_MAX - attempt - 1
                    if retryable and remaining > 0:
                        backoff = TEXT_RETRY_BASE_DELAY * (2 ** attempt) * (0.7 + 0.6 * random.random())
                        log.warning("chat: transient error on %s (try %d/%d): %s -> retry in %.2fs",
                                    model_id, attempt + 1, TEXT_RETRY_MAX, e, backoff)
                        await asyncio.sleep(backoff)
                        continue
                    log.warning("chat: model %s failed (try %d/%d): %s", model_id, attempt + 1, TEXT_RETRY_MAX, e)
                    break  # 次のモデル候補へ
        if last_err:
            raise last_err
        raise RuntimeError("No text model candidates configured.")

    try:
        await append_log(uid, {"user_id": uid, "role":"user", "modality":"text", "text": user_text})
        resp = await _call_with_fallback()
        text = (getattr(resp, "text", "") or str(resp)).strip()[:1000]
        await append_log(uid, {"user_id": uid, "role":"assistant", "modality":"text", "text": text})
        return {"reply": text}
    except genai_errors.ClientError as e:
        log.exception("Gemini API error: %s", e)
        if _is_retryable_genai_error(e):
            friendly = "すみません、モデルが混み合っています。少し待ってからもう一度お試しください。"
            return {"reply": friendly, "error": "temporarily_unavailable"}
        msg = "Gemini APIキーまたはモデル設定を確認してください。"
        return JSONResponse({"reply": "すみません、応答に失敗しました。", "error": msg, "detail": str(e)}, status_code=500)
    except Exception as e:
        log.exception("text generation failed: %s", e)
        if _is_retryable_genai_error(e):
            friendly = "すみません、ただいま混雑しています。もう一度お試しください。"
            return {"reply": friendly, "error": "temporarily_unavailable"}
        return JSONResponse({"reply": "すみません、応答に失敗しました。", "error": str(e)}, status_code=500)

# ====== 音声用 WebSocket（ログイン必須：短命トークンで認証） ======
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    token = ws.query_params.get("t")
    if not token:
        await ws.close(code=1008)
        return
    try:
        uid = _signer.unsign(token, max_age=WS_TOKEN_TTL).decode()
    except (BadSignature, SignatureExpired):
        await ws.close(code=1008)
        return

    if not LIVE_MODEL_AVAILABLE:
        await ws.accept()
        await ws.send_json({
            "type": "error",
            "message": "Gemini Live モデルが現在利用できません。テキストチャットのみご利用ください。管理者に GEMINI_MODEL と API キーを確認してください。",
        })
        await ws.close(code=1013)
        return

    await ws.accept()

    user_id = uid
    display_name = await _get_display_name(uid)

    bridge = WSBridge(ws, user_id)
    worker = GeminiLiveWorker(bridge)
    # 直近履歴を Live の system_instruction に差し込む（bot.pyと同じ）
    try:
        history_text = await read_tail(user_id, 4000)
        worker.set_context_from_history(user_id, display_name, history_text)
    except Exception:
        pass
    await worker.start()

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                # 48kHz s16 PCM チャンク（ブラウザから）
                await bridge.audio_q.put(msg["bytes"])
            elif "text" in msg and msg["text"] is not None:
                payload = None
                try:
                    payload = json.loads(msg["text"])
                except Exception:
                    payload = None
                if isinstance(payload, dict) and payload.get("type") == "ttsAck":
                    bridge.notify_tts_ack()
    except WebSocketDisconnect:
        pass
    finally:
        await worker.stop()

# ======（重要）同一オリジンにまとめる：フロントエンド配信 ======
# backend/ から見た ../frontend を配信対象にする
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if not os.path.isdir(FRONTEND_DIR):
    log.warning("Frontend dir not found: %s", FRONTEND_DIR)

# 既存の /chat や /ws を定義した “後” に、最後にマウントするのが安全
# html=True により / へアクセスで index.html を返す
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
