# backend/main.py
import os, asyncio, json, contextlib, re, random, hashlib
from datetime import datetime, timezone, timedelta
from typing import Any

import numpy as np
import fitz
from fastapi import FastAPI, Request, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
import psycopg
from psycopg import errors as pg_errorsF
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool
# --- 重要: 72バイト制限を回避するため bcrypt_sha256 を使用 ---
from passlib.hash import bcrypt_sha256 as bcrypt

# ====== ここから bot.py 由来の共通設定を “ほぼそのまま” ======F
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
# MODEL_ID is used only for the optional Gemini Live fallback in text chat.
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
    # v1beta では `*-latest` のモデルIDが存在しないことがあるため、
    # 安定して存在する ID を優先する。
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]:
    if _cand not in _TEXT_MODEL_CANDIDATES:
        _TEXT_MODEL_CANDIDATES.append(_cand)
SYSTEM_PERSONALITY = os.getenv(
    "SYSTEM_PERSONALITY",
    "どんな質問にもわかりやすく、まとまった考えで1000文字以内で答えてください。"
)

SESSION_SECRET = os.getenv("SESSION_SECRET", "dev-secret-change-me")
LIVE_MODEL_AVAILABLE = True  # Gemini Live を利用できるかどうか（起動時チェックで更新）
TEXT_RETRY_MAX = int(os.getenv("TEXT_RETRY_MAX", "3"))  # テキスト生成時のリトライ回数（モデルごと）
TEXT_RETRY_BASE_DELAY = float(os.getenv("TEXT_RETRY_BASE_DELAY", "0.6"))  # リトライ時の初期待機（秒）

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s")
log = logging.getLogger("web-bot")

def _env_flag(name: str, default: str = "0") -> bool:
    v = os.getenv(name, default)
    return str(v).strip().lower() in ("1", "true", "yes", "on")

def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default

def _jst_now_iso() -> str:
    return datetime.now(timezone(timedelta(hours=9))).isoformat()

# generateContent 失敗時に Live で代替する（モデル/プラン事情で Live だけ動く環境向け）
CHAT_LIVE_FALLBACK = _env_flag("CHAT_LIVE_FALLBACK", "1")

# ====== RAG（data/ 配下のテキストをシンプルに注入） ======
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
RAG_MAX_TOTAL_CHARS = int(os.getenv("RAG_MAX_TOTAL_CHARS", "7000"))   # システムプロンプトに載せる総文字数上限
RAG_PER_FILE_CHARS = int(os.getenv("RAG_PER_FILE_CHARS", "1800"))     # 1ファイルあたりの抜粋上限
RAG_FILE_SIZE_LIMIT = int(os.getenv("RAG_FILE_SIZE_LIMIT", str(2 * 1024 * 1024)))  # 既定: 2MB まで読み込み
RAG_ALLOWED_EXTS = {".txt", ".md", ".markdown", ".json", ".csv", ".tsv", ".yaml", ".yml", ".log"}

# ====== RAG（vectors.pt をDBへ取り込み、ベクトル検索で参照） ======
RAG_VECTOR_ENABLED = _env_flag("RAG_VECTOR_ENABLED", "1")
RAG_VECTOR_MODE = (os.getenv("RAG_VECTOR_MODE") or "sql").strip().lower()
if RAG_VECTOR_MODE not in ("sql", "memory"):
    RAG_VECTOR_MODE = "sql"
RAG_VECTOR_INDEX = (os.getenv("RAG_VECTOR_INDEX") or "none").strip().lower()
if RAG_VECTOR_INDEX not in ("hnsw", "ivfflat", "none"):
    RAG_VECTOR_INDEX = "none"
RAG_VECTOR_INDEX_LISTS = _env_int("RAG_VECTOR_INDEX_LISTS", 100)
RAG_VECTOR_INDEX_M = _env_int("RAG_VECTOR_INDEX_M", 16)
RAG_VECTOR_INDEX_EF = _env_int("RAG_VECTOR_INDEX_EF", 64)
RAG_VECTOR_DIM = _env_int("RAG_VECTOR_DIM", 0)
RAG_VECTOR_SQL_AVAILABLE = False
RAG_VECTOR_COLUMN_TYPE = "array"

def _resolve_vectors_path(path: str) -> str:
    if path and os.path.isfile(path):
        return path
    candidates: list[str] = []
    if path:
        base = os.path.dirname(path) or "."
        name = os.path.basename(path)
        if name == "vectors.pt":
            candidates.append(os.path.join(base, "vector.pt"))
        elif name == "vector.pt":
            candidates.append(os.path.join(base, "vectors.pt"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.append(os.path.join(project_root, "vectors.pt"))
    candidates.append(os.path.join(project_root, "vector.pt"))
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return path

RAG_VECTORS_PATH = _resolve_vectors_path(
    os.getenv(
        "RAG_VECTORS_PATH",
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vectors.pt")),
    )
)
RAG_TEXTS_PATH = os.getenv("RAG_TEXTS_PATH", "")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "4"))
RAG_MIN_SIM = float(os.getenv("RAG_MIN_SIM", "0.2"))
RAG_VECTOR_MAX_TOTAL_CHARS = int(os.getenv("RAG_VECTOR_MAX_TOTAL_CHARS", "3200"))
RAG_VECTOR_PER_CHUNK_CHARS = int(os.getenv("RAG_VECTOR_PER_CHUNK_CHARS", "1200"))
RAG_EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "text-embedding-004")
RAG_EMBED_TASK_TYPE = os.getenv("RAG_EMBED_TASK_TYPE", "retrieval_query")
RAG_INGEST_ON_START = _env_flag("RAG_INGEST_ON_START", "1")
RAG_FORCE_RELOAD = _env_flag("RAG_FORCE_RELOAD", "1")
RAG_CACHE_IN_MEMORY = _env_flag("RAG_CACHE_IN_MEMORY", "1")
RAG_FALLBACK_ENABLED = _env_flag("RAG_FALLBACK_ENABLED", "1")
RAG_FALLBACK_TOP_K = int(os.getenv("RAG_FALLBACK_TOP_K", str(RAG_TOP_K)))
RAG_FALLBACK_MAX_TOTAL_CHARS = int(os.getenv("RAG_FALLBACK_MAX_TOTAL_CHARS", str(RAG_VECTOR_MAX_TOTAL_CHARS)))
RAG_FALLBACK_PER_CHUNK_CHARS = int(os.getenv("RAG_FALLBACK_PER_CHUNK_CHARS", str(RAG_VECTOR_PER_CHUNK_CHARS)))
RAG_DOC_EMBED_TASK_TYPE = os.getenv("RAG_DOC_EMBED_TASK_TYPE", "retrieval_document")

PDF_MAX_BYTES = _env_int("PDF_MAX_BYTES", 50 * 1024 * 1024)
PDF_CHUNK_SIZE_CHARS = _env_int("PDF_CHUNK_SIZE_CHARS", 1200)
PDF_CHUNK_OVERLAP_CHARS = _env_int("PDF_CHUNK_OVERLAP_CHARS", 150)
PDF_EMBED_BATCH_SIZE = _env_int("PDF_EMBED_BATCH_SIZE", 32)


class _RagCache:
    def __init__(
        self,
        embeddings: np.ndarray,
        texts: list[str],
        metas: list[dict],
        user_ids: list[str | None] | None = None,
        doc_ids: list[int | None] | None = None,
    ):
        self.embeddings = embeddings
        self.texts = texts
        self.metas = metas
        total = len(texts)
        if user_ids is None:
            user_ids = [None] * total
        if doc_ids is None:
            doc_ids = [None] * total
        if len(user_ids) < total:
            user_ids = list(user_ids) + [None] * (total - len(user_ids))
        if len(doc_ids) < total:
            doc_ids = list(doc_ids) + [None] * (total - len(doc_ids))
        self.user_ids = user_ids
        self.doc_ids = doc_ids
        self.dim = int(embeddings.shape[1]) if embeddings is not None and embeddings.size else 0


RAG_CACHE: _RagCache | None = None
RAG_CACHE_LOCK = asyncio.Lock()
RAG_FALLBACK_TEXTS: list[str] | None = None
RAG_FALLBACK_TEXTS_LOCK = asyncio.Lock()


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


def _truncate_text(text: str, limit: int) -> str:
    if limit <= 0:
        return ""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit]


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _looks_like_pdf(data: bytes) -> bool:
    return bool(data) and data[:5] == b"%PDF-"


def _clean_pdf_text(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"-\s*\n", "", text)
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pdf_text(pdf_bytes: bytes) -> list[dict]:
    if not pdf_bytes:
        return []
    pages: list[dict] = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        for idx in range(doc.page_count):
            page = doc.load_page(idx)
            raw = page.get_text("text") or ""
            cleaned = _clean_pdf_text(raw)
            pages.append({"page": idx + 1, "text": cleaned})
    finally:
        doc.close()
    return pages


def chunk_text(pages: list[dict], chunk_size_chars: int = 1200, overlap_chars: int = 150) -> list[dict]:
    if not pages or chunk_size_chars <= 0:
        return []
    step = max(1, chunk_size_chars - max(0, overlap_chars))
    chunks: list[dict] = []
    for page in pages:
        text = (page.get("text") or "").strip()
        if not text:
            continue
        page_num = page.get("page")
        for start in range(0, len(text), step):
            content = text[start : start + chunk_size_chars].strip()
            if not content:
                continue
            meta = {"page_start": page_num, "page_end": page_num, "offset": start}
            chunks.append({"content": content, "meta": meta})
    return chunks


def _extract_text_from_mapping(obj: dict) -> str:
    for key in ("text", "content", "page_content", "chunk", "document", "doc", "body"):
        val = obj.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _extract_meta_from_mapping(obj: dict) -> dict:
    meta = obj.get("metadata") or obj.get("meta")
    if isinstance(meta, dict):
        return meta
    # Fallback: keep non-text fields as metadata
    meta = {}
    for key, val in obj.items():
        if key in ("text", "content", "page_content", "chunk", "document", "doc", "body"):
            continue
        if key in ("embedding", "vector", "vectors", "values"):
            continue
        meta[key] = val
    return meta


def _to_float_vector(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    try:
        if hasattr(val, "detach"):
            return val.detach().cpu().float().numpy()
        if isinstance(val, np.ndarray):
            return val.astype(np.float32, copy=False)
        if isinstance(val, (list, tuple)):
            arr = np.asarray(val, dtype=np.float32)
            if arr.ndim == 1:
                return arr
            if arr.ndim == 2 and arr.shape[0] == 1:
                return arr[0]
    except Exception:
        return None
    return None


def _extract_embedding_from_mapping(obj: dict) -> np.ndarray | None:
    for key in ("embedding", "vector", "vectors", "values"):
        if key in obj:
            return _to_float_vector(obj.get(key))
    return None


def _extract_text_from_obj(obj: Any) -> str:
    if isinstance(obj, str):
        return obj.strip()
    if isinstance(obj, dict):
        return _extract_text_from_mapping(obj)
    for attr in ("text", "content", "page_content", "chunk", "document", "doc", "body"):
        try:
            val = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _extract_meta_from_obj(obj: Any) -> dict:
    if isinstance(obj, dict):
        return _extract_meta_from_mapping(obj)
    for attr in ("metadata", "meta"):
        try:
            val = getattr(obj, attr)
        except Exception:
            continue
        if isinstance(val, dict):
            return val
    return {}


def _coerce_texts_and_metas(seq: Any) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    metas: list[dict] = []
    if not isinstance(seq, (list, tuple)):
        return texts, metas
    for item in seq:
        text = _extract_text_from_obj(item)
        if not text:
            continue
        texts.append(text)
        metas.append(_extract_meta_from_obj(item))
    return texts, metas


def _to_float_matrix(val: Any) -> np.ndarray | None:
    if val is None:
        return None
    try:
        if hasattr(val, "detach"):
            arr = val.detach().cpu().float().numpy()
        elif isinstance(val, np.ndarray):
            arr = val.astype(np.float32, copy=False)
        elif isinstance(val, (list, tuple)):
            arr = np.asarray(val, dtype=np.float32)
        else:
            return None
        if getattr(arr, "dtype", None) == object:
            return None
        if arr.ndim == 1:
            return arr.reshape(1, -1)
        if arr.ndim == 2:
            return arr
    except Exception:
        return None
    return None


def _expand_embeddings(val: Any) -> list[np.ndarray]:
    mat = _to_float_matrix(val)
    if mat is not None:
        return [mat[i].astype(np.float32, copy=False) for i in range(mat.shape[0])]
    if isinstance(val, (list, tuple)):
        return [e for e in (_to_float_vector(v) for v in val) if e is not None]
    return []


def _looks_like_text_item(item: Any) -> bool:
    if isinstance(item, str):
        return True
    if isinstance(item, dict):
        return bool(_extract_text_from_mapping(item))
    for attr in ("text", "content", "page_content", "chunk", "document", "doc", "body"):
        try:
            val = getattr(item, attr)
        except Exception:
            continue
        if isinstance(val, str) and val.strip():
            return True
    return False


def _looks_like_text_seq(seq: Any) -> bool:
    if not isinstance(seq, (list, tuple)):
        return False
    for item in list(seq)[:5]:
        if _looks_like_text_item(item):
            return True
    return False


def _load_texts_from_path(path: str) -> tuple[list[str], list[dict]]:
    if not path:
        return [], []
    if not os.path.isfile(path):
        log.warning("RAG: texts file not found: %s", path)
        return [], []

    texts: list[str] = []
    metas: list[dict] = []
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".jsonl":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        texts.append(line)
                        metas.append({})
                        continue
                    if isinstance(obj, str):
                        texts.append(obj)
                        metas.append({})
                        continue
                    if isinstance(obj, dict):
                        text = _extract_text_from_mapping(obj)
                        if not text:
                            continue
                        texts.append(text)
                        metas.append(_extract_meta_from_mapping(obj))
                        continue
        elif ext == ".json":
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                obj = json.load(f)
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        texts.append(item)
                        metas.append({})
                    elif isinstance(item, dict):
                        text = _extract_text_from_mapping(item)
                        if not text:
                            continue
                        texts.append(text)
                        metas.append(_extract_meta_from_mapping(item))
            elif isinstance(obj, dict):
                # {id: text} style
                for key, val in obj.items():
                    if isinstance(val, str):
                        texts.append(val)
                        metas.append({"id": key})
        else:
            # Plain text: one chunk per non-empty line
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    texts.append(line)
                    metas.append({})
    except Exception as e:
        log.warning("RAG: failed to read texts from %s: %s", path, e)
        return [], []
    return texts, metas


def _guess_rag_texts_path() -> str:
    if RAG_TEXTS_PATH:
        return RAG_TEXTS_PATH
    base = os.path.splitext(RAG_VECTORS_PATH)[0]
    for ext in (".jsonl", ".json", ".txt"):
        cand = base + ext
        if os.path.isfile(cand):
            return cand
    search_dirs = [os.path.dirname(RAG_VECTORS_PATH)]
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if project_root not in search_dirs:
        search_dirs.append(project_root)
    for name in (
        "rag_texts.jsonl",
        "rag_texts.json",
        "rag_texts.txt",
        "texts.jsonl",
        "texts.json",
        "texts.txt",
        "extracted_text",
        "extracted_text.txt",
    ):
        for base_dir in search_dirs:
            cand = os.path.join(base_dir, name)
            if os.path.isfile(cand):
                return cand
    return ""


def _load_vectors_pt(path: str) -> tuple[list[np.ndarray], list[str], list[dict]]:
    if not os.path.isfile(path):
        log.warning("RAG: vectors file not found: %s", path)
        return [], [], []
    try:
        import torch  # heavy import; keep local
    except Exception as e:
        log.warning("RAG: torch not available, cannot load %s: %s", path, e)
        return [], [], []

    try:
        obj = torch.load(path, map_location="cpu")
    except Exception as e:
        log.warning("RAG: failed to load vectors from %s: %s", path, e)
        return [], [], []

    embeddings: list[np.ndarray] = []
    texts: list[str] = []
    metas: list[dict] = []

    def _append(emb: np.ndarray | None, text: str | None = None, meta: dict | None = None):
        if emb is None:
            return
        embeddings.append(emb.astype(np.float32, copy=False))
        if text is not None:
            texts.append(text)
        if meta is not None:
            metas.append(meta)

    mat = _to_float_matrix(obj)
    if mat is not None:
        embeddings = [mat[i].astype(np.float32, copy=False) for i in range(mat.shape[0])]
        return embeddings, texts, metas

    if isinstance(obj, (list, tuple)):
        # Tuple/List (embeddings, texts, metas) format
        if len(obj) in (2, 3) and _to_float_matrix(obj[0]) is not None and _looks_like_text_seq(obj[1]):
            embeddings = _expand_embeddings(obj[0])
            texts, metas = _coerce_texts_and_metas(obj[1])
            if len(obj) >= 3 and isinstance(obj[2], (list, tuple)):
                meta_list = [m if isinstance(m, dict) else {} for m in obj[2]]
                if texts:
                    if len(meta_list) >= len(texts):
                        metas = meta_list[:len(texts)]
                    else:
                        metas = list(meta_list) + [{} for _ in range(len(texts) - len(meta_list))]
                else:
                    metas = meta_list
            return embeddings, texts, metas

    if isinstance(obj, list):
        # List of embeddings
        if obj and _to_float_vector(obj[0]) is not None:
            for item in obj:
                _append(_to_float_vector(item))
        # List of dicts
        elif obj and isinstance(obj[0], dict):
            for item in obj:
                emb = _extract_embedding_from_mapping(item)
                if emb is None:
                    continue
                _append(emb, _extract_text_from_mapping(item), _extract_meta_from_mapping(item))
        # List of tuples (text, emb, meta)
        elif obj and isinstance(obj[0], (list, tuple)):
            for item in obj:
                if not item:
                    continue
                emb = None
                text = None
                meta = None
                if len(item) >= 1:
                    emb = _to_float_vector(item[0])
                if emb is None and len(item) >= 2:
                    emb = _to_float_vector(item[1])
                    if isinstance(item[0], str):
                        text = item[0]
                if emb is not None and text is None and len(item) >= 2 and isinstance(item[1], str):
                    text = item[1]
                if len(item) >= 3 and isinstance(item[2], dict):
                    meta = item[2]
                _append(emb, text, meta)
    elif isinstance(obj, dict):
        emb_list = None
        for key in ("embeddings", "vectors", "embedding"):
            if key in obj:
                emb_list = obj.get(key)
                break
        text_list = None
        for key in ("texts", "documents", "contents", "chunks"):
            if key in obj:
                text_list = obj.get(key)
                break
        meta_list = obj.get("metadatas") or obj.get("metadata")

        if emb_list is not None:
            embeddings = _expand_embeddings(emb_list)
        if text_list is not None and isinstance(text_list, (list, tuple)):
            if all(isinstance(t, str) for t in text_list):
                texts = [t for t in text_list if isinstance(t, str) and t.strip()]
            else:
                texts, metas = _coerce_texts_and_metas(text_list)
        if meta_list is not None and isinstance(meta_list, list):
            meta_norm = [m if isinstance(m, dict) else {} for m in meta_list]
            if texts and len(meta_norm) >= len(texts):
                metas = meta_norm[:len(texts)]
            else:
                metas = meta_norm

    # Clean up None embeddings
    embeddings = [e for e in embeddings if e is not None]
    return embeddings, texts, metas


def _compose_system_instruction(
    base_personality: str,
    history_excerpt: str | None,
    display_name: str | None = None,
    rag_context: str | None = None,
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
    rag = rag_context if rag_context is not None else _load_rag_context()
    if rag:
        parts.append(
            f"""以下は外部知識ベースの抜粋です。
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
# 会話履歴（chat_logs）について:
# - PERSIST_CHAT_LOGS=1 のときのみ DB に保存する（既定: 0=保存しない）
# - INJECT_CHAT_HISTORY=1 のときのみ 過去ログをプロンプトに注入する（既定: 0=注入しない）
PERSIST_CHAT_LOGS = _env_flag("PERSIST_CHAT_LOGS", "0")
INJECT_CHAT_HISTORY = _env_flag("INJECT_CHAT_HISTORY", "0")
PURGE_CHAT_LOGS_ON_START = _env_flag("PURGE_CHAT_LOGS_ON_START", "0")


def _get_pool() -> ConnectionPool:
    global DB_POOL
    if DB_POOL is None:
        DB_POOL = ConnectionPool(
            DATABASE_URL,
            min_size=int(os.getenv("DB_MIN_CONN", "1")),
            max_size=int(os.getenv("DB_MAX_CONN", "5")),
            kwargs={"autocommit": False},
        )
    return DB_POOL


@contextlib.contextmanager
def _db():
    pool = _get_pool()
    with pool.connection() as con:
        with con.transaction():
            yield con


def _init_db():
    with _db() as con:
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
        con.execute(
            """
            DO $$
            BEGIN
                IF NOT EXISTS (
                    SELECT 1
                    FROM pg_constraint
                    WHERE conname = 'chat_logs_user_id_fkey'
                ) THEN
                    DELETE FROM chat_logs
                    WHERE user_id NOT IN (SELECT id FROM users);
                    ALTER TABLE chat_logs
                        ADD CONSTRAINT chat_logs_user_id_fkey
                        FOREIGN KEY (user_id) REFERENCES users(id)
                        ON DELETE CASCADE;
                END IF;
            END $$;
            """
        )
        _ensure_rag_schema(con)
        _ensure_rag_docs_schema(con)


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
        with _db() as con:
            cur = con.execute("SELECT display_name FROM users WHERE id=%s", (uid,))
            row = cur.fetchone()
            return row[0] if row else "ユーザー"

    return await asyncio.to_thread(_fetch)


# ====== 会話ログ（PostgreSQL に保存） ======
async def append_log(uid: str, rec: dict) -> None:
    if not PERSIST_CHAT_LOGS:
        return
    rec = dict(rec)
    rec.setdefault("ts", _jst_now_iso())
    ts_val = rec.get("ts")
    try:
        ts = datetime.fromisoformat(ts_val)
    except Exception:
        ts = datetime.now(timezone.utc)

    def _insert():
        with _db() as con:
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
    if not INJECT_CHAT_HISTORY:
        return ""
    def _fetch() -> str:
        with _db() as con:
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


async def _purge_chat_logs() -> None:
    def _purge():
        with _db() as con:
            con.execute("TRUNCATE TABLE chat_logs")

    await asyncio.to_thread(_purge)


# ====== RAG（vectors.pt -> DB） ======
def _normalize_matrix(mat: np.ndarray) -> np.ndarray:
    if mat.size == 0:
        return mat
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / (norms + 1e-8)

def _vector_literal(vec: np.ndarray) -> str:
    if vec is None:
        return "[]"
    if not isinstance(vec, np.ndarray):
        vec = np.asarray(vec, dtype=np.float32)
    vec = vec.astype(np.float32, copy=False).ravel()
    parts = (repr(float(x)) for x in vec.tolist())
    return "[" + ",".join(parts) + "]"


def _infer_vector_dim(embeddings: list[np.ndarray]) -> int:
    for emb in embeddings:
        if emb is None:
            continue
        if isinstance(emb, np.ndarray) and emb.ndim == 1 and emb.size:
            return int(emb.shape[0])
        arr = _to_float_vector(emb)
        if arr is not None and arr.size:
            return int(arr.shape[0])
    return 0


def _rag_table_exists(con) -> bool:
    cur = con.execute("SELECT to_regclass('public.rag_chunks')")
    row = cur.fetchone()
    return bool(row and row[0])


def _rag_embedding_column_type(con) -> str:
    cur = con.execute(
        """
        SELECT data_type, udt_name
        FROM information_schema.columns
        WHERE table_name='rag_chunks' AND column_name='embedding'
        """
    )
    row = cur.fetchone()
    if not row:
        return ""
    data_type, udt_name = row
    if udt_name == "vector":
        return "vector"
    if data_type == "ARRAY":
        return "array"
    return udt_name or data_type or ""


def _ensure_vector_index(con) -> None:
    if RAG_VECTOR_INDEX == "none":
        return
    try:
        if RAG_VECTOR_INDEX == "hnsw":
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS rag_chunks_embedding_hnsw_idx
                    ON rag_chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = %s, ef_construction = %s)
                """,
                (max(2, RAG_VECTOR_INDEX_M), max(8, RAG_VECTOR_INDEX_EF)),
            )
        elif RAG_VECTOR_INDEX == "ivfflat":
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS rag_chunks_embedding_ivfflat_idx
                    ON rag_chunks USING ivfflat (embedding vector_cosine_ops)
                    WITH (lists = %s)
                """,
                (max(1, RAG_VECTOR_INDEX_LISTS),),
            )
    except Exception as e:
        log.warning("RAG: failed to create vector index (%s): %s", RAG_VECTOR_INDEX, e)


def _ensure_rag_schema(con, embed_dim: int | None = None) -> None:
    global RAG_VECTOR_SQL_AVAILABLE, RAG_VECTOR_COLUMN_TYPE, RAG_VECTOR_DIM
    use_sql = RAG_VECTOR_MODE == "sql"
    vector_ok = False
    if use_sql:
        try:
            cur = con.execute("SELECT 1 FROM pg_extension WHERE extname='vector'")
            if cur.fetchone():
                vector_ok = True
            else:
                con.execute("CREATE EXTENSION IF NOT EXISTS vector")
                vector_ok = True
        except Exception as e:
            log.warning("RAG: pgvector extension unavailable: %s", e)
            vector_ok = False
    RAG_VECTOR_SQL_AVAILABLE = vector_ok

    if embed_dim and embed_dim > 0 and RAG_VECTOR_DIM <= 0:
        RAG_VECTOR_DIM = embed_dim
    dim = RAG_VECTOR_DIM if RAG_VECTOR_DIM > 0 else (embed_dim or 0)

    exists = _rag_table_exists(con)
    desired_vector = vector_ok and dim > 0
    if not exists:
        emb_type = "REAL[]"
        if desired_vector:
            emb_type = f"vector({dim})"
        con.execute(
            f"""
            CREATE TABLE IF NOT EXISTS rag_chunks(
                id BIGSERIAL PRIMARY KEY,
                idx INTEGER UNIQUE,
                content TEXT NOT NULL,
                metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                embedding {emb_type} NOT NULL,
                user_id TEXT NULL,
                doc_id BIGINT NULL
            )
            """
        )
    else:
        if desired_vector:
            col_type = _rag_embedding_column_type(con)
            if col_type != "vector":
                if dim > 0:
                    try:
                        con.execute(
                            f"""
                            ALTER TABLE rag_chunks
                            ALTER COLUMN embedding TYPE vector({dim})
                            USING embedding::vector
                            """
                        )
                    except Exception as e:
                        log.warning("RAG: failed to convert embedding column to vector(%d): %s", dim, e)
                else:
                    log.warning("RAG: vector mode requested but dimension is unknown. Set RAG_VECTOR_DIM.")

        con.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS user_id TEXT")
        con.execute("ALTER TABLE rag_chunks ADD COLUMN IF NOT EXISTS doc_id BIGINT")

    con.execute(
        """
        CREATE INDEX IF NOT EXISTS rag_chunks_idx_idx
            ON rag_chunks(idx)
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS rag_chunks_user_doc_idx
            ON rag_chunks(user_id, doc_id)
        """
    )

    RAG_VECTOR_COLUMN_TYPE = _rag_embedding_column_type(con) or ("vector" if desired_vector else "array")
    if vector_ok and RAG_VECTOR_COLUMN_TYPE == "vector":
        _ensure_vector_index(con)


def _ensure_rag_docs_schema(con) -> None:
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS rag_docs(
            id BIGSERIAL PRIMARY KEY,
            user_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            sha256 TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )
    con.execute(
        """
        CREATE INDEX IF NOT EXISTS rag_docs_user_idx
            ON rag_docs(user_id, created_at DESC, id DESC)
        """
    )
    con.execute(
        """
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1
                FROM pg_constraint
                WHERE conname = 'rag_docs_user_id_fkey'
            ) THEN
                DELETE FROM rag_docs
                WHERE user_id NOT IN (SELECT id FROM users);
                ALTER TABLE rag_docs
                    ADD CONSTRAINT rag_docs_user_id_fkey
                    FOREIGN KEY (user_id) REFERENCES users(id)
                    ON DELETE CASCADE;
            END IF;
        END $$;
        """
    )


def _rag_use_sql() -> bool:
    return RAG_VECTOR_MODE == "sql" and RAG_VECTOR_SQL_AVAILABLE and RAG_VECTOR_COLUMN_TYPE == "vector"


def _build_rag_cache_from_rows(rows: list[tuple]) -> _RagCache | None:
    if not rows:
        return None
    texts: list[str] = []
    metas: list[dict] = []
    embeddings: list[np.ndarray] = []
    user_ids: list[str | None] = []
    doc_ids: list[int | None] = []
    for row in rows:
        if len(row) >= 6:
            _idx, content, metadata, embedding, user_id, doc_id = row[:6]
        else:
            _idx, content, metadata, embedding = row[:4]
            user_id, doc_id = None, None
        if not content:
            continue
        emb = _to_float_vector(embedding)
        if emb is None:
            continue
        texts.append(content)
        metas.append(metadata if isinstance(metadata, dict) else {})
        embeddings.append(emb)
        user_ids.append(user_id)
        doc_ids.append(doc_id)
    if not embeddings:
        return None
    mat = np.stack(embeddings).astype(np.float32, copy=False)
    mat = _normalize_matrix(mat)
    return _RagCache(mat, texts, metas, user_ids, doc_ids)


def _load_rag_cache_from_db() -> _RagCache | None:
    with _db() as con:
        cur = con.execute(
            """
            SELECT idx, content, metadata, embedding::float4[], user_id, doc_id
            FROM rag_chunks
            ORDER BY idx ASC
            """
        )
        rows = cur.fetchall()
    return _build_rag_cache_from_rows(rows)


def _ingest_rag_vectors() -> _RagCache | None:
    if not RAG_VECTOR_ENABLED or not RAG_INGEST_ON_START:
        return None
    global RAG_VECTORS_PATH
    resolved = _resolve_vectors_path(RAG_VECTORS_PATH)
    if resolved != RAG_VECTORS_PATH:
        log.warning("RAG: vectors file not found at %s; using %s", RAG_VECTORS_PATH, resolved)
        RAG_VECTORS_PATH = resolved

    existing = 0
    rows: list[tuple] | None = None
    with _db() as con:
        cur = con.execute("SELECT COUNT(*) FROM rag_chunks")
        existing = cur.fetchone()[0]
        if existing and not RAG_FORCE_RELOAD:
            _ensure_rag_schema(con, RAG_VECTOR_DIM if RAG_VECTOR_DIM > 0 else None)
            log.info("RAG: rag_chunks already populated (%d rows).", existing)
            if RAG_CACHE_IN_MEMORY:
                cur = con.execute(
                    """
                    SELECT idx, content, metadata, embedding::float4[], user_id, doc_id
                    FROM rag_chunks
                    ORDER BY idx ASC
                    """
                )
                rows = cur.fetchall()
            else:
                return None
    if existing and not RAG_FORCE_RELOAD:
        return _build_rag_cache_from_rows(rows or [])

    embeddings, vec_texts, vec_metas = _load_vectors_pt(RAG_VECTORS_PATH)
    if not embeddings:
        log.warning("RAG: no embeddings loaded from %s", RAG_VECTORS_PATH)
        if existing:
            log.warning("RAG: keeping existing rag_chunks (%d rows).", existing)
            return _load_rag_cache_from_db() if RAG_CACHE_IN_MEMORY else None
        return None

    texts_path = _guess_rag_texts_path()
    if texts_path and texts_path != RAG_TEXTS_PATH:
        log.info("RAG: using texts file %s", texts_path)
    file_texts, file_metas = _load_texts_from_path(texts_path)
    texts = file_texts if file_texts else vec_texts
    metas = file_metas if file_metas else vec_metas

    if not texts:
        log.warning("RAG: no texts provided. Set RAG_TEXTS_PATH or embed texts into vectors.pt.")
        if existing:
            log.warning("RAG: keeping existing rag_chunks (%d rows).", existing)
            return _load_rag_cache_from_db() if RAG_CACHE_IN_MEMORY else None
        return None

    if len(texts) != len(embeddings):
        n = min(len(texts), len(embeddings))
        log.warning(
            "RAG: vectors/texts length mismatch (vectors=%d, texts=%d). Truncating to %d.",
            len(embeddings),
            len(texts),
            n,
        )
        if n <= 0:
            if existing:
                log.warning("RAG: keeping existing rag_chunks (%d rows).", existing)
                return _load_rag_cache_from_db() if RAG_CACHE_IN_MEMORY else None
            return None
        embeddings = embeddings[:n]
        texts = texts[:n]
        metas = metas[:n] if metas else []

    if len(metas) != len(texts):
        metas = list(metas) + [{} for _ in range(len(texts) - len(metas))]

    embed_dim = _infer_vector_dim(embeddings)

    with _db() as con:
        _ensure_rag_schema(con, embed_dim)
        use_vector = _rag_embedding_column_type(con) == "vector"
        insert_sql = (
            "INSERT INTO rag_chunks(idx, content, metadata, embedding, user_id, doc_id) VALUES (%s, %s, %s, %s::vector, %s, %s)"
            if use_vector
            else "INSERT INTO rag_chunks(idx, content, metadata, embedding, user_id, doc_id) VALUES (%s, %s, %s, %s, %s, %s)"
        )
        if RAG_FORCE_RELOAD:
            con.execute("TRUNCATE TABLE rag_chunks")
        batch: list[tuple] = []
        inserted_embeddings: list[np.ndarray] = []
        inserted_texts: list[str] = []
        inserted_metas: list[dict] = []
        for i, (emb, text, meta) in enumerate(zip(embeddings, texts, metas)):
            text = (text or "").strip()
            if not text:
                continue
            emb = _to_float_vector(emb)
            if emb is None:
                continue
            idx = len(inserted_embeddings)
            inserted_embeddings.append(emb)
            inserted_texts.append(text)
            inserted_metas.append(meta if isinstance(meta, dict) else {})
            if use_vector:
                emb_val: Any = _vector_literal(emb)
            else:
                emb_val = emb.astype(np.float32).tolist()
            batch.append((idx, text, Json(inserted_metas[-1]), emb_val, None, None))
            if len(batch) >= 200:
                con.cursor().executemany(insert_sql, batch)
                batch.clear()
        if batch:
            con.cursor().executemany(insert_sql, batch)

    if not inserted_embeddings:
        log.warning("RAG: no rows inserted into rag_chunks.")
        return None

    log.info("RAG: inserted %d rows into rag_chunks.", len(inserted_embeddings))
    if RAG_CACHE_IN_MEMORY:
        mat = np.stack(inserted_embeddings).astype(np.float32, copy=False)
        mat = _normalize_matrix(mat)
        return _RagCache(mat, inserted_texts, inserted_metas, [None] * len(inserted_texts), [None] * len(inserted_texts))
    return None


async def _ensure_rag_ready() -> None:
    global RAG_CACHE
    if not RAG_VECTOR_ENABLED or not RAG_INGEST_ON_START:
        return
    try:
        cache = await asyncio.to_thread(_ingest_rag_vectors)
        if cache is None and RAG_CACHE_IN_MEMORY:
            cache = await asyncio.to_thread(_load_rag_cache_from_db)
        if cache is not None:
            RAG_CACHE = cache
    except Exception as e:
        log.warning("RAG: init failed: %s", e)


async def _get_rag_cache() -> _RagCache | None:
    global RAG_CACHE
    if not RAG_VECTOR_ENABLED:
        return None
    if not RAG_CACHE_IN_MEMORY:
        return await asyncio.to_thread(_load_rag_cache_from_db)
    if RAG_CACHE is not None:
        return RAG_CACHE
    async with RAG_CACHE_LOCK:
        if RAG_CACHE is None:
            RAG_CACHE = await asyncio.to_thread(_load_rag_cache_from_db)
    return RAG_CACHE


def _build_embed_config(task_type: str | None, title: str | None = None):
    if not task_type and not title:
        return None
    kwargs: dict[str, Any] = {}
    if task_type:
        kwargs["task_type"] = task_type
    if title:
        kwargs["title"] = title
    try:
        return types.EmbedContentConfig(**kwargs)
    except TypeError:
        if "title" in kwargs:
            kwargs.pop("title", None)
        if not kwargs:
            return None
        try:
            return types.EmbedContentConfig(**kwargs)
        except Exception:
            return None


async def _embed_query(client: genai.Client, text: str) -> np.ndarray | None:
    if not text:
        return None
    if not RAG_EMBED_MODEL:
        log.warning("RAG: RAG_EMBED_MODEL is empty.")
        return None

    def _embed() -> list[float]:
        cfg = _build_embed_config(RAG_EMBED_TASK_TYPE)
        resp = client.models.embed_content(
            model=RAG_EMBED_MODEL,
            contents=text,
            config=cfg,
        )
        if getattr(resp, "embeddings", None):
            return resp.embeddings[0].values or []
        return []

    try:
        values = await asyncio.to_thread(_embed)
    except Exception as e:
        log.warning("RAG: embedding failed: %s", e)
        return None
    if not values:
        return None
    vec = np.asarray(values, dtype=np.float32)
    if vec.ndim != 1:
        return None
    return vec


async def _embed_document_chunks(client: genai.Client, contents: list[str], title: str | None = None) -> list[list[float]]:
    if not contents:
        return []
    if not RAG_EMBED_MODEL:
        raise RuntimeError("RAG_EMBED_MODEL is empty.")
    batch_size = max(1, min(PDF_EMBED_BATCH_SIZE, 128))
    embeddings: list[list[float]] = []

    for i in range(0, len(contents), batch_size):
        batch = contents[i : i + batch_size]

        def _embed_batch() -> list[list[float]]:
            cfg = _build_embed_config(RAG_DOC_EMBED_TASK_TYPE, title)
            resp = client.models.embed_content(
                model=RAG_EMBED_MODEL,
                contents=batch,
                config=cfg,
            )
            if getattr(resp, "embeddings", None):
                return [(emb.values or []) for emb in resp.embeddings]
            return []

        values = await asyncio.to_thread(_embed_batch)
        if not values or len(values) != len(batch):
            raise RuntimeError("Failed to embed document chunks.")
        embeddings.extend(values)

    return embeddings


def _tokenize_fallback_query(query: str) -> list[str]:
    q = (query or "").strip()
    if not q:
        return []
    tokens: set[str] = set()
    for m in re.finditer(r"[A-Za-z0-9_]{2,}", q):
        tokens.add(m.group().lower())
    for m in re.finditer(r"[ぁ-んァ-ン一-龠]{2,}", q):
        tokens.add(m.group())
    if not tokens and len(q) >= 2:
        tokens.add(q.lower())
    return list(tokens)


def _simple_text_rag(query: str, texts: list[str]) -> str:
    if not texts:
        return ""
    tokens = _tokenize_fallback_query(query)
    if not tokens:
        return ""
    scored: list[tuple[int, int, str]] = []
    for i, text in enumerate(texts):
        if not text:
            continue
        lowered = text.lower()
        score = 0
        for tok in tokens:
            if tok in lowered:
                score += 1
        if score <= 0:
            continue
        scored.append((score, i, text))
    if not scored:
        return ""
    scored.sort(key=lambda x: (-x[0], x[1]))
    parts: list[str] = []
    budget = RAG_FALLBACK_MAX_TOTAL_CHARS
    for rank, (_score, _idx, text) in enumerate(scored[:max(1, RAG_FALLBACK_TOP_K)]):
        snippet = _truncate_text(text, RAG_FALLBACK_PER_CHUNK_CHARS)
        if not snippet:
            continue
        block = f"[F{rank+1}]\n{snippet}"
        if budget - len(block) < -100:
            break
        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            break
    return "\n\n".join(parts)


def _build_rag_blocks(items: list[tuple[str, dict, float]]) -> str:
    parts: list[str] = []
    budget = RAG_VECTOR_MAX_TOTAL_CHARS
    for text, meta, score in items:
        if score < RAG_MIN_SIM:
            continue
        snippet = _truncate_text(text, RAG_VECTOR_PER_CHUNK_CHARS)
        if not snippet:
            continue
        label = f"[{len(parts)+1}]"
        if isinstance(meta, dict):
            src = meta.get("source") or meta.get("title") or meta.get("id")
            if src:
                label = f"{label} {src}"
        block = f"{label}\n{snippet}"
        if budget - len(block) < -100:
            break
        parts.append(block)
        budget -= len(block)
        if budget <= 0:
            break
    return "\n\n".join(parts)


async def _get_fallback_texts() -> list[str]:
    global RAG_FALLBACK_TEXTS
    if RAG_FALLBACK_TEXTS is not None:
        return RAG_FALLBACK_TEXTS
    async with RAG_FALLBACK_TEXTS_LOCK:
        if RAG_FALLBACK_TEXTS is None:
            path = _guess_rag_texts_path()
            if not path:
                RAG_FALLBACK_TEXTS = []
            else:
                texts, _ = await asyncio.to_thread(_load_texts_from_path, path)
                RAG_FALLBACK_TEXTS = texts
    return RAG_FALLBACK_TEXTS


async def _fallback_text_rag(query: str, texts_hint: list[str] | None = None) -> str:
    if not RAG_FALLBACK_ENABLED:
        return ""
    texts = texts_hint if texts_hint else await _get_fallback_texts()
    return _simple_text_rag(query, texts)


async def _retrieve_rag_context_sql(query: str, client: genai.Client, uid: str | None = None) -> str:
    if not RAG_VECTOR_ENABLED or not _rag_use_sql():
        return ""
    qvec = await _embed_query(client, query)
    if qvec is None:
        return await _fallback_text_rag(query)
    if RAG_VECTOR_DIM and qvec.shape[0] != RAG_VECTOR_DIM:
        log.warning("RAG: embedding dim mismatch (query=%d, index=%d)", qvec.shape[0], RAG_VECTOR_DIM)
        return await _fallback_text_rag(query)

    vec_literal = _vector_literal(qvec)
    top_k = max(1, RAG_TOP_K)

    def _fetch():
        with _db() as con:
            cur = con.execute(
                """
                SELECT content, metadata, 1 - (embedding <=> %s::vector) AS sim
                FROM rag_chunks
                WHERE user_id IS NULL OR user_id = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (vec_literal, uid, vec_literal, top_k),
            )
            return cur.fetchall()

    try:
        rows = await asyncio.to_thread(_fetch)
    except Exception as e:
        log.warning("RAG: SQL retrieval failed: %s", e)
        return await _fallback_text_rag(query)

    if not rows:
        return await _fallback_text_rag(query)

    items: list[tuple[str, dict, float]] = []
    for content, metadata, sim in rows:
        if not content:
            continue
        meta = metadata if isinstance(metadata, dict) else {}
        try:
            score = float(sim)
        except Exception:
            score = 0.0
        items.append((content, meta, score))

    context = _build_rag_blocks(items)
    if context:
        return context
    return await _fallback_text_rag(query)


async def _retrieve_rag_context_memory(query: str, client: genai.Client, uid: str | None = None) -> str:
    if not RAG_VECTOR_ENABLED:
        return ""
    cache = await _get_rag_cache()
    if cache is None or not cache.texts:
        return await _fallback_text_rag(query, cache.texts if cache else None)

    qvec = await _embed_query(client, query)
    if qvec is None:
        return await _fallback_text_rag(query, cache.texts if cache else None)
    if cache.dim and qvec.shape[0] != cache.dim:
        log.warning("RAG: embedding dim mismatch (query=%d, index=%d)", qvec.shape[0], cache.dim)
        return await _fallback_text_rag(query, cache.texts if cache else None)

    allowed_idxs: list[int] = []
    if uid:
        for i, user_id in enumerate(cache.user_ids):
            if user_id is None or user_id == uid:
                allowed_idxs.append(i)
    else:
        for i, user_id in enumerate(cache.user_ids):
            if user_id is None:
                allowed_idxs.append(i)
    if not allowed_idxs:
        return await _fallback_text_rag(query, cache.texts if cache else None)

    qvec = qvec / (np.linalg.norm(qvec) + 1e-8)
    emb_mat = cache.embeddings[allowed_idxs]
    sims = emb_mat @ qvec
    if sims.size == 0:
        return ""

    top_k = max(1, min(RAG_TOP_K, sims.shape[0]))
    idxs = np.argpartition(sims, -top_k)[-top_k:]
    idxs = idxs[np.argsort(sims[idxs])[::-1]]

    items: list[tuple[str, dict, float]] = []
    for rel_idx in idxs:
        idx = allowed_idxs[rel_idx]
        text = cache.texts[idx]
        if not text:
            continue
        meta = cache.metas[idx] if idx < len(cache.metas) else {}
        items.append((text, meta, float(sims[rel_idx])))

    context = _build_rag_blocks(items)
    if context:
        return context
    return await _fallback_text_rag(query, cache.texts if cache else None)


async def _retrieve_rag_context(query: str, client: genai.Client, uid: str | None = None) -> str:
    if not RAG_VECTOR_ENABLED:
        return ""
    if _rag_use_sql():
        return await _retrieve_rag_context_sql(query, client, uid)
    return await _retrieve_rag_context_memory(query, client, uid)

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
    if PURGE_CHAT_LOGS_ON_START:
        try:
            await _purge_chat_logs()
            log.warning("chat_logs purged on startup (PURGE_CHAT_LOGS_ON_START=1).")
        except Exception as e:
            log.warning("failed to purge chat_logs on startup: %s", e)
    await _ensure_rag_ready()
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
        try:
            with _db() as con:
                cur = con.execute("SELECT 1 FROM users WHERE id=%s", (uid,))
                if cur.fetchone():
                    return False
                # bcrypt_sha256 でハッシュ化（72バイト制限の影響を避ける）
                con.execute(
                    "INSERT INTO users(id, pw_hash, display_name, created_at) VALUES(%s,%s,%s,NOW())",
                    (uid, bcrypt.hash(pw), name)
                )
                return True
        except pg_errors.UniqueViolation:
            return False
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
        with _db() as con:
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

# ====== PDF RAG（アップロード／一覧／削除） ======
@app.post("/rag/upload/pdf")
async def rag_upload_pdf(req: Request, file: UploadFile = File(...)):
    uid = req.session.get("user_id")
    if not uid:
        raise HTTPException(401)
    if not RAG_VECTOR_ENABLED:
        raise HTTPException(400, "RAG は無効です。")

    filename = os.path.basename(file.filename or "document.pdf")
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(400, "PDFのみアップロード可能です。")
    content_type = (file.content_type or "").lower()
    if content_type and content_type != "application/pdf":
        raise HTTPException(400, "Content-Type が application/pdf ではありません。")

    data = await file.read()
    if not data:
        raise HTTPException(400, "ファイルが空です。")
    if len(data) > PDF_MAX_BYTES:
        raise HTTPException(413, f"PDFサイズが上限({PDF_MAX_BYTES} bytes)を超えています。")
    if not _looks_like_pdf(data):
        raise HTTPException(400, "PDFヘッダーが確認できません。")

    pages = await asyncio.to_thread(extract_pdf_text, data)
    pages = [p for p in pages if (p.get("text") or "").strip()]
    if not pages:
        raise HTTPException(400, "PDFからテキストを抽出できませんでした。")

    chunks = await asyncio.to_thread(
        chunk_text,
        pages,
        PDF_CHUNK_SIZE_CHARS,
        PDF_CHUNK_OVERLAP_CHARS,
    )
    if not chunks:
        raise HTTPException(400, "抽出テキストが短すぎます。")

    contents = [c.get("content", "") for c in chunks]
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        raw_embeddings = await _embed_document_chunks(client, contents, title=filename)
    except Exception as e:
        log.warning("RAG: document embedding failed: %s", e)
        raise HTTPException(500, "embedding failed")

    embeddings: list[np.ndarray] = []
    for values in raw_embeddings:
        emb = _to_float_vector(values)
        if emb is None:
            raise HTTPException(500, "embedding failed")
        embeddings.append(emb)
    embed_dim = _infer_vector_dim(embeddings)
    if embed_dim <= 0:
        raise HTTPException(500, "embedding failed")

    sha256 = _sha256_hex(data)

    def _insert():
        with _db() as con:
            _ensure_rag_schema(con, embed_dim)
            _ensure_rag_docs_schema(con)
            cur = con.execute(
                """
                INSERT INTO rag_docs(user_id, filename, sha256)
                VALUES (%s, %s, %s)
                RETURNING id
                """,
                (uid, filename, sha256),
            )
            doc_id = cur.fetchone()[0]
            use_vector = _rag_embedding_column_type(con) == "vector"
            insert_sql = (
                "INSERT INTO rag_chunks(idx, content, metadata, embedding, user_id, doc_id) "
                "VALUES (%s, %s, %s, %s::vector, %s, %s)"
                if use_vector
                else "INSERT INTO rag_chunks(idx, content, metadata, embedding, user_id, doc_id) "
                "VALUES (%s, %s, %s, %s, %s, %s)"
            )
            batch: list[tuple] = []
            inserted = 0
            for chunk, emb in zip(chunks, embeddings):
                content = (chunk.get("content") or "").strip()
                if not content:
                    continue
                meta = dict(chunk.get("meta") or {})
                meta["filename"] = filename
                meta["doc_id"] = doc_id
                page_start = meta.get("page_start")
                if page_start:
                    meta.setdefault("source", f"{filename} p.{page_start}")
                else:
                    meta.setdefault("source", filename)
                emb_val = _vector_literal(emb) if use_vector else emb.astype(np.float32).tolist()
                batch.append((None, content, Json(meta), emb_val, uid, doc_id))
                inserted += 1
                if len(batch) >= 200:
                    con.cursor().executemany(insert_sql, batch)
                    batch.clear()
            if batch:
                con.cursor().executemany(insert_sql, batch)
        return doc_id, inserted

    doc_id, inserted_chunks = await asyncio.to_thread(_insert)
    if RAG_CACHE_IN_MEMORY:
        global RAG_CACHE
        RAG_CACHE = None

    return {
        "ok": True,
        "doc_id": doc_id,
        "filename": filename,
        "pages": len(pages),
        "chunks": inserted_chunks,
    }


@app.get("/rag/docs")
async def rag_list_docs(req: Request):
    uid = req.session.get("user_id")
    if not uid:
        raise HTTPException(401)

    def _fetch():
        with _db() as con:
            cur = con.execute(
                """
                SELECT d.id, d.filename, d.created_at, d.sha256, COUNT(c.id) AS chunks
                FROM rag_docs d
                LEFT JOIN rag_chunks c
                    ON c.doc_id = d.id AND c.user_id = d.user_id
                WHERE d.user_id = %s
                GROUP BY d.id
                ORDER BY d.created_at DESC, d.id DESC
                """,
                (uid,),
            )
            rows = cur.fetchall()
        docs: list[dict] = []
        for doc_id, filename, created_at, sha256, chunks in rows:
            try:
                created_iso = created_at.isoformat()
            except Exception:
                created_iso = str(created_at)
            docs.append(
                {
                    "doc_id": doc_id,
                    "filename": filename,
                    "created_at": created_iso,
                    "sha256": sha256,
                    "chunks": int(chunks or 0),
                }
            )
        return docs

    docs = await asyncio.to_thread(_fetch)
    return {"docs": docs}


@app.delete("/rag/docs/{doc_id}")
async def rag_delete_doc(doc_id: int, req: Request):
    uid = req.session.get("user_id")
    if not uid:
        raise HTTPException(401)

    def _delete():
        with _db() as con:
            cur = con.execute(
                "SELECT 1 FROM rag_docs WHERE id=%s AND user_id=%s",
                (doc_id, uid),
            )
            if not cur.fetchone():
                return None
            cur = con.execute(
                "DELETE FROM rag_chunks WHERE user_id=%s AND doc_id=%s",
                (uid, doc_id),
            )
            chunks_deleted = cur.rowcount or 0
            con.execute(
                "DELETE FROM rag_docs WHERE id=%s AND user_id=%s",
                (doc_id, uid),
            )
            return chunks_deleted

    chunks_deleted = await asyncio.to_thread(_delete)
    if chunks_deleted is None:
        raise HTTPException(404, "doc not found")
    if RAG_CACHE_IN_MEMORY:
        global RAG_CACHE
        RAG_CACHE = None
    return {"ok": True, "doc_id": doc_id, "chunks_deleted": chunks_deleted}

# ====== テキストチャット（ログイン必須） ======
@app.post("/chat")
async def chat(req: Request, payload: dict):
    uid = req.session.get("user_id")
    if not uid:
        raise HTTPException(401)
    user_text = (payload.get("text") or "").strip() or "こんにちは。"
    excerpt = (await read_tail(uid, 6000)).strip() if INJECT_CHAT_HISTORY else ""
    display_name = req.session.get("display_name", "ユーザー")

    client = genai.Client(api_key=GEMINI_API_KEY)
    rag_context = ""
    if RAG_VECTOR_ENABLED:
        try:
            rag_context = await _retrieve_rag_context(user_text, client, uid)
        except Exception as e:
            log.warning("RAG: retrieval failed: %s", e)
            rag_context = ""
    sys_inst = _compose_system_instruction(
        SYSTEM_PERSONALITY,
        excerpt,
        display_name,
        rag_context=rag_context if rag_context else None,
    )

    async def _call_via_live_once() -> str:
        """
        generateContent が使えない/失敗する環境向けに、Gemini Live で 1 turn だけテキスト生成する。
        - MODEL_ID が native-audio の場合は AUDIO を要求し、output_transcription を拾う。
        - そうでない場合は TEXT を要求する。
        """
        if not LIVE_MODEL_AVAILABLE:
            raise RuntimeError("Gemini Live model is not available.")

        native_audio_model = "native-audio" in (MODEL_ID or "")
        response_modalities = ["AUDIO"] if native_audio_model else ["TEXT"]
        config = {
            "response_modalities": response_modalities,
            "system_instruction": sys_inst,
            "thinking_config": {
                "thinking_budget": 0,
                "include_thoughts": False,
            },
        }
        if native_audio_model:
            config["output_audio_transcription"] = {}

        def _merge_text(current: str, incoming: str) -> str:
            incoming = (incoming or "").strip("\ufeff")
            if not incoming:
                return current
            if not current:
                return incoming
            # Live の一部フィールドは「全文」を返すことがあるため、重複を抑える
            if incoming.startswith(current):
                return incoming
            if current.startswith(incoming):
                return current
            if incoming in current:
                return current
            if current in incoming:
                return incoming
            sep = "" if (current.endswith(("\n", " ")) or incoming.startswith(("\n", " "))) else ""
            return current + sep + incoming

        async with client.aio.live.connect(model=MODEL_ID, config=config) as sess:
            await sess.send(input=user_text, end_of_turn=True)
            out_text = ""
            # 短い 1turn なのでタイムアウトを設ける（無限待ち防止）
            async with asyncio.timeout(20):
                async for msg in sess.receive():
                    sc = getattr(msg, "server_content", None)

                    piece = getattr(msg, "text", None)
                    if not piece and sc and getattr(sc, "output_transcription", None):
                        try:
                            piece = sc.output_transcription.text or ""
                        except Exception:
                            piece = None
                    if not piece and sc and getattr(sc, "model_turn", None):
                        try:
                            for p in (sc.model_turn.parts or []):
                                if getattr(p, "text", None):
                                    piece = p.text
                                    break
                        except Exception:
                            piece = None

                    if piece:
                        out_text = _merge_text(out_text, piece)

                    if sc and getattr(sc, "turn_complete", False):
                        break

            return (out_text or "").strip()

    async def _call_with_fallback():
        global TEXT_MODEL_ID
        # NOTE:
        # - モデル候補の末尾に「存在しないモデルID」が混ざっている場合でも、
        #   先に遭遇した 429 等の "待てば回復" 系エラーを上書きして
        #   404 を最終エラーにしてしまうと、UI には誤誘導な文言が出てしまう。
        #   そのため、最終的に全滅した場合は「より有用なエラー」を優先して返す。
        best_retryable_err: Exception | None = None
        last_err: Exception | None = None
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
                    if retryable and best_retryable_err is None:
                        best_retryable_err = e
                    remaining = TEXT_RETRY_MAX - attempt - 1
                    if retryable and remaining > 0:
                        backoff = TEXT_RETRY_BASE_DELAY * (2 ** attempt) * (0.7 + 0.6 * random.random())
                        log.warning("chat: transient error on %s (try %d/%d): %s -> retry in %.2fs",
                                    model_id, attempt + 1, TEXT_RETRY_MAX, e, backoff)
                        await asyncio.sleep(backoff)
                        continue
                    log.warning("chat: model %s failed (try %d/%d): %s", model_id, attempt + 1, TEXT_RETRY_MAX, e)
                    break  # 次のモデル候補へ
        # 全候補が失敗した場合は、末尾の 404 等よりも、先に遭遇した 429/503 等を優先。
        if best_retryable_err is not None:
            raise best_retryable_err
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
            upper = str(e).upper()
            if "QUOTA" in upper or "BILLING" in upper or "PLAN" in upper:
                friendly = "すみません、Gemini API の利用上限に達している可能性があります。プラン/課金設定/クォータをご確認ください。"
            else:
                friendly = "すみません、モデルが混み合っています。少し待ってからもう一度お試しください。"
            return {"reply": friendly, "error": "temporarily_unavailable"}
        # generateContent が弾かれる環境では Live だけ動くことがあるため、最後に代替を試す
        if CHAT_LIVE_FALLBACK and LIVE_MODEL_AVAILABLE:
            try:
                live_text = await _call_via_live_once()
                if live_text:
                    live_text = live_text.strip()[:1000]
                    await append_log(uid, {"user_id": uid, "role":"assistant", "modality":"text", "text": live_text})
                    return {"reply": live_text, "via": "live"}
            except Exception as live_e:
                log.exception("chat: Live fallback failed: %s", live_e)
        msg = "Gemini APIキーまたはモデル設定を確認してください。"
        return JSONResponse({"reply": "すみません、応答に失敗しました。", "error": msg, "detail": str(e)}, status_code=500)
    except Exception as e:
        log.exception("text generation failed: %s", e)
        if _is_retryable_genai_error(e):
            friendly = "すみません、ただいま混雑しています。もう一度お試しください。"
            return {"reply": friendly, "error": "temporarily_unavailable"}
        return JSONResponse({"reply": "すみません、応答に失敗しました。", "error": str(e)}, status_code=500)

# ======（重要）同一オリジンにまとめる：フロントエンド配信 ======
# backend/ から見た ../frontend を配信対象にする
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
if not os.path.isdir(FRONTEND_DIR):
    log.warning("Frontend dir not found: %s", FRONTEND_DIR)

# 既存の /chat を定義した “後” に、最後にマウントするのが安全
# html=True により / へアクセスで index.html を返す
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
