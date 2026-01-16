# web-bot

*開発途中のプロジェクトです。

##

##ペルソナ

##ストーリーボード

##アプリの概要

##システム構成

##アプリ構成、フォルダ構成

##デモ動画

##DB構成（外部、トランザクション、ER図）

##チーム開発方法





## 概要
本プロジェクトは、Gemini API を利用した音声・テキスト対応のWebチャットボットです。  
FastAPI をバックエンドに、以下の機能を提供します。

- テキストチャット（Gemini Text Model）
- PostgreSQL（ユーザー認証 / 任意で会話ログ保存）
- セッションベースのユーザー認証（登録 / ログイン）

フロントエンドは同一オリジンで配信されます。

---

## 主な機能
- ユーザー登録 / ログイン / ログアウト
- テキストチャット API
- （任意）会話履歴のDB保存 / プロンプト注入（環境変数で制御）
- RAG（data/ テキスト注入 + vectors.pt ベクトルRAG）


---

## 技術スタック
### バックエンド
- Python 
- FastAPI
- WebSocket
- Gemini API（Text / Live）
- VOICEVOX
- PostgreSQL
- psycopg / psycopg_pool
- passlib（bcrypt_sha256）

### その他
- Docker / docker-compose
- セッション認証（Cookie）

---

## ディレクトリ構成（抜粋）
```text
.
├── backend/                # FastAPI バックエンド
│   ├── main.py             # API / WebSocket エントリーポイント
│   └── requirements.txt    # Python 依存関係
├── data/                   # RAG 用テキストデータ
│   └── knowledge.txt       # ナレッジデータ
├── frontend/               # フロントエンド（静的ファイル）
│   ├── index.html          # メイン画面
│   ├── auth.html           # ログイン / 登録画面
│   ├── app.js              # フロントエンドロジック
│   └── pcm-worklet.js      # 音声処理（Web Audio API）
├── .env.example            # 環境変数サンプル
├── .gitignore
├── Dockerfile              # アプリケーション用 Docker 定義
├── docker-compose.yml      # 開発用コンテナ構成
└── README.md
```

---

## ベクトルRAG（vectors.pt）
`/vectors.pt` に保存された埋め込みをPostgreSQLへ取り込み、テキストチャット時に関連情報を検索してプロンプトへ注入します。  
現在は **pgvector によるSQL検索（`RAG_VECTOR_MODE=sql`）** が既定です。  
DB は pgvector 拡張が必要です（`docker-compose.yml` は pgvector 搭載イメージを使用）。

### 使い方（最低限）
1. `vectors.pt` をリポジトリ直下に配置（既定パス: `/app/vectors.pt`）
2. 対応するテキストを `rag_texts.jsonl` などに用意し、`RAG_TEXTS_PATH` で指定  
   - 1行1JSONの形式で、`text`（本文）と任意の `metadata` を持つ想定
3. `.env` などで `RAG_TEXTS_PATH` と `RAG_EMBED_MODEL` を設定

### 例（JSONL 1行）
```json
{"text":"サンプル本文...", "metadata":{"source":"example.txt","title":"サンプル"}}
```

### 主な環境変数
- `RAG_VECTORS_PATH`（既定: `/app/vectors.pt`）
- `RAG_TEXTS_PATH`（必須: vectors と同件数のテキスト）
- `RAG_EMBED_MODEL`（既定: `text-embedding-004`）
- `RAG_FORCE_RELOAD`（`1` で再取り込み）
- `RAG_VECTOR_MODE`（`sql`|`memory`）
- `RAG_VECTOR_DIM`（埋め込み次元を明示する場合）
- `RAG_VECTOR_INDEX`（`hnsw`|`ivfflat`|`none`）
- `RAG_CACHE_IN_MEMORY`（SQLモード時は `0` 推奨）
