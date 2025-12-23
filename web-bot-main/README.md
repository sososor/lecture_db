# web-bot

*開発途中のプロジェクトです。

## 概要
本プロジェクトは、Gemini API を利用した音声・テキスト対応のWebチャットボットです。  
FastAPI をバックエンドに、以下の機能を提供します。

- テキストチャット（Gemini Text Model）
- 音声入力・音声出力（Gemini Live + VOICEVOX）
- WebSocket によるリアルタイム音声通信
- PostgreSQL による会話ログ保存
- セッションベースのユーザー認証（登録 / ログイン）

フロントエンドは同一オリジンで配信されます。

---

## 主な機能
- ユーザー登録 / ログイン / ログアウト
- テキストチャット API
- 音声チャット（WebSocket）
- 会話履歴のDB保存
- RAG（data/ ディレクトリのテキストをプロンプトに注入）
- VOICEVOX による音声合成

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
