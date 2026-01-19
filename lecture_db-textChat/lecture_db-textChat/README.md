# web-bot

*開発途中のプロジェクトです。




##

##ペルソナ
N,S


ペルソナ（例）​
- 属性：新卒のクラウドエンジニア​
- 状況：クラウドエンジニアとして学ぶべき内容が多く、AWSの公式ドキュメントなどを読んで逐一情報を得たいが、分量が多く億劫に感じている。​
- 目的（Goals）​
ドキュメント全体を読む時間がないので、“今知りたい答え” を短時間で得たい​

重要ポイントを 自分の言葉で説明できる形 にまとめたい​

​
- 困りごと（Pain / Frustration）​

キーワード検索だと 該当箇所が多すぎる or 見つからない​

英語/専門用語が多く、理解に時間がかかる​

##ストーリーボード
Ｎ


Scene 1：課題・業務で「答えが必要」になる​
状況：資料/手順書/ドキュメントが長く、読む時間がない​
感情：焦り・不安（〆切、作業ミスのリスク）​

Scene 2：Webチャットボットにログイン​
行動：ユーザー登録 → ログイン（セッション維持）​
期待：「ここで聞けば早いはず」​

Scene 3：自然文で質問する​
入力例：「〇〇の手順は？」「△△の違いは？」「エラーの原因候補は？」​
ポイント：キーワードではなく “やりたいこと” で聞ける​

Scene 4：回答＋関連情報（ナレッジ）を受け取る​
出力：結論 → 理由 → 手順/注意点（短く整理）​
追加：関連する説明（ドキュメント由来の要約）を一緒に提示​
感情：理解できた／次の行動が明確になった​

Scene 5：追加質問で深掘りし、作業を完了​
行動：「具体例は？」「前提条件は？」「設定値のおすすめは？」と追い質問​
成果：必要情報が揃い、課題提出・発表準備・作業が完了​

Scene 6：後日、履歴を見返して再利用​
行動：過去の会話ログを参照（任意）​
効果：再調査が減り、学習/作業の効率が上がる​


##アプリの概要
Ｎ


##システム構成
S

## System Architecture Diagram

![System Architecture](System%20Architecture.png)



##アプリ構成、フォルダ構成




##デモ動画
ｓ
## Demo Video

[![Demo Video](https://img.youtube.com/vi/AFXqn3MlxAA/0.jpg)](https://www.youtube.com/watch?v=AFXqn3MlxAA)


##DB構成（外部、トランザクション、ER図）
Ｎ

##チーム開発方法
Ｓ

- GithubのKanbanボードを使い、タスクを設定し、進めた
- ディスコードで通話及びチャットでやり取り






-----------------------------------------------
## 概要
本プロジェクトは、Gemini API を利用したテキスト専用のWebチャットボットです。  
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
- Gemini API（Text）
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
│   ├── main.py             # API エントリーポイント
│   └── requirements.txt    # Python 依存関係
├── data/                   # RAG 用テキストデータ
│   └── knowledge.txt       # ナレッジデータ
├── frontend/               # フロントエンド（静的ファイル）
│   ├── index.html          # メイン画面
│   ├── auth.html           # ログイン / 登録画面
│   ├── app.js              # フロントエンドロジック
│   └── bus.js              # フロントエンドイベントバス
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
