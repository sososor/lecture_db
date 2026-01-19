# AskMyPDF

*開発途中のプロジェクトです。

## ペルソナ
ペルソナ（例）  
属性：AWSを学習中の大学生  

状況：  
AWSに興味があるものの、学ぶべき内容が多く、AWSの公式ドキュメントなどを読んで逐一情報を得たいが、分量が多く億劫に感じている。AIの回答には正確性を求めたい。  

目的：  
ドキュメントを辞書的な使い方で使いたいので、“今知りたい答え” を短時間で得たい。重要ポイントをわかりやすい表現で解説してほしい。  

困りごと：  
・キーワード検索だと 該当箇所が多すぎる or 見つからない  
・英語/専門用語が多く、理解に時間がかかる  

## ストーリーボード
手早くAWSの学習を行いたいが、やり方を巡って困っている。  
AWS公式ドキュメントは網羅性が高いため、これのドキュメントを辞書のように用いながら学習したいものの、英語で書かれている他、分量が多く、とても自力では学習が捗らない。  

## アプリの概要
### 解決したい課題
・長文ドキュメントから「必要な部分」を探すのに時間がかかる  
・理解に必要な前提/用語が多く、学習コストが高い  
・調べた内容が分散し、同じ調査を繰り返してしまう  

### 提案する解決策：テキストチャットボット
・自然文で質問 → 要点を整理して回答  
・外部情報を参照して回答品質を安定化  

## システム構成
![System Architecture](System%20Architecture.png)

## アプリ構成、フォルダ構成
### 主な機能
・ユーザー登録 / ログイン / ログアウト（セッションベース）  
・テキストチャット（生成AI）  
・会話履歴のDB保存  
・RAG：ナレッジを検索して回答に反映（ベクトル検索）  
・外部情報をPDF形式でアップロード可能且つアップロードされたPDFの内容をベクトル化してDB保存。  
・チャットログの表示。  

```
.
├── AGENTS.md
├── Dockerfile
├── LICENSE
├── README.md
├── backend
│ ├── main.py
│ └── requirements.txt
├── docker-compose.yml
├── extracted_text.txt
└── frontend
├── app.js
├── assets
│ └── README.md
├── auth.html
├── index.html
├── pcm-worklet.js
└── room.css
```

## デモ動画
[![Demo Video](https://img.youtube.com/vi/AFXqn3MlxAA/0.jpg)](https://www.youtube.com/watch?v=AFXqn3MlxAA)

## DB構成（外部、トランザクション、ER図）

### users
- PK id (text)
- pw_hash
- display_name
- created_at

### rag_docs
- PK id (bigint)
- FK user_id
- created_at

### rag_chunks
- PK id (bigint)
- idx
- content
- metadata (jsonb)
- embedding (vector(768))
- user_id
- doc_id

### chat_logs
- FK user_id



## ベクトルRAG
アップロードされたPDFをベクトル化し、PDFの内容をベクトル化したデータとchunk化したテキストデータをDBに保存し、テキストチャット時に関連情報を検索してプロンプトへ注入します。  
現在は **pgvector によるSQL検索（`RAG_VECTOR_MODE=sql`）** が既定です。  
DB は pgvector 拡張が必要です（`docker-compose.yml` は pgvector 搭載イメージを使用）。


## チーム開発方法
GithubのKanbanボードを使い、タスクを設定し、進めた  

ディスコードで通話及びチャットでやり取り  

アプリはCODEX CLI(OpenAI)で作成






### 主な環境変数
- `RAG_TEXTS_PATH`（必須: vectors と同件数のテキスト）
- `RAG_EMBED_MODEL`（既定: `text-embedding-004`）
- `RAG_FORCE_RELOAD`（`1` で再取り込み）
- `RAG_VECTOR_MODE`（`sql`|`memory`）
- `RAG_VECTOR_DIM`（埋め込み次元を明示する場合）
- `RAG_VECTOR_INDEX`（`hnsw`|`ivfflat`|`none`）
- `RAG_CACHE_IN_MEMORY`（SQLモード時は `0` 推奨）
