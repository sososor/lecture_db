# AGENTS.md

## 目的
データベースの整合性を強化するため、外部キー制約を導入し、全ての DB 操作を明示的なトランザクションで実行する（autocommit を無効化）。

## 現状（要点）
- `ConnectionPool` が `autocommit=True` のため、各クエリが即時コミットされる。
- `chat_logs.user_id` は `users.id` への参照制約が無い。
- 複数クエリで構成される処理でも、ロールバック境界が明示されていない。

## ゴール
- `chat_logs.user_id` に FOREIGN KEY を追加し、ユーザー削除時の扱いを明確化。
- autocommit を停止し、各 DB 操作を `with con.transaction():` で実行する。
- 失敗時はロールバックされ、部分的な更新が残らない。

## 変更対象
- `backend/main.py`（DB 初期化・CRUD・RAG ingest など）
- 必要なら `README.md`（運用上の注意点を追記）

## 実装方針
### 1) autocommit 停止 + 共通トランザクションヘルパ
- `ConnectionPool(..., kwargs={"autocommit": True})` を削除するか `False` に変更。
- トランザクション管理を統一するため、共通ヘルパを用意する。
  - 例: `_db()` / `_with_tx()` を定義し、`with pool.connection() as con: with con.transaction():` を必ず通す。
- DB への全アクセスはこのヘルパ経由で行う。

### 2) FOREIGN KEY 追加（idempotent に）
- `chat_logs` 作成後に FK を追加する。
- Postgres には `ADD CONSTRAINT IF NOT EXISTS` が無いので、`DO $$ BEGIN ... END $$;` で存在確認を行う。
- 推奨例:
  - `FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE`
  - ログを残したい場合は `ON DELETE RESTRICT` / `NO ACTION` に変更。

### 3) 既存データの整合
- 既存データに不整合があると FK 追加で失敗するため、事前に整理する。
  - 例: `DELETE FROM chat_logs WHERE user_id NOT IN (SELECT id FROM users);`
- 段階適用したい場合は `NOT VALID` で追加 → データ整理 → `VALIDATE CONSTRAINT`。

### 4) トランザクション化する処理の範囲
- 1回の処理で複数クエリが走る箇所は **同一 connection + 同一 transaction** にまとめる。
  - `_init_db`（DDL + index + RAG schema）
  - `append_log`, `read_tail`, `_purge_chat_logs`
  - 認証系（register/login）での SELECT/INSERT
  - RAG ingest / ロード（`_ingest_rag_vectors`, `_load_rag_cache_from_db`）
- 既存の「事前チェック→INSERT」の流れは同一トランザクション内で実行し、競合時は UNIQUE 制約のエラー処理を追加する。

## 動作確認
- 起動後に `\d chat_logs` で FK が付与されていること。
- 例外発生時に partial insert が残らないこと（rollback を確認）。
- 登録/ログ保存/RAG ingest が従来通り動作すること。
