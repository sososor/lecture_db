# AGENTS.md

## 目的
このリポジトリ（/app = web-bot）から **3DルームUI / アバター表示 / Webカメラの顔トラッキング** 機能だけを削除し、
音声・テキストチャットのUIは通常の2Dページとして残す。

## 変更範囲（必須）
### 1) 3DルームUI / アバター表示の除去
- `frontend/index.html` から `room.js` の読み込みを削除する。
- `frontend/room.js` は不要なので削除する（または読み込まない前提で空にする）。
- 3D用DOM（`#stage`, `#ui3d`）を廃止し、UIは通常のHTML構造にする。

### 2) Webカメラ顔トラッキングの除去
- `frontend/index.html` から以下を削除する。
  - `<video id="webcam">`
  - `@tensorflow/tfjs` と `face-landmarks-detection` の `<script>`
- `getUserMedia` を呼ぶ顔トラッキング処理が残らないことを確認する。
  - マイク用 `getUserMedia({ audio: true })` は残す。

### 3) CSSの整理
- `frontend/room.css` は3D専用のため、
  - 削除して新しいUI用CSSを用意する、または
  - 2D UIのためのスタイルに置き換える。
- `#stage`, `#webcam`, `canvas`, `#ui3d` など3D専用スタイルは削除する。

### 4) 付随ドキュメントの更新
- `frontend/assets/README.md` にあるVRM/アバター案内は削除またはファイル削除。
- READMEに3D/アバター/顔トラッキングの記述があれば更新する。

## 影響確認（最低限）
- ページ読み込み時に **カメラ許可ダイアログが出ない**。
- コンソールに `room.js` 未ロードや `THREE` 参照エラーが出ない。
- UI（接続/切断/ミュート、ログ、チャット入力）が画面内で操作できる。
- 音声/テキスト機能（`frontend/app.js`）が動作する。

## 触らない領域
- バックエンド（`/app/backend`）のAPI実装は変更しない。
- 音声・テキストチャットのロジックは維持する。
