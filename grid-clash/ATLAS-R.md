# ATLAS-R｜全域競速 v1.0

目前網頁已升級為 [v1.1 循環脫困修正版](ATLAS-R-v1.1.md)，沿用以下權重。下文保留 v1.0 模型與歷史評估記錄。

網頁版本：15×15／30×30 可選，ATLAS-R 限雙人。30×30 沿用相同權重，是尚未完成棋力評估的實驗性模式，不套用 15×15 的勝率。

模型是既有 selected_v2 權重的 actor，並非重新訓練；配合一回合同時動作搜尋（最多 16 分支），評分為 0.75 期望空間優勢＋0.25 最差回應＋0.5 log policy prior。搜尋不讀取玩家已輸入的方向。

兩份 ONNX 的權重相同，只是輸入尺寸不同。各模型來源及 SHA-256 記錄在同名 JSON。模型在 Web Worker 中使用固定版 ONNX Runtime Web 1.22.0、單執行緒 WASM 推論；執行庫由 jsDelivr 下載，模型由本站提供。不需要 Python 後端。載入失敗會暫停並顯示錯誤，不會暗中換成其他 AI。

棋盤／人數變更會重設對局。暫停、重開、切換策略會丟棄未完成的推論，不讓舊動作落入新局。3 人以上 ATLAS 選項停用，原本三種策略仍可使用。

## 驗證與重製

- `node tests/collision.test.cjs`：實際網頁圈地、搶格取消、暫時禁止格。
- `node tests/atlas.test.cjs`：15／30 棋盤共 160 個 Python 局面與動作比對、搜尋無副作用。
- `node tests/browser.test.cjs`：需 Playwright 與 Edge；實際 WASM logits 比對、雙尺寸單步、玩家操作、取消與下載失敗。
- `python -m training.verify --size 15` / `--size 30`：Python 與實際網頁規則差分。
- `python -m training.export_web --size 15` / `--size 30`：需原始 `training/runs/redesign15/selected_v2.zip`（本機訓練輸出不納入 Git）、PyTorch、onnx、onnxruntime 及 training requirements。網頁遊玩不需要這些套件。

15×15 歷史評估：固定開局對三種原 AI 共 300/300 勝；隨機前四步開局 115/120 勝。這些結果不代表無敵，也不代表對人類或 30×30 的勝率。詳見 `training/DAGGER_15x15.md`。
