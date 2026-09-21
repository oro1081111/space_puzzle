# PV2 三人 Policy/Value 試作：未通過發布門檻

本次是新流程的初期實作與兩階段試訓練，**不是已達成高強度的 ATLAS-R 新版**。沒有修改 `grid-clash`，沒有把候選模型放入網站。正式網頁保留 Trio v1.1／其他模式原模型。訓練程式與負結果可以提交；模型上線必須另通過驗收。

## 已實作

- 15、30 階均實際蒐集三人完整對局，不再只將 15 階輸入放大。
- 14 channel 分開編碼自身／兩位對手領地與棋子，加上空白、禁格、回合比例、無得分比例、尺寸、座標及尾盤歷史長度。搜尋保留完整規則歷史，但神經網路只讀歷史長度，仍是簡化觀測。
- 120,298 參數的小型 residual CNN，輸出方向 Policy、三位玩家第一名份額 Value、最終已佔領地比例。
- 終局並列第一平均分配勝者份額；不是將第二名當成勝利。
- 旋轉／鏡射同步轉換動作、合法遮罩；交換兩個敵人時同步轉換 territory/head channel 與 Value 標籤。
- 多回合同時動作 rollout：自身各合法方向，配四組敵方樣本，共用隨機數比較自身選項；敵方採 80% policy＋20% 合法均勻探索。三回合後由 Value 評分，終局使用真實勝者份額。
- 搜尋在私有棋盤上運作，各方決策前都只讀同一份公開狀態，不偷看對手已選動作。沒有 MCTS、Nash 求解或完整列舉未來所有聯合行動，不能這樣宣稱。
- 搜尋動作分布回流訓練 Policy，真實終局回流訓練 Value。小型新 buffer 與教師 replay 等比例按資料集抽樣，避免資料量差距讓新 buffer 被淹沒。
- 按完整對局切分 train/validation（game id mod 5），不把同一局切成訓練及驗證；15／30 階第二輪各自選 validation checkpoint，但它們共享混合尺寸訓練架構。
- 保留 seed、來源雜湊、checkpoint、每局結果與 validation 指標；模型及資料只在忽略版控的 `training/runs/`。

## 實際執行

| 階段 | 15 階 | 30 階 |
|---|---:|---:|
| 三種舊策略教師對局 | 120 局／5,760 筆 | 120 局／5,760 筆 |
| 搜尋回流對局 | 9 局／432 筆 | 18 局／864 筆 |

共 267 局、12,816 筆玩家視角局面，含按局保留的 validation 資料，不能全數稱為 training samples。每局最多取 16 個時間點×三位玩家，並非所有經過的局面。

- Warm-up：兩種尺寸混合，6 epochs，AdamW lr=.0003，batch=64，109.5 秒。
- 搜尋回流後微調：6 epochs，每 epoch 120 batches，lr=.0001，86.9 秒。
- CPU 單 native／PyTorch 執行緒、BelowNormal；曾同時跑兩個低優先權評估／資料程序，不是全程只用一個 CPU core。
- 冷啟動 15 階 self-play 曾出現 934 次行動才結束的對局，因此在第 9 局完整結束後以 PAUSE 停止原定 18 局批次，保留已完成資料。30 階改用一位候選＋兩位穩定舊策略，最大 396 次行動。規則終止不等同對局品質良好。
- 初步 Python 測速，四組樣本：三回合搜尋約 15 階 32 ms／30 階 101 ms 每次決策。不是瀏覽器或全局平均延遲承諾。

## 對戰結果：開發篩選，非最終獨立測試

固定正常開局 seed=840001；APEX／BASTION／SWEEP 的九種有序配對（含同策略），各輪換三席，共 27 場。所有下列模型比較使用相同 seed 與組合。只是一個種子的開發篩選，不能宣稱普遍勝率，沒有通過者就不消耗最終留出測試集。

| 模型／決策 | 15 階單獨第一 | 30 階單獨第一 |
|---|---:|---:|
| 現行網頁控制器 | 13／27（另並列 1） | 1／27 |
| Warm-up 純 Policy | 5／27（另並列 1） | 8／27 |
| Warm-up＋三回合 Value 搜尋 | 1／27 | 3／27 |
| 回流微調後純 Policy | 4／27 | 5／27 |

回流微調後三回合搜尋以預先設定的 70% 正常開局最低門檻提前淘汰：27 場至少需要 19 勝；累積 9 場非單獨第一後，即使剩餘全勝仍不可能達標。未測場次另列，不能用部分場次計算整體勝率。這只是最低篩選，通過仍需多種子、最弱座位／組合與瀏覽器驗收；不是自動核准發布。

第二輪搜尋：兩種尺寸各已測 **11 場、2 勝、9 敗**，各 **16 場未執行**，沒有未終止對局。即使剩餘全勝也只有 18／27，無法達到至少 19 勝，因此均淘汰。這不是「2／27 勝」，也不把 2／11 當成完整矩陣的勝率。

Checkpoint SHA256（檔案雜湊）：

- Warm-up `best.pt`：`011434769905a30da32946fbb3a15fe40f22d38a4113c818c951a7be82b49c72`。
- 微調 `best-15.pt`（第 4 epoch）：`aeb95ad38d42514b66f4901e199d3a4cbbfeebbb19ddf55d8b783318798cd142`。
- 微調 `best-30.pt`（第 5 epoch）：`baffbf82eb21d5e35524bd07d7c69c8c927e417cc7809ec26ce8376b66b5ba77`。

## Value 診斷

Warm-up 在未參與訓練的教師對局，以棋盤已佔領比例分段，Brier error 越低越好：

| 尺寸／階段 | 樣本數 | 模型 | 三人均勻 1/3 基準 |
|---|---:|---:|---:|
| 15 開局（<20%） | 261 | .6718 | .6667 |
| 15 中盤（20%～70%） | 426 | .5826 | .6667 |
| 15 收官（≥70%） | 465 | .2381 | .6667 |
| 30 開局 | 270 | .6818 | .6667 |
| 30 中盤 | 330 | .5937 | .6667 |
| 30 收官 | 552 | .2692 | .6667 |

平均 validation loss 下降不能證明開局變強：模型主要在收官更會猜勝者，開局甚至未超過均勻基準。加入 Value 搜尋後實際成績退步是已觀察事實；搜尋分布轉移、對手預測誤差、開局 Value 不可靠與長期路線不足是後續要分開驗證的假說，不能宣稱已證明單一根因。

第二輪搜尋回流也未讓純 Policy 勝場提升。本輪沒有足夠證據支持繼續擴大同一設定，更不能降低發布標準來宣稱成功。

## 驗證／發布界線

- 13 項 Python 回歸通過，包含分開玩家編碼、兩尺寸、對稱增強、私有棋盤、合法動作、並列第一、Value 確實影響動作以及提前淘汰算術。
- 既有 160 筆搜尋 fixture 及 JS 碰撞／圈地／尾盤回歸通過；30 階 13,548 項 Python／JS 規則比對通過。所有新資料對局也逐步比對。
- Warm-up ONNX 的 Policy／Value／領地三個輸出，15／30 各 12 個局面×三視角與 PyTorch 數值一致。非整除 adaptive pooling 原匯出失敗，已用完全相同分箱的 slice mean 修正並驗證。
- ONNX 檔只放本機 `pv2-local-export15-v2`／`pv2-local-export30-v2`，`release_approved=false`。沒有新增網站 worker 的 14-channel 輸入或 JS 多回合搜尋；強度未通過，瀏覽器整合延後，不能說這個候選已能在網頁玩。
- **本輪未完成「強 AI 上線」目標。**不覆蓋正式模型，也不把框架完成說成強度完成。

## 重現入口

```powershell
.venv\Scripts\python.exe -m unittest training.test_pv training.test_learning training.test_three_player
.venv\Scripts\python.exe -m training.pv_run collect --size 15 --games 120 --seed 810001 --output training/runs/pv2-teacher15-repro
.venv\Scripts\python.exe -m training.pv_run collect --size 30 --games 120 --seed 820001 --output training/runs/pv2-teacher30-repro
.venv\Scripts\python.exe -m training.pv_run train --data training/runs/pv2-teacher15-repro/data.npz training/runs/pv2-teacher30-repro/data.npz --epochs 6 --seed 830001 --output training/runs/pv2-warmup-repro
.venv\Scripts\python.exe -m training.pv_run evaluate --size 15 --games 1 --seed 840001 --model training/runs/pv2-warmup-repro/best.pt --depth 0 --output training/runs/pv2-policy-repro
.venv\Scripts\python.exe -m training.pv_run evaluate --size 15 --games 1 --seed 840001 --model training/runs/pv2-warmup-repro/best.pt --depth 3 --samples 4 --output training/runs/pv2-search-repro
```

所有輸出目錄必須尚不存在。`--size 30` 可測 30 階；省略 `--model` 評估現行網頁控制器。`collect --model ... --league` 混合 self-play 與舊策略；`--legacy-only` 使用兩個固定舊策略。`train --batches 120` 按資料集等比例 replay；`evaluate --gate-stop` 只在無法達到固定門檻時提前結束，summary 明列 `planned`／`not_run`。`export` 只做本機匯出與數值檢查，不部署。

本次搜尋資料分別使用 seed=850001（15 階，`--league`，完成 9 局後停止）及 seed=860001（30 階，`--legacy-only`，18 局）；微調從 warm-up 開始，四份資料等比例抽樣，seed=870001、`--epochs 6 --batches 120 --lr 0.0001`。第二輪評估維持 seed=840001、depth=3、samples=4，加入 `--gate-stop`。

## 後續工作，不是已完成事項

先針對開局與偏離教師路線的局面補反事實動作／實際終局資料，檢查 Value 是否能正確排序動作，而不只是猜收官贏家。再比較保守短搜尋、長搜尋及不同對手模型；只有搜尋確實帶來提升，才擴大聯盟自對弈。較長訓練或 GPU 費用需另確定，不保證單純延長時間能達到 70%。
