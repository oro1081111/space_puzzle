# ATLAS-R Trio v1.0：15×15 三人專用初版

## 結論

2026-09-21，沿用目前雙人版網路微調三人 PPO。只替換 **15×15、剛好 3 人** 的模型；雙人、30×30、4～8 人權重不變。仍是瀏覽器本機推論，不需 Python。

獨立測試有小幅第一名率改善，但**尚不能宣稱統計上確定更強、全面勝過原 AI 或達到高強度**。平均分差未改善，部分對手組合退步。這是三人專用訓練的初始版本。

## 訓練

- 起點：`training/runs/redesign15/selected_v2.zip`，凍結雙人 actor/critic 暖啟動。
- 7 channel：自身領地、所有敵方領地、空白、自身棋子、所有敵方棋子、禁格、剩餘回合比例。修正 Python 觀察原本只適用兩人的敵方索引。
- MaskablePPO，CNN 不改尺寸，CPU 每程序單執行緒、BelowNormal；訓練只有一個環境。
- 32,768 個學習決策、411 個完成對局，202.1 秒；三席隨機輪替（122／140／149 局）。
- 對手組合：APEX+BASTION、BASTION+SWEEP、SWEEP+APEX 各 2/9；兩個凍結 ATLAS 共 1/3。出現舊檢查點後，一半 ATLAS 組合改抽凍結檢查點；同一局不更新敵方權重。
- 實際敵方由網頁控制器決策；學習者執行 PPO 自己抽出的合法動作，避免以搜尋後動作冒充 on-policy 樣本。部署評估才使用完整神經網路＋搜尋＋脫困控制器。
- Reward：單獨第一 +1、並列第一 0、其餘 -1；輔助 `.1 × 相對最高對手分差變化 / 225`。不是 MAPPO；沒有新增集中式 critic。
- learning rate 3e-5、gamma .999、GAE .98、rollout 256、batch 128、epochs 4、entropy .01、target KL .02。
- 開局前綴隨機抽 0／4／8 次動作；規則與網頁逐步比對。

## 選模與獨立評估

每組包含三席輪替與四種對手組合。所有評估使用 4 次空白優先隨機動作前綴，**不是正式固定出生局面勝率**。勝指單獨第一；並列第一另計。

| 資料 | 舊版勝／並列／敗 | 16,384 步候選 | 32,768 步候選 |
|---|---:|---:|---:|
| 開發集，36 場 | 9／0／27 | 9／2／25 | 6／0／30 |
| 獨立第一批，120 場 | 42／1／77 | 46／0／74 | 不再測試 |
| 獨立確認批，360 場 | 85／4／271 | 92／6／262 | 不再測試 |

根據開發集先選定 **16,384 步**，再測兩批不同種子的獨立資料。未依獨立集重挑檢查點。最後模型退步，所以不部署最後檔案。

兩批獨立資料合計，新舊各 480 場：

| 對手組合（各 120 場） | 舊版單獨第一 | Trio 單獨第一 |
|---|---:|---:|
| APEX＋BASTION | 28 | 25 |
| BASTION＋SWEEP | 29 | 30 |
| SWEEP＋APEX | 31 | 34 |
| 舊 ATLAS＋舊 ATLAS | 39 | 49 |
| 合計 | 127（26.5%） | 138（28.8%） |

- 並列第一：舊版 5，新版 6。沒有未結束對局。
- 平均「自身減最高敵方」分差：-12.581 → -12.594，沒有改善。
- 最大連續無得分嘗試：18 → 39；每場此最大值的平均：6.23 → 6.63，稍微增加。
- 終局：舊版填滿 455／末格互搶 23／尾盤週期 2；新版 456／21／3。既有僵局規則未修改。
- 這些是配對、共享開局的測試，並非 480 個完全獨立隨機樣本；不得把 +2.3 百分點宣稱為顯著優勢。弱點仍是三人對手辨識、座位公平性與對 APEX+BASTION 的穩定性。

## 發布與驗證

- 名稱：ATLAS-R Trio v1.0。網頁選 ATLAS-R 即可；只在 15 階三人時自動載入 `atlas-r-3p-v1.onnx`。
- checkpoint SHA256：`64db1fd8ea4f5fefd5c99ff9f7a13e5377728714f9d8ebce34c1b62b5410227b`。
- ONNX SHA256：`e494d5bc2a781ed9673fd35ecd25813aa7f798e2d062b02a326ecc4c268f5097`，547,340 bytes。
- 原始訓練前規則驗證：6,395 項；7 項 Python 測試、既有 160 筆雙人搜尋 fixture、碰撞／圈地／尾盤回歸均通過。
- 三人 ONNX/PyTorch 數值比對通過；瀏覽器 WASM 抽查、三人切回雙人的快取隔離通過。
- Edge 真實三人新模型全自對弈：190 回合／194 次嘗試，填滿 225 格。雙人 15／30 階完整對局仍通過，2～8 人兩種大小首回合及人機、暫停、過期結果、離線錯誤保護通過。
- 此次沒有進行四人以上專用訓練，也沒有 30 階專用訓練。

## 重現

在儲存庫根目錄使用現有 `.venv`。輸出資料夾必須尚不存在，避免混入舊資料。

```powershell
.venv\Scripts\python.exe -m training.verify --size 15
.venv\Scripts\python.exe -m unittest training.test_three_player training.test_learning
.venv\Scripts\python.exe -m training.three_player train --run training/runs/atlas3-round1 --steps 32768 --seed 310001
.venv\Scripts\python.exe -m training.three_player evaluate --model training/runs/atlas3-round1/snapshot_000016384.zip --output training/runs/atlas3-mid-dev --seeds 3 --seed 330001
.venv\Scripts\python.exe -m training.three_player evaluate --model training/runs/atlas3-round1/snapshot_000016384.zip --output training/runs/atlas3-mid-heldout --seeds 10 --seed 340001
.venv\Scripts\python.exe -m training.three_player evaluate --model training/runs/atlas3-round1/snapshot_000016384.zip --output training/runs/atlas3-mid-confirm --seeds 30 --seed 350001
.venv\Scripts\python.exe -m training.export_three_player training/runs/atlas3-round1/snapshot_000016384.zip
```

省略 `--model` 就評估凍結基準；同一批使用相同 `--seeds` 和 `--seed`。各次 run 保留 manifest、SHA256、每局分數／完整動作序列及 summary；訓練保留 checkpoint、PPO 指標與每局記錄。這些大檔位於忽略版控的 `training/runs/`。

## 下一輪建議

暫不擴大尺寸／人數，也不直接加倍 PPO 步數。先補固定開局與更多種子評估，針對 APEX+BASTION 及低勝率座位蒐集失敗局面；再評估把兩個對手分開編碼、座位／棋盤對稱增強。任何表示法變更要同步 Python 與瀏覽器並重新訓練，不能只改輸入 channel。這些是後續計畫，本輪未實作。
