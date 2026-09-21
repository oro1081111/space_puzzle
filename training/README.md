# Grid Clash 訓練工作區

第三輪新增受控補訓、無副作用老師查詢與一回合同時動作搜尋；完整正／負結果見 [DAGGER_15x15.md](DAGGER_15x15.md)。本輪補訓未勝過 v2；較強的候選是 **v2 權重＋空間搜尋**，不是新的 DAgger 權重。

```powershell
.\.venv\Scripts\python.exe -m unittest training.test_learning
.\.venv\Scripts\python.exe -m training.compare training/runs/redesign15/selected_v2.zip --opponents apex bastion sweep --seeds 50 --seed-start 15000 --spread-seeds --search-depth 1 --output training/runs/search_reproduction
```

`--search-depth 1` 使用凍結 policy 輔助的一回合同時動作搜尋；加 `--no-policy-prior` 是不需模型推論的空間搜尋控制組。`--opening-steps 4` 是另計的隨機前綴局面測試，不能混入正式固定開局結果。`--spread-seeds` 產生固定、分散的 seed 清單並存入 manifest。評估與部署不同，網頁 AI 尚未更換。

第二版設計與結果見 [REDESIGN_15x15.md](REDESIGN_15x15.md)。第一版紀錄保留於 [ASSESSMENT_15x15.md](ASSESSMENT_15x15.md)，不代表目前候選模型實力。

## 第二版重現指令

以下在儲存庫根目錄依序執行，不要同時啟動重型工作。所有模型只留本機。

```powershell
.\.venv\Scripts\python.exe -m unittest training.test_learning
.\.venv\Scripts\python.exe -m training.imitate collect --episodes 300 --seed 100000
.\.venv\Scripts\python.exe -m training.imitate train --epochs 12 --seed 42
.\.venv\Scripts\python.exe -m training.evaluate training/runs/imitate15/bc_best.zip --opponents greedy apex bastion sweep --seeds 5 --seed-start 9400 --quiet --output training/runs/imitate15/development.json
.\.venv\Scripts\python.exe -m training.train --size 15 --steps 100352 --envs 2 --threads 1 --subprocess --expert --learning-rate 0.00005 --resume training/runs/imitate15/bc_best.zip --run training/runs/expert15
```

`--expert` 使用真正網頁 AI，每場均勻抽取 APEX、BASTION、SWEEP、空白優先。`--learning-rate` 可覆蓋載入模型的學習率。`metrics/progress.csv` 保留 PPO 診斷。模仿模型 `bc_best.zip` 是驗證 cross entropy 最低者，不等同對戰最強；仍需另做評估與選模。

受控示範重放與序列評估：

```powershell
.\.venv\Scripts\python.exe -m training.train --size 15 --steps 32768 --envs 2 --threads 1 --subprocess --expert --learning-rate 0.00002 --replay-data training/runs/teacher15.npz --resume training/runs/expert15/snapshot_000016384.zip --run training/runs/replay15
.\.venv\Scripts\python.exe -m training.compare training/runs/imitate15/bc_best.zip training/runs/expert15/latest.zip --seeds 5 --seed-start 9400 --output training/runs/comparison_example
```

重放是 PPO 後額外做模仿更新，非純 PPO；回合上限與遊戲獎勵不變。示範陣列不放進模型檔案，只有續訓重放時才需要原始資料。一般 MaskablePPO 載入仍可推論。`compare` 每個模型／對手完成即存檔；重跑只復用雜湊與種子一致的報告，避免不同權重混進同一份比較。

## 第一版與共同設定

此目錄執行本機 CPU 訓練；不發布網頁，不改動線上 AI。依最新指示，先縮為 **15×15**，人數範圍仍限定 **2～4 人**。目前網路與訓練環境仍是雙人版，先完成雙人基準，再擴充三、四人身分 channel 與勝負獎勵；不開發 5～8 人訓練。正式網頁仍維持 30×30。

15×15 使用獨立 run `pilot15`，重新訓練小型網路，不直接套用 30×30 權重。舊模型與已完成評估保留在 `pilot`；舊版完整評估已依使用者改方向而中止，不能宣稱完成 30×30 實力判定。

筆電初始配置：Core Ultra 5 225H（14 核心）、約 32 GB RAM。使用 2 個模擬程序、1 條 PyTorch 運算執行緒、native math 每程序 1 條執行緒、Windows Below Normal 優先權。訓練與評估依序進行，不同時跑。風扇聲不能當成溫度測量；系統未提供可讀的 CPU 溫度。此配置優先保持筆電可操作性，並非效能最佳化結論。

## 規則與一致性

規則版本：`exclusive-reach-cancel-round-v1`。

- 同步收集動作；多人搶格取消整回合，新增禁止格，成功結算後才清除。
- 禁止格影響當次移動，不影響永久可達範圍與圈地。
- 空白格只有一個玩家可達時填色；不覆蓋既有領地。
- 無法擴張的電腦停止移動。棋盤填滿、全員無法擴張、或超過格數 × 8 回合後結束（15×15 為第 1801 回合、30×30 為第 7201 回合）。
- 訓練採對稱的純 AI 回合規則；不模擬真人點擊牆壁時的 UI 拒絕事件。
- 搶格取消仍是一個訓練狀態轉換，但不增加遊戲回合。

`oracle.cjs` 直接載入目前 HTML 的規則與原版 AI，不重寫 JavaScript 遊戲邏輯；測試 adapter 只把初始化固定尺寸換成指定尺寸。15×15 的出生位置沿用相同比例公式，核心規則不變。`engine.py` 使用 SciPy 的四向連通區標記實作 Python 可達範圍。

## 安裝與驗證

在儲存庫根目錄執行以下 PowerShell 命令。現有 `.venv` 已安装套件，無需重建。

```powershell
.\.venv\Scripts\python.exe -m pip install -r training/requirements.txt
node tests/collision.test.cjs
.\.venv\Scripts\python.exe -m training.verify --size 15
.\.venv\Scripts\python.exe -c "from pettingzoo.test import parallel_api_test; from training.env import ParallelGame; parallel_api_test(ParallelGame(size=15),num_cycles=100)"
```

驗證包含 2～4 人逐回合差分、完整對局、圈地、碰撞與上限案例。報告在 `training/runs/verification15.json`（30×30 為 `verification.json`）。訓練入口檢查驗證通過且 HTML SHA-256 一致；更動 Python 規則後也必須重新驗證。

## 第一輪訓練

```powershell
.\.venv\Scripts\python.exe -m training.train --size 15 --steps 100352 --envs 2 --threads 1 --subprocess --run training/runs/pilot15
```

- 7 個 channel：己方領地、敵方領地、空白、己方棋子、敵方棋子、暫時禁止格、剩餘回合比例。
- 小型 CNN（15×15 約 9.7 萬參數；30×30 約 29 萬），四個方向的 Policy 與 Value。
- MaskablePPO：256 steps/env、batch 256、4 epochs、learning rate 0.0003、gamma 0.999、GAE lambda 0.98、entropy 0.02。
- 終局 +1/-1/0；輔助獎勵為 0.1 × 領地差變化 / 格數。
- 隨機交換座位。無合法選擇時由環境推進強制等待，直到能決策或結束。
- 起初 90% 空白優先對手、10% 隨機對手。有模型快照後變為 50% 凍結快照、40% 空白優先、10% 隨機。
- 每 16384 個 learner transition 存不可變快照。每局對手固定；不是邊打邊更新。
- Gym wrapper 的一次 learner transition 可能包含額外強制等待；訓練步數不等同遊戲回合或所有玩家的動作總數。
- 格數 × 12 次引擎決策的訓練安全上限記為 truncation、使用 value bootstrap，不判輸贏；自然回合終局仍按正式分數判勝負。

此輪尚未將三種昂貴的網頁 heuristic 加入訓練抽樣；它們已接入正式 JavaScript 對戰評估，後續依瓶頸量測決定訓練接法。尚無 ONNX 匯出、MAPPO、搜尋或多人訓練。

## 評估

```powershell
.\.venv\Scripts\python.exe -m training.evaluate training/runs/pilot15/latest.zip --opponents random greedy --seeds 5 --output training/runs/pilot15/evaluation-simple.json
.\.venv\Scripts\python.exe -m training.evaluate training/runs/pilot15/latest.zip --opponents apex bastion sweep --seeds 2 --output training/runs/pilot15/evaluation-heuristic.json
```

每個 seed 打兩個座位。預設採隨機 policy 抽樣，可加 `--deterministic` 另外測確定性選步。評估回合使用正式規則；若超過額外 20000 決策安全上限，只報 unfinished、不算勝利。預設 seed 9000 起是開發評估集，不是最後的獨立驗收集。小樣本不能認證實力。

正式強度驗收仍需每種 heuristic 至少 500 局、不同訓練 seed、未參與選模型的測試集與信賴區間。

## 續跑與產物

```powershell
.\.venv\Scripts\python.exe -m training.train --size 15 --resume training/runs/pilot15/latest.zip --steps 1000000 --envs 2 --threads 1 --subprocess --run training/runs/pilot15
```

`--steps` 是本次增加的訓練量，會向上取整到完整 rollout。可恢復權重、優化器與步數，但不保證恢復完全相同的環境/RNG 狀態。CLI 正常退出會保存 latest；突然強制終止時使用上一個完整 snapshot。

需要暫停時，在該 run 目錄建立名為 `PAUSE` 的檔案；訓練在下一個 callback 停止並保存 `latest.zip`，報告標示 paused。模型更新期间需等該次更新完成。再次明確授權續跑後才移除 PAUSE 檔案。`config-history.jsonl` 保留舊執行設定。

產物在 `training/runs/<名稱>/`，已排除 Git：

- `config.json`：參數、依賴版本、規則版本、HTML 雜湊。
- `progress.json`、`episodes.jsonl`：進度、對手種類、分數、碰撞、終局紀錄。
- `untrained.zip`、`snapshot_*.zip`、`latest.zip`：未訓練基準、凍結對手與可續跑模型。
- `training_report.json`：完成步數、時間、參數量。
- `evaluation-*.json`：逐局與彙總成績。

不要用平均訓練 reward 宣稱變強。比較固定開發種子的對戰結果，再決定延長、修正獎勵或加入較強對手。
