# ATLAS-R Trio v1.1：三人決策修正

## 結論與範圍

只修改 **15×15、3 人**。保留 Trio v1.0 神經網路權重，修正決策控制器；不是再次 PPO 訓練，也不是新模型的學習成果。雙人、30×30、4～8 人的決策分支不變。

正常開局 270 場：單獨第一從 **32（11.9%）提高到 119（44.1%）**。九種舊策略組合與三個座位的分組勝場皆改善。隨機前綴局面 135 場：**33（24.4%）提高到 44（32.6%）**。所有測試對局結束，但不代表穩贏或對所有局面都變強。

## 為什麼不先增加訓練量

v1.0 的上線判準偏重混合對手平均結果，缺少正常開局驗證。它訓練直接走方向，上線卻使用另一層搜尋決策。先修控制器並保持權重不變，可以隔離改善來源，不用把效果誤歸因於 PPO。

程式查證與消融測試指出兩項可直接處理的限制：

1. 三人局兩位敵人各最多四種合法動作，至多 16 組；舊版用 16 次分層抽樣，會重複、也可能漏掉組合。抽樣權重還假設舊策略會照 ATLAS 的策略分布行動，沒有經過敵方行為預測驗證。
2. 原控制器主要比較一回合後的距離估分，沒有使用舊策略已有的多步圈地路線。模型稍微改變方向機率，不足以補上持續規劃的缺口。

正常開局開發測試，seed=71，六種異策略有序組合×三席，共 18 場：

| 控制器 | 單獨第一 |
|---|---:|
| 原版 | 2 |
| 完整列舉＋均勻敵方權重 | 4 |
| 上項但移除自身神經網路偏好 | 2 |
| 上項改成只估自身領地 | 2 |
| 完整列舉＋多步圈地＋保留神經網路偏好 | 8 |
| 上項移除神經網路偏好 | 8 |

選定「完整列舉＋多步圈地＋保留自身偏好」後，才跑下列不同種子的驗證；沒有依驗證結果再調參數。

## 實作選擇

- **反應式搜尋完整列舉**：每個自身合法動作檢查所有敵方聯合動作，不漏列零模型機率但仍合法的走法。無法行動的敵人以 `null` 表示。每組等權重；這不是宣稱敵人真的均勻隨機，也不偷看實際待執行動作。
- **多步圈地優先**：重用 `findEnclosurePlan`，既有預設深度 22、beam 60。保持計畫，逐步執行；沒有計畫時才使用神經網路輔助反應式搜尋與既有脫困。
- **計畫可中斷**：下一步必須合法而且是空白；被禁格或敵方領地擋住即清除。既有碰撞取消會清空路線，大片圈地得分也會觸發清除。整局結束與尾盤循環規則不改。
- **不擴大架構**：沒有新增套件、MCTS、中央 critic 或新輸入 channel。多步圈地是現有幾何／領地規劃，不是完整模擬所有敵人未來 22 步，仍可能遇到敵方中途封路。
- **維持推論範圍**：只有 `n===15 && a.length===3` 進新分支。Trio v1.0 ONNX 檔案與 SHA256 不變；版本 v1.1 指控制器。

## 獨立驗證

使用實際 HTML 控制器、Python 規則逐步核對。基準固定為 commit `d218222` 的 Trio v1.0，兩版使用同一份三人模型。舊策略包括 APEX、BASTION、SWEEP，對手有序配對共九種，包含相同策略；每組輪流三個座位。

| 測試 | 舊版勝／並列第一／敗 | 新版勝／並列第一／敗 |
|---|---:|---:|
| 正常開局，10 seeds×9 組×3 席＝270 | 32／1／237 | 119／3／148 |
| 4 次隨機前綴，5 seeds×9 組×3 席＝135 | 33／1／101 | 44／1／90 |

正常開局分組，每組 30 場：

| 對手（按席次順序） | 舊版勝 | 新版勝 |
|---|---:|---:|
| APEX＋APEX | 3 | 11 |
| APEX＋BASTION | 3 | 12 |
| APEX＋SWEEP | 4 | 13 |
| BASTION＋APEX | 4 | 11 |
| BASTION＋BASTION | 5 | 19 |
| BASTION＋SWEEP | 2 | 18 |
| SWEEP＋APEX | 3 | 9 |
| SWEEP＋BASTION | 6 | 14 |
| SWEEP＋SWEEP | 2 | 12 |

正常開局三席各 90 場：22→52、6→34、4→33 勝。平均自身分數減最高對手分數：-21.97→-4.94；最大行動嘗試數 150→111。

隨機局面仍有退步：第三席 17→13／45 勝，BASTION＋SWEEP 7→3／15 勝，SWEEP＋SWEEP 5→3／15 勝；整體平均分差 -18.47→-13.67，最大嘗試數 101→104。不可只報正常開局提升而忽略這些弱點。共享局面／種子的資料並非全部互相獨立，不把比例直接當作普遍勝率或顯著性證明。

## 驗證與重現

- 新增聯合動作數量／唯一性／權重和／合法性測試，以及路線步進／禁格不消耗路線測試。
- 原有 160 筆雙人搜尋 fixture、碰撞、圈地、尾盤結束測試保持通過。
- 真實瀏覽器三人全自對弈：71 回合、73 次嘗試填滿棋盤；不是統計強度測試。雙人 15／30 階完整對局、模型數值比對、模式切換、暫停、過期回應與離線失敗保護也通過。
- 正式程式另重跑正常開局與隨機局面，與選定的原型結果逐筆比較，不只依原型判定。

```powershell
# 保留舊控制器，重現消融；所有輸出資料夾須尚不存在。
.venv\Scripts\python.exe -m training.trio_search_eval --source-ref d218222 --output training/runs/trio-ablation-repro
.venv\Scripts\python.exe -m training.trio_search_eval --source-ref d218222 --variants baseline exact-enclosure --same-style --seeds 911 2387 6503 11807 19751 32719 48311 70183 90709 121007 --output training/runs/trio-fixed-repro
.venv\Scripts\python.exe -m training.trio_search_eval --source-ref d218222 --variants baseline exact-enclosure --same-style --prefix 4 --seeds 17713 29347 51827 79843 139021 --output training/runs/trio-prefix-repro
# 正式工作樹的實作，不注入原型：
.venv\Scripts\python.exe -m training.trio_search_eval --source-ref current --variants baseline --same-style --seeds 911 2387 6503 11807 19751 32719 48311 70183 90709 121007 --output training/runs/trio-production-repro
node --test tests/atlas.test.cjs tests/collision.test.cjs
.venv\Scripts\python.exe -m unittest training.test_learning training.test_three_player
.venv\Scripts\python.exe -m training.verify --size 15
```

`baseline` 參數表示不注入實驗變更；搭配 `--source-ref current` 時就是目前正式程式，而非歷史舊版。原型僅透過離線 Oracle 的環境變數啟用，瀏覽器不能選取或注入。每次測試保存配置、檔案雜湊及逐局 seed／對手／座位／分數／結束狀態。

下一步若繼續追求更高強度，優先研究計畫中途的風險重估及隨機局面的退步，再考慮對路線選擇訓練模型；不能把現有四方向 PPO 直接當作已學會長程規劃。
