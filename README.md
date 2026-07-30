# Ray-based Population-Based Training (PBT) 系統

本專案是一個基於 [Ray](https://docs.ray.io/) 構建的高效能分散式**基於群體訓練（Population-Based Training, PBT）**超參數優化框架。系統支援異質運算資源（CPU / GPU 混合）的動態任務調度、工作偷取（Work-Stealing）策略、檢查點區域性（Checkpoint Locality）優化，以及實時訓練指標與日誌監控。

---

## 🌟 核心特色

1. **Population-Based Training (PBT) 演化機制**
   - 自動在訓練過程中動態調整超參數（如 Learning Rate、Momentum、Weight Decay、Batch Size）。
   - 定期評估試驗（Trial）表現，淘汰表現不佳的群體，並將表現優異者的權重與探索後的超參數進行突變（Mutation）複製。

2. **異質資源調度與工作偷取 (Work-Stealing Strategy)**
   - 支援 CPU 與 GPU Worker 混合排程。
   - 當 GPU 資源閒置且無待處理任務時，觸發偷取策略（Stealing Strategy），搶占並掛起低優先度的 CPU 訓練任務，移交給 GPU 加速執行。

3. **檢查點區域性優化 (Checkpoint Locality)**
   - 優先將 Trial 派發給先前保存過該 Trial Checkpoint 的 Worker，減少跨節點與網路模型權重的傳輸開銷。

4. **可擴充的任務策略 (Pluggable Task Strategy)**
   - 採用 Protocol 導向設計（`TaskStrategy`），預設支援：
     - **BERT / SST-2**: 基於 Hugging Face `MobileBertForSequenceClassification` 的文本分類微調。
     - **ResNet-18 / CIFAR-10**: 卷積神經網路影像分類。
     - **ResNet-50 / CIFAR-100**: 深層卷積網路小圖分類。

5. **實時視覺化與監控看板**
   - **ASCII 實時看板**: 動態將各 Trial 狀態、部署節點 IP、Worker ID、超參數與 Accuracy 輸出至文字進度檔。
   - **TensorBoard 整合**: 支援全域最佳解（Global Best Accuracy）與個股 Trial 的進度繪圖。
   - **自動化日誌打包**: 訓練結束後自動收集分散式集群中的所有 Worker 與 Scheduler 日誌並壓縮匯出。

---

- **Tuner**: 系統入口與協調者，負責初始化組件與掌控整體訓練生命週期。
- **TrialManager**: 管理所有 Trial 的生命週期狀態轉移（`PENDING` $\rightarrow$ `WAITING` $\rightarrow$ `RUNNING` $\rightarrow$ `TERMINATED` / `FAILED`），計算世代進度並執行 PBT 突變。
- **TrialScheduler**: 負責配對 Worker 與 Trial，執行 GPU/CPU 分派及任務偷取。
- **WorkerManager / Worker**: 實作模型構建、資料載入、訓練步驟（`train_step`）與評估（`evaluate`）。
- **TensorBoardManager**: 負責紀錄 TensorBoard 指標與全域統計資訊。

---

## ⚙️ 環境需求與安裝

### 1. 基礎環境

- **Python**: $\ge 3.10$
- **PyTorch**: 支援 CUDA 之版本（若需 GPU 加速）
- **Ray**: $\ge 2.0.0$

### 2. 安裝依賴套件

```bash
pip install torch torchvision transformers datasets ray tensorboard psutil
```

## 📝 設定檔說明 (`config.json`)

系統執行時會優先讀取專案根目錄下的 config.json。若檔案不存在則自動使用預設值。

### 範例 config.json

```JSON
{
  "dataset_path": "~/Documents/dataset/",
  "stop_accuracy": 0.90,
  "max_generation": 5,
  "iteration_per_generation": 20,
  "trial_progress_output_path": "~/Documents/Ray_PBT/trial_progress.output",
  "gpu_trials_limit": 2,
  "cpu_trials_limit": 1,
  "enable_checkpoint_locality": true
}
```

# 參數詳細說明

|參數名稱|型態|預設值|說明|
|----|-----|-----|-----|
|dataset_path|str|~/Documents/dataset/|CIFAR-10 / CIFAR-100 等資料集下載與儲存路徑|
|stop_accuracy|float|0.90|提前停止訓練的標的準確率（Target Accuracy）|
|max_generation|int|5|PBT 演化的最大世代數量|
|iteration_per_generation|int|20|每個世代中包含的訓練 Iteration 數|
|trial_progress_output_path|str|~/Documents/Ray_PBT/...|ASCII 終端實時監控看板寫入路徑|
|gpu_trials_limit|int|2|每個 GPU Worker 同時掛載/執行的 Trial 上限|
|cpu_trials_limit|int|1|每個 CPU Worker 同時執行的 Trial 上限|
|enable_checkpoint_locality|bool|true|是否啟用 Checkpoint 就近分配優先權機制|

---

## 🚀 使用說明 (Usage)

### 1. 快速開始 (Quick Start)

直接執行主程式即可啟動 Ray PBT 超參數搜尋（預設使用 BERT SST-2 分類任務，啟動 40 個 Trials）：

```Bash
python main.py --trial_num 40
```

### 2. 命令列參數說明

  ```Bash
  python main.py [-h] [--trial_num TRIAL_NUM]
  ```

- --trial_num: 設定初始 Population 的 Trial 總數（預設：40）。

### 3. 切換訓練任務與模型策略

若要切換不同的訓練任務（例如由 BERT 改為 ResNet-18 或 ResNet-50），只需修改 main.py 中的任務策略與超參數生成器：

#### 使用 MobileBERT 微調 SST-2（預設）

```Python
from src.hyperparameter import BertHyperparameter
from src.task_strategy import BertSST2Task


# 生成初始超參數
trial_states = generate_trial_states(BertHyperparameter, trial_num)

# 初始化 Tuner 並傳入任務策略
tuner = Tuner.remote(
    trial_states,
    BertSST2Task(),
    Path(f"./runs/BERT/Trial{trial_num}/{timestamp}").resolve(),
)
```

#### 使用 ResNet-18 訓練 CIFAR-10

```Python
from src.hyperparameter import CNNHyperparameter
from src.task_strategy import ResNet18Cifar10Task

trial_states = generate_trial_states(CNNHyperparameter, trial_num)

tuner = Tuner.remote(
    trial_states,
    ResNet18Cifar10Task(),
    Path(f"./runs/ResNet18_CIFAR10/Trial{trial_num}/{timestamp}").resolve(),
)
```

### 使用 ResNet-50 訓練 CIFAR-100

```Python
from src.hyperparameter import CNNHyperparameter
from src.task_strategy import ResNet50CIFAR100Task

trial_states = generate_trial_states(CNNHyperparameter, trial_num)

tuner = Tuner.remote(
    trial_states,
    ResNet50CIFAR100Task(),
    Path(f"./runs/ResNet50_CIFAR100/Trial{trial_num}/{timestamp}").resolve(),
)
```

## 🛠️ 如何擴充自訂任務

若您想要在自己的資料集或模型上使用此 PBT 框架，只需完成以下兩個步驟：

### 步驟 1：定義超參數類別 (Inherit Hyperparameter)

建立一個繼承自 `Hyperparameter` 的 DataClass，並實作 `random()` 與 `explore()` 方法：

```Python
from dataclasses import dataclass
import random
from src.utils import Hyperparameter

@dataclass
class CustomHyperparameter(Hyperparameter):
    lr: float
    batch_size: int
    weight_decay: float
    cpu_batch_size: int = 32

    @classmethod
    def random(cls) -> "CustomHyperparameter":
        return cls(
            lr=random.uniform(1e-4, 1e-2),
            batch_size=random.choice([16, 32, 64]),
            weight_decay=random.uniform(1e-5, 1e-3),
        )

    def explore(self) -> "CustomHyperparameter":
        # 定義 PBT 突變邏輯（例如：Learing Rate 微調 0.8x 或 1.2x）
        return CustomHyperparameter(
            lr=self.lr * random.choice([0.8, 1.2]),
            batch_size=self.batch_size,
            weight_decay=self.weight_decay * random.choice([0.8, 1.2]),
        )
```

### 步驟 2：實作任務策略類別 (Implement `TaskStrategy`)

繼承 `TaskStrategy[CustomHyperparameter]` 並實作對應介面：

```Python
from src.task_strategy import TaskStrategy
from src.utils import Checkpoint

class CustomTask(TaskStrategy[CustomHyperparameter]):
    def build_model(self, hyperparameter, checkpoint, device):
        model = MyModel()
        if not checkpoint.is_empty():
            model.load_state_dict(checkpoint.model_state_dict)
        return model.to(device)

    def build_optimizer(self, model, hyperparameter, checkpoint, device):
        optimizer = torch.optim.AdamW(model.parameters(), lr=hyperparameter.lr)
        if not checkpoint.is_empty():
            optimizer.load_state_dict(checkpoint.optimizer_state_dict)
        return optimizer

    def build_dataloaders(self, hyperparameter, device):
        # 回傳 (train_loader, test_loader, valid_loader)
        ...

    def train_step(self, model, optimizer, dataloader, device):
        # 執行 1 個世代/分批步進的訓練
        ...

    def evaluate(self, model, dataloader, device) -> float:
        # 回傳當前準確率 (Accuracy) 或評估指標
        ...
```

## 專案目錄結構

```txt
.
├── config-example.json               # 設定檔範本（提供參數設定範例與預設值）
├── config.json                       # 系統實際執行的參數設定檔（包含數據路徑、GPU/CPU 限制等）
├── LICENSE                           # 專案開源授權條款檔案（MIT License）
├── logs/                             # 訓練日誌、Worker/Scheduler 終端輸出與打包歸檔儲存目錄
├── main.py                           # 專案主程式進入點（解析命令行參數、啟動 Ray 並初始化 Tuner）
├── pyproject.toml                    # Python 專案建構與套件管理配置（uv / poetry 規範）
├── README.md                         # 專案說明文件（包含架構圖、模組介紹與使用教學）
├── requirements.txt                  # Python 依賴套件清單（提供 pip 環境安裝）
├── test.py                           # 單元測試執行主腳本
├── uv.lock                           # uv 套件管理器的套件版本鎖定檔 (Lockfile)
│
├── src/                              # 核心系統原始碼目錄
│   ├── __init__.py                   # 將 src 宣告為 Python Package
│   ├── config.py                     # 讀取並解析 config.json 的全域配置管理類別
│   ├── hyperparameter.py             # 超參數資料結構定義與 PBT 隨機採樣 / 突變 (Mutation) 邏輯
│   ├── task_strategy.py              # 訓練任務策略介面 (Protocol) 與實作（BERT、ResNet-18/50）
│   ├── tensorboard_manager.py        # 非同步 TensorBoard 日誌寫入與群體指標紀錄 Actor
│   ├── trial_manager.py              # Trial 狀態佇列管理與 PBT 世代淘汰 / 複製演化機制
│   ├── trial_scheduler.py            # 異質資源調度器（包含 GPU/CPU 派發、Checkpoint 就近優先與 Work-Stealing）
│   ├── trial_state.py                # 單一 Trial 的生命週期狀態 (PENDING, RUNNING...) 與元數據結構
│   ├── tuner.py                      # Tuner 核心協調 Actor（掌控 PBT 訓練生命週期主迴圈與日誌歸檔）
│   ├── utils.py                      # 通用工具函式（Checkpoint 封裝、列舉型態與 ASCII 繪圖）
│   ├── worker_manager.py             # Worker 管理器 Actor（負責創建與維護分散式集群 Worker）
│   └── worker.py                     # 實體運算 Worker Actor（執行模型的 train_step 與 evaluate）
│
└── test/                             # 單元測試套件目錄 (Unit Tests)
    ├── __init__.py                   # 將 test 宣告為 Python Package
    ├── test_checkpoint.py            # 測試模型權重與優化器 Checkpoint 的 Save/Load 快照機制
    ├── test_trial_manager_transitions.py  # 測試 TrialManager 的狀態轉移 (State Machine) 與佇列排隊邏輯
    ├── test_trial_state.py          # 測試 TrialState 物件建立、屬性更新與邊界條件
    └── test_worker.py                # 測試 Worker 的模型訓練步驟、評估與數據回傳
```
