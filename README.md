# 影像處理期末專案  
## 醫學影像分割 — 前臂肌肉的 MRI 影像分割

---

## 一、專案簡介

本期末專案旨在讓學生實作**醫學影像分割**的完整流程，  
透過影像處理技術，對前臂的 **MRI 影像** 進行肌肉與相關組織的自動化分割。

學生需根據提供的 **T1 與 T2 MRI 影像**，設計影像處理或演算法方法，  
將影像中的組織區域分割為以下三種類別：

- **CT**（Carpal Tunnel，腕隧道）
- **FT**（Flexor Tendons，屈肌肌腱）
- **MN**（Median Nerve，正中神經）

分割結果將以 **Dice Coefficient（Dice 相似度係數）** 作為評估指標，  
用以量化預測結果與 Ground Truth 之間的重疊程度。

<div style="break-after: page; page-break-after: always;"></div>

## 二、資料集說明（Dataset）

本作業共提供 **10 組前臂 MRI 影像資料**，檔名編號如下：

```
0.png ~ 9.png
```

### 每組資料包含：

- **T1 MRI 影像**
- **T2 MRI 影像**
- **Ground Truth segmentation masks**
  - CT mask
  - FT mask
  - MN mask

### 資料特性

- 所有影像皆已對齊（registered）
- Ground Truth mask 為二值影像（0 / 1）
- 需同時利用 **T1 與 T2 影像資訊** 進行三類組織的分割

---

## 三、作業目標

學生需完成以下目標：

1. 理解醫學影像中不同組織在 T1 / T2 影像下的特性差異  
2. 設計影像處理或演算法流程，完成自動化分割  
3. 正確產生三類組織的 segmentation mask（CT / FT / MN）  
4. 使用 Dice Coefficient 評估分割結果  
5. 將方法整合至提供的 GUI 系統中進行視覺化與驗證  

---

## 四、專案架構說明

專案已提供一套 **PyQt5 視覺化介面（Segmentation Viewer）**，功能包含：

- 載入 T1 / T2 MRI 影像
- 顯示 Ground Truth segmentation
- 顯示學生實作的預測 segmentation
- 計算並顯示 Dice Coefficient
- 支援影像逐張瀏覽與切換

學生**僅需實作分割演算法核心函式**，不需修改 GUI 架構。

---

## 五、需完成的核心函式

### `predict_mask(t1_img, t2_img)`

```python
def predict_mask(t1_img, t2_img):
    """
    根據輸入的 T1 與 T2 MRI 影像，
    產生對應的 segmentation mask。
    """
```

- 輸入
    - t1_img：T1 MRI 影像（灰階，uint8）
    - t2_img：T2 MRI 影像（灰階，uint8）
- 輸出
    - pred_bin：二值 segmentation mask（0 / 1）

---

## 六、評估指標（Evaluation Metric）

### Dice Coefficient

本專案採用 **Dice Coefficient（Dice 相似度係數）** 作為分割結果的評估指標，用以衡量模型預測結果與 Ground Truth 之間的重疊程度，其定義如下：

\[
Dice = \frac{2 |A \cap B|}{|A| + |B|}
\]

其中：
- \(A\) 為預測的 segmentation mask  
- \(B\) 為 Ground Truth mask  

Dice Coefficient 的數值範圍介於 **0 到 1**：
- **1** 表示預測結果與 Ground Truth 完全一致  
- **0** 表示兩者完全沒有重疊  

系統會自動針對 **CT / FT / MN** 三類組織分別計算 Dice 分數，並顯示於介面中。

---

<div style="break-after: page; page-break-after: always;"></div>

## 七、環境安裝（Environment Setup）

本專案使用 **Python 3.8.20**，請確保執行環境符合版本需求。  
學生可選擇以下任一方式進行環境建置。

### 方法一：使用 `requirements.txt`

1. 建立並啟用 Python 環境（可使用既有環境）
2. 於專案目錄下執行：

```bash
pip install -r requirements.txt
```

### 方法二：建立 Conda 環境（建議）

```bash
conda create -n mri_seg python=3.8.20
conda activate mri_seg
pip install -r requirements.txt
```

---

## 八、程式執行方式

請於專案目錄中執行以下指令啟動系統：

```bash
python main.py
```

---

<div style="break-after: page; page-break-after: always;"></div>

## 九、注意事項

- 分割結果必須依據 T1 / T2 原始影像 產生
- 不可直接使用 Ground Truth mask 作為預測結果
- 輸出的 segmentation mask 必須為二值影像（0 或 1）
- 程式需可正常執行，不可產生 runtime error
- 預測結果應合理反映實際分割誤差，而非完美對齊 Ground Truth