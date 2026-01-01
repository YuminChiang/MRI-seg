import sys
import os

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QGroupBox, QHBoxLayout, QVBoxLayout, QSpinBox, QTabWidget, QGridLayout,
    QFrame, QMessageBox
)
    
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt

import cv2
import numpy as np

SIZE = 350

def make_image_box(size=SIZE):
    lbl = QLabel()
    lbl.setFixedSize(size, size)
    lbl.setFrameStyle(QFrame.Box | QFrame.Plain)
    lbl.setLineWidth(3)
    lbl.setAlignment(Qt.AlignCenter)
    return lbl


def draw_mask(image, mask):
    """
    image: RGB uint8 (H, W, 3)
    mask: 0/1 或 bool (H, W)
    回傳：亮綠半透明 overlay 後的 RGB uint8
    """
    masked_image = image.copy()
    mask_bool = mask.astype(bool)

    # 將 mask 區域塗成亮綠色
    masked_image[mask_bool] = (0, 255, 0)

    # blend: 原圖 30% + 綠色 70%
    return cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

def draw_predict_mask(base_img, gt_mask, pred_mask):
    """
    base_img: RGB uint8 (H, W, 3)
    gt_mask: 0/1 GT mask
    pred_mask: 0/1 預測 mask（畫紅線）
    """
    # Step 1: 先畫綠色透明 GT mask
    overlay = draw_mask(base_img, gt_mask)

    # Step 2: 再畫紅線框（thickness=1）
    mask_u8 = (pred_mask.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.drawContours(bgr, contours, -1, (0, 0, 255), thickness=1)

    # 回到 RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def get_arm_roi(t1, t2):
    """提取手臂區域 ROI
    
    使用 Otsu 自適應閾值對 T1 和 T2 影像進行二值化，
    融合後通過形態學閉運算填補孔洞，最後提取最大輪廓作為 ROI。
    
    Args:
        t1: T1 加權影像，灰度圖 (H, W)，uint8
        t2: T2 加權影像，灰度圖 (H, W)，uint8
    
    Returns:
        roi: 手臂區域二值 mask，shape=(H, W)，數值 0 或 255
    
    Notes:
        若未檢測到輪廓，返回全白 mask (255)
    """
    # 高斯模糊 + Otsu 閾值
    t1_blur = cv2.GaussianBlur(t1, (5, 5), 0)
    t2_blur = cv2.GaussianBlur(t2, (5, 5), 0)
    _, t1_bin = cv2.threshold(t1_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, t2_bin = cv2.threshold(t2_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # 融合 + 閉運算
    merged = cv2.bitwise_or(t1_bin, t2_bin)
    closed = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    
    # 提取最大輪廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.full_like(t1, 255, dtype=np.uint8)
    
    roi = np.zeros_like(t1, dtype=np.uint8)
    cv2.drawContours(roi, [max(contours, key=cv2.contourArea)], -1, 255, cv2.FILLED)
    return roi

def filter_by_distance(mask, min_area=30, max_dist=70):
    """基於質心距離過濾連通組件
    
    計算所有組件的加權質心，保留面積大於閾值且距離質心較近的組件，
    用於去除離群的小雜訊區域。
    
    Args:
        mask: 輸入二值 mask，shape=(H, W)，uint8
        min_area: 組件最小面積閾值（像素數），默認 30
        max_dist: 到加權質心的最大歐氏距離，默認 70
    
    Returns:
        result: 過濾後的二值 mask，shape=(H, W)
        weighted_center: 加權質心座標 (x, y)，若無有效組件則為 None
    
    Notes:
        質心採用面積加權計算，較大組件對質心位置影響更大
    """
    num, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask, dtype=np.uint8)
    
    if num <= 1:
        return result, None
    
    # 面積篩選
    valid_ids = [i for i in range(1, num) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not valid_ids:
        return result, None
    
    # 計算加權質心
    areas = stats[valid_ids, cv2.CC_STAT_AREA]
    centers = centroids[valid_ids]
    total_area = areas.sum()
    
    if total_area == 0:
        return result, None
    
    weighted_center = (areas[:, None] * centers).sum(axis=0) / total_area
    
    # 距離過濾
    for idx in valid_ids:
        if np.linalg.norm(centroids[idx] - weighted_center) <= max_dist:
            result[labels == idx] = 255
    
    return result, weighted_center

def dilate_constrained(mask, ref_t1, ref_t2, iters=1, grad_thresh=40):
    """執行邊緣約束的條件膨脹
    
    結合 T1 和 T2 影像的形態學梯度，定義安全生長區域，
    在避開強邊緣的前提下進行膨脹，防止跨越組織邊界。
    
    Args:
        mask: 種子區域 mask，shape=(H, W)，uint8
        ref_t1: T1 參考影像，用於計算梯度，shape=(H, W)，uint8
        ref_t2: T2 參考影像，用於計算梯度，shape=(H, W)，uint8
        iters: 膨脹迭代次數，默認 1
        grad_thresh: 邊緣梯度閾值，梯度大於此值視為邊界，默認 40
    
    Returns:
        current: 膨脹後的 mask，shape=(H, W)，uint8
    
    Notes:
        使用 3x3 結構元素進行膨脹
        若某次迭代無有效生長則提前終止
    """
    kernel = np.ones((3, 3), np.uint8)
    
    # 計算梯度並定義安全區
    grad_t1 = cv2.morphologyEx(ref_t1, cv2.MORPH_GRADIENT, kernel)
    grad_t2 = cv2.morphologyEx(ref_t2, cv2.MORPH_GRADIENT, kernel)
    grad_max = cv2.max(grad_t1, grad_t2)
    _, safe_zone = cv2.threshold(grad_max, grad_thresh, 255, cv2.THRESH_BINARY_INV)
    
    # 迭代膨脹
    current = mask.copy()
    for _ in range(iters):
        expanded = cv2.dilate(current, kernel, iterations=1)
        new_pixels = cv2.bitwise_xor(expanded, current)
        valid_growth = cv2.bitwise_and(new_pixels, safe_zone)
        
        if cv2.countNonZero(valid_growth) == 0:
            break
        
        current = cv2.bitwise_or(current, valid_growth)
    
    return current

def remove_small(mask, min_size):
    """移除面積小於閾值的連通組件
    
    Args:
        mask: 輸入二值 mask，shape=(H, W)，uint8
        min_size: 最小面積閾值（像素數）
    
    Returns:
        result: 過濾後的 mask，僅保留面積 >= min_size 的組件
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    result = np.zeros_like(mask, dtype=np.uint8)
    
    for i in range(1, num):
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            result[labels == i] = 255
    
    return result

# ==========================================
# Section 2: CT (腕隧道) 生成
# ==========================================
def get_ct_mask(ft_mask, t1, t2, iters=15):
    """生成腕隧道 (Carpal Tunnel) 分割 mask
    
    採用三階段策略：
    1. 凸包初始化：從 FT mask 的凸包開始
    2. 引導生長：基於 T1/T2 雙模態特徵構建引導圖，進行受限區域生長
    3. 幾何平滑：融合凸包和橢圓擬合結果，高斯平滑後二值化
    
    Args:
        ft_mask: 屈指肌腱 (Flexor Tendons) mask，shape=(H, W)，uint8
        t1: T1 加權影像，灰度圖，shape=(H, W)，uint8
        t2: T2 加權影像，灰度圖，shape=(H, W)，uint8
        iters: 區域生長迭代次數，默認 15
    
    Returns:
        result: 腕隧道二值 mask，shape=(H, W)，數值 0 或 255
    
    Notes:
        - T1 引導圖：保留灰度值 40-200 的組織區域，排除脂肪（>200）
        - T2 引導圖：使用 Otsu 自適應閾值檢測高亮區域
        - 搜索範圍限制在初始凸包膨脹 35 像素的區域內
    """
    # 初始化：FT 凸包
    pts = cv2.findNonZero(ft_mask)
    ct = np.zeros_like(ft_mask, dtype=np.uint8)
    if pts is not None:
        hull = cv2.convexHull(pts)
        cv2.drawContours(ct, [hull], -1, 255, cv2.FILLED)
    
    # 構建引導圖
    t1_tissue = cv2.inRange(t1, 40, 200)
    t2_blur = cv2.GaussianBlur(t2, (5, 5), 0)
    otsu_val, _ = cv2.threshold(t2_blur, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, t2_bright = cv2.threshold(t2_blur, otsu_val * 0.9, 255, cv2.THRESH_BINARY)
    
    guide = cv2.bitwise_or(t1_tissue, t2_bright)
    _, fat = cv2.threshold(t1, 200, 255, cv2.THRESH_BINARY)
    guide = cv2.bitwise_and(guide, cv2.bitwise_not(fat))
    guide = cv2.morphologyEx(guide, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    
    # 限制搜索範圍
    boundary = cv2.dilate(ct, np.ones((35, 35), np.uint8))
    guide = cv2.bitwise_and(guide, boundary)
    
    # 區域生長
    kernel = np.ones((3, 3), np.uint8)
    for _ in range(iters):
        expanded = cv2.dilate(ct, kernel, iterations=1)
        growth = cv2.bitwise_and(cv2.bitwise_xor(expanded, ct), guide)
        if cv2.countNonZero(growth) == 0:
            break
        ct = cv2.bitwise_or(ct, growth)
    
    # 幾何平滑
    pts = cv2.findNonZero(ct)
    if pts is None:
        return ct
    
    hull_mask = np.zeros_like(ct, dtype=np.uint8)
    cv2.drawContours(hull_mask, [cv2.convexHull(pts)], -1, 255, cv2.FILLED)
    
    ellipse_mask = np.zeros_like(ct, dtype=np.uint8)
    if len(pts) >= 5:
        cv2.ellipse(ellipse_mask, cv2.fitEllipse(pts), 255, cv2.FILLED)
    else:
        ellipse_mask = hull_mask.copy()
    
    fused = cv2.bitwise_or(hull_mask, ellipse_mask)
    smoothed = cv2.GaussianBlur(fused, (21, 21), 0)
    _, result = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    
    return result

# ==========================================
# Section 3: MN (正中神經) 檢測
# ==========================================
def get_mn_mask(t2, ct_mask, ft_mask, ref_t1, ref_t2, min_area=50):
    """檢測正中神經 (Median Nerve) 區域
    
    在 CT 內且 FT 外的搜索區域內，使用以下策略檢測 MN：
    1. 計算 FT 質心作為參考點
    2. 使用 Otsu 閾值檢測 T2 影像中的亮點候選
    3. 對候選進行條件膨脹增強
    4. 基於橢圓擬合度（面積誤差）篩選最佳候選
    
    Args:
        t2: T2 加權影像，灰度圖，shape=(H, W)，uint8
        ct_mask: 腕隧道 mask，shape=(H, W)，uint8
        ft_mask: 屈指肌腱 mask，shape=(H, W)，uint8
        ref_t1: T1 參考影像（用於膨脹邊緣約束），shape=(H, W)，uint8
        ref_t2: T2 參考影像（用於膨脹邊緣約束），shape=(H, W)，uint8
        min_area: 候選組件最小面積閾值（膨脹後），默認 50
    
    Returns:
        mn: 正中神經二值 mask，shape=(H, W)，數值 0 或 255
        debug: 膨脹後的所有候選區域 mask（調試用）
        ft_center: FT 質心座標 (x, y)
        best_ell: 最佳橢圓參數 ((cx, cy), (w, h), angle)，無則為 None
        best_score: 最佳橢圓擬合分數（越低越好），無則為 None
    
    Notes:
        - 橢圓評分公式：|實際面積 - 橢圓面積| / 橢圓面積
        - 若 FT mask 為空，返回空 mask 和圖像中心作為默認參考點
        - 條件膨脹參數：迭代 1 次，梯度閾值 90
    """
    mn = np.zeros_like(t2, dtype=np.uint8)
    
    # 計算 FT 質心
    num_ft = cv2.connectedComponentsWithStats(ft_mask, connectivity=8)[0]
    if num_ft <= 1:
        h, w = t2.shape
        empty = np.zeros_like(t2, dtype=np.uint8)
        return mn, empty, (w // 2, h // 2), None, None
    
    M = cv2.moments(ft_mask)
    ft_center = (int(M["m10"] / (M["m00"] + 1e-5)), int(M["m01"] / (M["m00"] + 1e-5)))
    
    # 定義搜索區
    search = cv2.bitwise_and(ct_mask, cv2.bitwise_not(ft_mask))
    pixels = t2[search > 0]
    
    if len(pixels) == 0:
        return mn, np.zeros_like(t2, dtype=np.uint8), ft_center, None, None
    
    # 檢測亮點候選
    otsu_val, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    _, candidates = cv2.threshold(t2, otsu_val, 255, cv2.THRESH_BINARY)
    candidates = cv2.bitwise_and(candidates, search)
    candidates = cv2.morphologyEx(candidates, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    
    # 條件膨脹
    grown = dilate_constrained(candidates, ref_t1, ref_t2, iters=1, grad_thresh=90)
    debug = grown.copy()
    
    # 輪廓分析與橢圓評分
    contours, _ = cv2.findContours(grown, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    best_cnt, best_ell, best_score = None, None, float('inf')
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or len(cnt) < 5:
            continue
        
        ell = cv2.fitEllipse(cnt)
        w, h = ell[1]
        ell_area = (np.pi * w * h) / 4.0
        
        if ell_area < 1e-6:
            continue
        
        score = abs(area - ell_area) / ell_area
        
        if score < best_score:
            best_score = score
            best_cnt = cnt
            best_ell = ell
    
    if best_cnt is not None:
        cv2.drawContours(mn, [best_cnt], -1, 255, cv2.FILLED)
    
    return mn, debug, ft_center, best_ell, best_score


def predict_mask(t1_img, t2_img, kind=None):
    """
    TODO【影像分割預測實作】

    本函式需根據輸入的 T1 與 T2 原始影像，
    設計一個影像處理或演算法流程，
    自動產生對應的分割結果 (segmentation mask)。

    輸入：
        t1_img (np.ndarray):
            T1 原始影像，shape = (H, W) ，dtype = uint8

        t2_img (np.ndarray):
            T2 原始影像，shape = (H, W) ，dtype = uint8

    輸出：
        pred_bin (np.ndarray):
            預測的二值 segmentation mask，
            shape = (H, W)，dtype = uint8，數值為 {0, 1}

    實作要求：
        1. 輸出 mask 尺寸必須與輸入影像相同
        2. 輸出必須為二值影像（0 或 1）
        3. 分割結果必須根據影像內容產生，
           不可使用 Ground Truth mask 作為輸入
        4. 需包含實際的影像處理或演算法流程，
           例如 thresholding、filtering、morphology、canny 等

    提示：
        - T1 與 T2 可擇一使用，或結合兩者資訊
        - 可自行設計規則或條件判斷

    評分重點：
        - 分割邏輯是否合理
        - 是否確實使用影像資訊進行預測
        - 程式可讀性與穩定性
    """
    # 參數配置
    DARK_THRESH = 30        # FT 暗區閾值
    MORPH_SIZE = 3          # 形態學結構元素大小
    
    FT_MIN_AREA = 20        # FT 最小面積
    FT_MAX_DIST = 40        # FT 質心最大距離
    FT_DILATE_ITERS = 2     # FT 膨脹次數
    FT_GRAD_THRESH = 80     # FT 梯度閾值
    FT_MIN_SIZE = 50        # FT 最小尺寸
    
    CT_GROW_ITERS = 3       # CT 生長迭代次數
    MN_MIN_AREA = 70        # MN 最小面積
    
    # === Stage 1: 屈指肌腱 (FT) 分割 ===
    arm_roi = get_arm_roi(t1_img, t2_img)
    
    # 提取雙模態暗區
    _, dark_t1 = cv2.threshold(t1_img, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
    _, dark_t2 = cv2.threshold(t2_img, DARK_THRESH, 255, cv2.THRESH_BINARY_INV)
    dark_seeds = cv2.bitwise_and(dark_t1, dark_t2)
    dark_seeds = cv2.bitwise_and(dark_seeds, arm_roi)
    
    # 形態學去噪
    kernel = np.ones((MORPH_SIZE, MORPH_SIZE), np.uint8)
    dark_seeds = cv2.morphologyEx(dark_seeds, cv2.MORPH_OPEN, kernel)
    
    # 距離過濾 + 條件膨脹 + 移除小組件
    ft, _ = filter_by_distance(dark_seeds, min_area=FT_MIN_AREA, max_dist=FT_MAX_DIST)
    ft = dilate_constrained(ft, t1_img, t2_img, iters=FT_DILATE_ITERS, grad_thresh=FT_GRAD_THRESH)
    ft = remove_small(ft, min_size=FT_MIN_SIZE)
    
    # === Stage 2: 腕隧道 (CT) 分割 ===
    ct = get_ct_mask(ft, t1_img, t2_img, iters=CT_GROW_ITERS)
    
    # === Stage 3: 正中神經 (MN) 分割 ===
    mn, *_ = get_mn_mask(t2_img, ct, ft, t1_img, t2_img, min_area=MN_MIN_AREA)
    
    # 返回指定類型的 mask
    masks = {"FT": ft, "CT": ct, "MN": mn}
    return (masks.get(kind, ft) > 0).astype(np.uint8)
    
def dice_coef(gt, pred):
    """
    gt, pred: 0/1 或 bool mask
    """
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    inter = np.logical_and(gt, pred).sum()
    s = gt.sum() + pred.sum()
    if s == 0:
        return 1.0
    return 2.0 * inter / s


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Segmentation Viewer")
        self.resize(1300, 700)

        # 影像列表
        self.t1_images = []
        self.t2_images = []

        # GT mask（已經 resize + binarize 過的 numpy）
        self.gt_masks = {
            "CT": [],
            "FT": [],
            "MN": [],
        }

        # 預測結果 mask（numpy, 0/1）
        self.pred_masks = {
            "CT": [],
            "FT": [],
            "MN": [],
        }

        # Dice per image
        self.dice_scores = {
            "CT": [],
            "FT": [],
            "MN": [],
        }

        self.idx = 0  # 當前第幾張（0-based）
        self.show_pred = False  # False: 顯示 GT mask; True: 顯示預測結果

        self.setup_ui()

    # ---------------- UI 佈局 ----------------
    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)

        # ========== 左邊：T1 / T2 ==========
        left_box = QGroupBox()
        left_layout = QVBoxLayout(left_box)

        left_layout.addWidget(QLabel("T1"))
        self.lbl_t1 = make_image_box()
        left_layout.addWidget(self.lbl_t1)

        left_layout.addWidget(QLabel("T2"))
        self.lbl_t2 = make_image_box()
        left_layout.addWidget(self.lbl_t2)
        left_layout.addStretch()

        # Load T1 + 左右切換
        btn_load_t1 = QPushButton("Load T1 folder")
        btn_prev = QPushButton("←")
        btn_next = QPushButton("→")

        btn_load_t1.clicked.connect(self.load_t1_folder)
        btn_prev.clicked.connect(self.prev_img)
        btn_next.clicked.connect(self.next_img)

        h1 = QHBoxLayout()
        h1.addWidget(btn_load_t1)
        h1.addStretch()
        h1.addWidget(btn_prev)
        h1.addWidget(btn_next)
        left_layout.addLayout(h1)

        # Load T2 + index
        btn_load_t2 = QPushButton("Load T2 folder")
        btn_load_t2.clicked.connect(self.load_t2_folder)

        self.spin_idx = QSpinBox()
        self.spin_idx.setMinimum(0)
        self.spin_idx.setMaximum(0)
        self.spin_idx.setValue(0)
        self.spin_idx.valueChanged.connect(self.go_index)
        
        self.lbl_filename = QLabel("")

        h2 = QHBoxLayout()
        h2.addWidget(btn_load_t2)
        h2.addStretch()
        h2.addWidget(self.spin_idx)
        h2.addWidget(self.lbl_filename)
        left_layout.addLayout(h2)

        # ========== 右邊：Tabs + CT/FT/MN ==========
        right_box = QGroupBox()
        right_layout = QVBoxLayout(right_box)

        self.tabs = QTabWidget()
        self.tab_t1 = QWidget()
        self.tab_t2 = QWidget()
        self.tabs.addTab(self.tab_t1, "T1")
        self.tabs.addTab(self.tab_t2, "T2")
        right_layout.addWidget(self.tabs)
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        # 每個 tab 各有一組 CT/FT/MN 顯示框 + Dice label
        self.result_boxes = {"T1": {}, "T2": {}}
        self.dice_labels = {"T1": {}, "T2": {}}
        self.build_tab("T1", self.tab_t1)
        self.build_tab("T2", self.tab_t2)

        # 下方：Load mask 三顆按鈕 + Predict
        bottom_layout = QHBoxLayout()

        btn_ct_mask = QPushButton("Load CT Mask folder")
        btn_ft_mask = QPushButton("Load FT Mask folder")
        btn_mn_mask = QPushButton("Load MN Mask folder")
        btn_predict = QPushButton("Predict")

        btn_ct_mask.clicked.connect(lambda: self.load_mask_folder("CT"))
        btn_ft_mask.clicked.connect(lambda: self.load_mask_folder("FT"))
        btn_mn_mask.clicked.connect(lambda: self.load_mask_folder("MN"))
        btn_predict.clicked.connect(self.predict_all)

        bottom_layout.addWidget(btn_ct_mask)
        bottom_layout.addWidget(btn_ft_mask)
        bottom_layout.addWidget(btn_mn_mask)
        bottom_layout.addSpacing(40)
        bottom_layout.addWidget(btn_predict)
        bottom_layout.addStretch()

        right_layout.addLayout(bottom_layout)

        # 加到 main layout
        main_layout.addWidget(left_box, 1)
        main_layout.addWidget(right_box, 3)

    # tab 裡面 CT / FT / MN 的三個框
    def build_tab(self, tab_name: str, container: QWidget):
        layout = QVBoxLayout(container)
        grid = QGridLayout()
        grid.setHorizontalSpacing(80)

        titles = ["CT", "FT", "MN"]
        for col, key in enumerate(titles):
            lbl_title = QLabel(key)
            box = make_image_box()
            lbl_dice = QLabel("Dice coefficient:")

            self.result_boxes[tab_name][key] = box
            self.dice_labels[tab_name][key] = lbl_dice

            grid.addWidget(lbl_title, 0, col, alignment=Qt.AlignCenter)
            grid.addWidget(box, 1, col, alignment=Qt.AlignCenter)
            grid.addWidget(lbl_dice, 2, col, alignment=Qt.AlignCenter)

        layout.addLayout(grid)
        layout.addStretch()

    # ---------------- 共用：更新 spin 上限 ----------------
    def update_spin_range(self):
        lengths = [len(self.t1_images), len(self.t2_images)]
        for lst in self.gt_masks.values():
            lengths.append(len(lst))
        max_len = max(lengths) if lengths else 0
        if max_len <= 0:
            self.spin_idx.setMaximum(0)
        else:
            self.spin_idx.setMaximum(max_len - 1)

    # ---------------- 載入影像資料夾 ----------------
    def load_folder_images(self, folder): # 按照檔名排序
        files = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        ]

        # 數字排序
        files = sorted(files, key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))

        return files

    def load_t1_folder(self):
        # folder = QFileDialog.getExistingDirectory(self, "Select T1 Folder")
        folder = './MRIsample/T1'
        if folder:
            self.t1_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    def load_t2_folder(self):
        # folder = QFileDialog.getExistingDirectory(self, "Select T2 Folder")
        folder = './MRIsample/T2'
        if folder:
            self.t2_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    # ---------------- 載入 mask 資料夾 ----------------
    def load_mask_folder(self, kind: str):
        # folder = QFileDialog.getExistingDirectory(self, f"Select {kind} Mask Folder")
        folder = f'./MRIsample/{kind}'
        if not folder:
            return

        files = self.load_folder_images(folder)
        size = (SIZE, SIZE)
        masks = []

        for path in files:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, size)
            mask_bin = (img > 127).astype(np.uint8)
            masks.append(mask_bin)

        self.gt_masks[kind] = masks
        # reset 該 kind 的預測
        self.pred_masks[kind] = []
        self.dice_scores[kind] = []

        self.show_pred = False  # 新 mask 進來，先回到 GT 顯示
        self.update_spin_range()
        self.update_base_images()  # 裡面會順便呼叫 update_results()

    # ---------------- 切換 index ----------------
    def prev_img(self):
        if self.idx > 0:
            self.idx -= 1
            self.spin_idx.blockSignals(True)
            self.spin_idx.setValue(self.idx)
            self.spin_idx.blockSignals(False)
            self.update_base_images()

    def next_img(self):
        if self.idx < self.spin_idx.maximum():
            self.idx += 1
            self.spin_idx.blockSignals(True)
            self.spin_idx.setValue(self.idx)
            self.spin_idx.blockSignals(False)
            self.update_base_images()

    def go_index(self, value):
        self.idx = value
        self.update_base_images()
        
    def update_filename_label(self):
        """
        根據目前 tab + idx 顯示對應影像的檔名
        """
        tab_name = "T1" if self.tabs.currentIndex() == 0 else "T2"
        base_list = self.t1_images if tab_name == "T1" else self.t2_images

        if base_list and self.idx < len(base_list):
            path = base_list[self.idx]
            name = os.path.basename(path)
            self.lbl_filename.setText(name)
        else:
            self.lbl_filename.setText("")

    # ---------------- 更新左邊 T1/T2 display ----------------
    def update_base_images(self):
        size = SIZE

        # T1
        if self.t1_images and self.idx < len(self.t1_images):
            pix = QPixmap(self.t1_images[self.idx]).scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_t1.setPixmap(pix)
        else:
            self.lbl_t1.clear()

        # T2
        if self.t2_images and self.idx < len(self.t2_images):
            pix = QPixmap(self.t2_images[self.idx]).scaled(
                size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.lbl_t2.setPixmap(pix)
        else:
            self.lbl_t2.clear()

        # 右邊 CT/FT/MN 同步更新
        self.update_results()
        
        # 更新檔案名稱
        self.update_filename_label()

    def on_tab_changed(self, index):
        # 每次切換 T1 / T2，都重新依照目前 tab 更新右側顯示
        self.update_results()
        self.update_filename_label()
    
    # ---------------- 核心：更新 CT / FT / MN 顯示 ----------------
    def update_results(self):
        """
        依照目前 tab (T1 or T2)，將
        - GT mask 或 預測 mask 疊到對應的 T1/T2 影像上
        - 更新 Dice label
        """
        tab_name = "T1" if self.tabs.currentIndex() == 0 else "T2"
        base_list = self.t1_images if tab_name == "T1" else self.t2_images

        if not base_list or self.idx >= len(base_list):
            for kind in ["CT", "FT", "MN"]:
                self.result_boxes[tab_name][kind].clear()
                self.dice_labels[tab_name][kind].setText("Dice coefficient:")
            return

        base_path = base_list[self.idx]
        base_img = cv2.imread(base_path)
        base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        base_img = cv2.resize(base_img, (SIZE, SIZE))

        for kind in ["CT", "FT", "MN"]:
            box = self.result_boxes[tab_name][kind]
            dice_label = self.dice_labels[tab_name][kind]

            mask_to_use = None
            dice_text = "Dice coefficient:"

            if self.show_pred and self.pred_masks[kind]:
                if self.idx < len(self.pred_masks[kind]):
                    mask_to_use = self.pred_masks[kind][self.idx]
                    if self.idx < len(self.dice_scores[kind]):
                        dice_text = f"Dice coefficient: {self.dice_scores[kind][self.idx]:.3f}"
            
            elif self.gt_masks[kind]:
                if self.idx < len(self.gt_masks[kind]):
                    mask_to_use = self.gt_masks[kind][self.idx]
                    dice_text = "Dice coefficient: -"

            if mask_to_use is None:
                box.clear()
                dice_label.setText("Dice coefficient:")
                continue
            
            # 是否有預測
            if not self.show_pred:
                # 僅顯示 GT mask
                overlay_np = draw_mask(base_img, mask_to_use)

            else:
                # 同時顯示 GT + 預測紅線
                gt = self.gt_masks[kind][self.idx] if self.idx < len(self.gt_masks[kind]) else None
                pred = self.pred_masks[kind][self.idx] if self.idx < len(self.pred_masks[kind]) else None

                if gt is None or pred is None:
                    box.clear()
                    continue
                
                overlay_np = draw_predict_mask(base_img, gt, pred)

            h, w, ch = overlay_np.shape
            bytes_per_line = ch * w
            qimg = QImage(
                overlay_np.data, w, h, bytes_per_line, QImage.Format_RGB888
            )
            pix = QPixmap.fromImage(qimg)

            box.setPixmap(pix)
            dice_label.setText(dice_text)

    # ---------------- Predict：針對所有圖做預測 + Dice ----------------
    def predict_all(self):
        """
        針對每個 kind (CT/FT/MN) 的所有 GT mask：
        - 產生 pred mask 
        - 計算 Dice
        之後將 self.show_pred 設為 True，
        再呼叫 update_results() 顯示預測 overlay + Dice
        """
        size = (SIZE, SIZE)

        try:
            for kind in ["CT", "FT", "MN"]:
                gt_list = self.gt_masks[kind]
                self.pred_masks[kind] = []
                self.dice_scores[kind] = []

                for i, gt_mask in enumerate(gt_list):
                    # 讀取對應 T1 / T2
                    t1_path = self.t1_images[i]
                    t2_path = self.t2_images[i]

                    t1_img = cv2.imread(t1_path, cv2.IMREAD_GRAYSCALE)
                    t2_img = cv2.imread(t2_path, cv2.IMREAD_GRAYSCALE)

                    t1_img = cv2.resize(t1_img, size)
                    t2_img = cv2.resize(t2_img, size)

                    pred_bin = predict_mask(t1_img, t2_img, kind) # TODO

                    d = dice_coef(gt_mask, pred_bin)
                    self.pred_masks[kind].append(pred_bin)
                    self.dice_scores[kind].append(d)

        except NotImplementedError:
            QMessageBox.warning(
                self,
                "尚未完成作業",
                "predict_mask() 尚未實作。\n\n"
                "請依照 TODO 說明，\n"
                "使用 T1 / T2 影像設計分割方法後再執行 Predict。"
            )
            return  # 中斷

        self.show_pred = True
        self.update_results()


# ---------------- main ----------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
