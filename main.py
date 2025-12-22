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

def predict_mask(t1_img, t2_img):
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
    # TODO: 請在此實作你的 segmentation 預測方法
    raise NotImplementedError("請完成 predict_mask(t1_img, t2_img)")


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
        folder = QFileDialog.getExistingDirectory(self, "Select T1 Folder")
        if folder:
            self.t1_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    def load_t2_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select T2 Folder")
        if folder:
            self.t2_images = self.load_folder_images(folder)
            self.idx = 0
            self.spin_idx.setValue(0)
            self.update_spin_range()
            self.update_base_images()

    # ---------------- 載入 mask 資料夾 ----------------
    def load_mask_folder(self, kind: str):
        folder = QFileDialog.getExistingDirectory(self, f"Select {kind} Mask Folder")
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

                    pred_bin = predict_mask(t1_img, t2_img) # TODO

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
