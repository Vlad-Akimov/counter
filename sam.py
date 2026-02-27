import os
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List

# ===== SAM IMPORTS =====
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator


# ==============================================================
# DATA STRUCTURES
# ==============================================================

@dataclass
class StackDetectionResult:
    mask: np.ndarray
    bbox: Tuple[int, int, int, int]
    score: float


# ==============================================================
# SAM STACK DETECTOR
# ==============================================================

class SAMStackDetector:
    """
    Automatic metal sheet stack detector using:
        1. Edge density ROI detection
        2. SAM automatic segmentation
        3. Mask scoring (area + texture + line parallelism)
    """

    def __init__(
        self,
        sam_checkpoint: str,
        model_type: str = "vit_b",
        device: str = "cuda"
    ):

        print("Loading SAM model...")
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)

        self.mask_generator = SamAutomaticMaskGenerator(
            model=sam,
            points_per_side=48,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.9,
            crop_n_layers=1,
            min_mask_region_area=500,
        )

    # ==========================================================
    # PREPROCESS
    # ==========================================================

    def reduce_glare(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.where(v > 200, v * 0.7, v).astype(np.uint8)
        hsv_corrected = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)

    def find_roi(self, image):
        """
        Edge density ROI (simplified version of your extractor)
        """

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(3.0, (8, 8))
        enhanced = clahe.apply(gray)
        denoised = cv2.medianBlur(enhanced, 5)

        sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)

        grad = np.sqrt(sobel_x**2 + sobel_y**2)
        grad = np.uint8(np.clip(grad, 0, 255))

        _, edge = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((3, 3), np.uint8)
        edge = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, kernel, iterations=2)

        # density map
        win = min(image.shape[:2]) // 15
        edge_f = edge.astype(np.float32) / 255.0
        kernel_d = np.ones((win, win), np.float32) / (win * win)
        density = cv2.filter2D(edge_f, -1, kernel_d)

        thr = 0.6 * np.max(density)
        mask = density > thr

        if not np.any(mask):
            h, w = image.shape[:2]
            return image, (0, 0, w, h), grad

        ys, xs = np.where(mask)

        x1, x2 = np.min(xs), np.max(xs)
        y1, y2 = np.min(ys), np.max(ys)

        # padding
        pad = win
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(image.shape[1], x2 + pad)
        y2 = min(image.shape[0], y2 + pad)

        roi = image[y1:y2, x1:x2]
        grad_roi = grad[y1:y2, x1:x2]

        return roi, (x1, y1, x2, y2), grad_roi

    # ==========================================================
    # SCORING
    # ==========================================================

    def area_score(self, mask):
        return mask.sum() / mask.size

    def density_score(self, mask, gradient):
        if mask.sum() == 0:
            return 0
        return gradient[mask].mean() / 255.0

    def line_score(self, mask):
        mask_u8 = (mask * 255).astype(np.uint8)
        edges = cv2.Canny(mask_u8, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, 60)

        if lines is None:
            return 0

        angles = [l[0][1] for l in lines]
        return 1 - np.std(angles) / np.pi

    # ==========================================================
    # MAIN STACK DETECTION
    # ==========================================================

    def detect_stack(self, image):

        image = self.reduce_glare(image)

        roi, (x1, y1, x2, y2), gradient = self.find_roi(image)

        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        masks = self.mask_generator.generate(roi_rgb)

        best_score = -1
        best_mask = None

        roi_area = roi.shape[0] * roi.shape[1]

        for m in masks:
            mask = m["segmentation"]

            area = mask.sum()
            if area < 0.1 * roi_area:
                continue

            a = self.area_score(mask)
            d = self.density_score(mask, gradient)
            l = self.line_score(mask)

            score = 0.4*a + 0.4*d + 0.2*l

            if score > best_score:
                best_score = score
                best_mask = mask

        if best_mask is None:
            raise RuntimeError("Stack not detected")

        # refine
        mask_u8 = best_mask.astype(np.uint8)*255
        kernel = np.ones((5,5), np.uint8)
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        ys, xs = np.where(mask_u8 > 0)
        bx1, bx2 = xs.min(), xs.max()
        by1, by2 = ys.min(), ys.max()

        # convert to global coords
        gx1, gy1 = bx1 + x1, by1 + y1
        gx2, gy2 = bx2 + x1, by2 + y1

        full_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        full_mask[y1:y2, x1:x2] = mask_u8

        return StackDetectionResult(
            mask=full_mask,
            bbox=(gx1, gy1, gx2, gy2),
            score=best_score
        )


# ==============================================================
# MAIN
# ==============================================================

def main():

    SAM_CHECKPOINT = "sam_vit_b.pth"   # <-- path to checkpoint
    IMAGE_PATH = "res/photos/q2.jpg"           # <-- test image

    if not os.path.exists(IMAGE_PATH):
        print("Image not found")
        return

    detector = SAMStackDetector(
        sam_checkpoint=SAM_CHECKPOINT,
        model_type="vit_b",
        device="cuda"
    )

    img = cv2.imread(IMAGE_PATH)

    result = detector.detect_stack(img)

    print("Detection score:", result.score)
    print("BBox:", result.bbox)

    # visualization
    vis = img.copy()
    x1, y1, x2, y2 = result.bbox
    cv2.rectangle(vis, (x1, y1), (x2, y2), (0,255,0), 2)

    overlay = vis.copy()
    overlay[result.mask>0] = (0,0,255)

    cv2.imshow("bbox", vis)
    cv2.imshow("mask", overlay)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
