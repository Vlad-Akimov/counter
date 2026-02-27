import cv2
import numpy as np
from ultralytics import YOLO
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d


class MetalStackCounter:

    def __init__(self, yolo_model_path):
        self.model = YOLO(yolo_model_path)

    # ===============================
    # STAGE 1 — DETECTION
    # ===============================
    def detect_stack(self, image):
        results = self.model(image)[0]

        if len(results.boxes) == 0:
            raise Exception("Stack not detected")

        # берём bbox с максимальным conf
        confs = results.boxes.conf.cpu().numpy()
        idx = np.argmax(confs)

        box = results.boxes.xyxy[idx].cpu().numpy().astype(int)

        x1, y1, x2, y2 = box
        return image[y1:y2, x1:x2], box

    # ===============================
    # glare reduction
    # ===============================
    def reduce_glare(self, img):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        v = np.where(v > 220, v * 0.6, v).astype(np.uint8)

        hsv = cv2.merge([h, s, v])
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # ===============================
    # STAGE 2 — COUNTING
    # ===============================
    def count_sheets(self, roi, debug=False):

        roi = self.reduce_glare(roi)

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # CLAHE
        clahe = cv2.createCLAHE(3.0, (8, 8))
        gray = clahe.apply(gray)

        # Sobel Y → горизонтальные границы
        sobel = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel = np.abs(sobel)

        # интеграция по X → 1D профиль
        profile = sobel.mean(axis=1)

        # сглаживание
        profile = gaussian_filter1d(profile, sigma=2)

        # поиск пиков
        peaks, _ = find_peaks(
            profile,
            distance=5,
            prominence=np.std(profile) * 0.3
        )

        if debug:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(5, 10))
            plt.plot(profile)
            plt.scatter(peaks, profile[peaks], c='r')
            plt.title(f"Peaks = {len(peaks)}")
            plt.show()

        return len(peaks)

    # ===============================
    # FULL PIPELINE
    # ===============================
    def process_image(self, path, debug=False):

        img = cv2.imread(path)

        roi, box = self.detect_stack(img)

        count = self.count_sheets(roi, debug)

        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.putText(
            img,
            f"sheets={count}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        return img, count


# ===============================
# MAIN
# ===============================
if __name__ == "__main__":

    counter = MetalStackCounter("best.pt")

    img, count = counter.process_image("res/photos/q2.jpg", debug=True)

    print("Sheets:", count)

    cv2.imshow("result", img)
    cv2.waitKey(0)