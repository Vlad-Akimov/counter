import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import os


@dataclass
class CountResult:
    sheet_count: int
    confidence: float
    segment_counts: list
    segment_confidences: list
    mean_count: float
    median_count: float


class MetalSheetCounter:

    def __init__(self, debug=True, columns=60, segments=40):
        self.debug = debug
        self.columns = columns
        self.segments = segments

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    def load_image(self, image_input):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        return image_input

    # -----------------------------------------------------
    # Preprocessing
    # -----------------------------------------------------

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # CLAHE for lighting normalization
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Mild smoothing (not too strong!)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        return gray

    # -----------------------------------------------------
    # Find vertical bounds
    # -----------------------------------------------------

    def find_vertical_bounds(self, gray):
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        row_energy = np.mean(sobel_y, axis=1)

        row_energy /= (row_energy.max() + 1e-6)

        mask = row_energy > 0.25

        if np.sum(mask) < 10:
            return 0, gray.shape[0]

        ys = np.where(mask)[0]
        return ys[0], ys[-1]

    # -----------------------------------------------------
    # Extract column signals
    # -----------------------------------------------------

    def column_signals(self, gray):

        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))

        h, w = sobel_y.shape
        xs = np.linspace(int(w*0.1), int(w*0.9), self.columns).astype(int)

        signals = []

        for x in xs:

            col = sobel_y[:, x].astype(np.float32)

            if np.mean(col) < 5:
                continue

            col -= col.mean()

            if np.std(col) < 1e-3:
                continue

            col /= (np.std(col) + 1e-6)

            signals.append(col)

        return np.array(signals)

    # -----------------------------------------------------
    # Dominant period using stabilized FFT
    # -----------------------------------------------------

    def dominant_period(self, signal):

        N = len(signal)

        # Autocorrelation (устойчивее FFT к гармоникам)
        corr = np.correlate(signal, signal, mode='full')
        corr = corr[N-1:]

        corr[0] = 0  # убираем нулевой лаг

        # Минимальный допустимый период (защита от переучёта)
        min_period = 8     # ← можно подстроить
        max_period = N // 2

        search_region = corr[min_period:max_period]

        if len(search_region) == 0:
            return None, 0

        peak_idx = np.argmax(search_region)
        period = peak_idx + min_period

        peak_value = search_region[peak_idx]
        mean_value = np.mean(search_region)

        if peak_value < 2.0 * mean_value:
            return None, 0

        quality = peak_value / (mean_value + 1e-6)

        return period, quality

    # -----------------------------------------------------
    # Count single segment
    # -----------------------------------------------------

    def count_single(self, gray):

        y1, y2 = self.find_vertical_bounds(gray)
        gray = gray[y1:y2, :]

        if gray.shape[0] < 50:
            return 0, 0, None, None

        signals = self.column_signals(gray)

        if len(signals) < 5:
            return 0, 0, None, None

        periods = []
        qualities = []

        for s in signals:
            p, q = self.dominant_period(s)
            if p is not None:
                periods.append(p)
                qualities.append(q)

        if len(periods) < 3:
            return 0, 0, None, None

        periods = np.array(periods)
        qualities = np.array(qualities)

        # Remove outliers using MAD
        median_period = np.median(periods)
        mad = np.median(np.abs(periods - median_period)) + 1e-6

        mask = np.abs(periods - median_period) < 2.5 * mad

        periods = periods[mask]
        qualities = qualities[mask]

        if len(periods) < 3:
            return 0, 0, None, None

        # Weighted median
        weighted_period = np.average(periods, weights=qualities)

        count = int(round(gray.shape[0] / weighted_period))
        count = max(1, min(count, 400))

        confidence = np.mean(qualities) / (1 + mad)

        return count, confidence, weighted_period, (y1, y2)

    # -----------------------------------------------------
    # Main count
    # -----------------------------------------------------

    def count(self, image_input):

        img = self.load_image(image_input)
        gray_full = self.preprocess(img)

        h, w = gray_full.shape
        segment_width = w // self.segments

        segment_counts = []
        segment_confidences = []
        segment_periods = []
        segment_bounds = []

        for i in range(self.segments):

            x1 = i * segment_width
            x2 = w if i == self.segments-1 else (i+1) * segment_width

            segment = gray_full[:, x1:x2]

            count, conf, period, bounds = self.count_single(segment)

            segment_counts.append(count)
            segment_confidences.append(conf)
            segment_periods.append(period)
            segment_bounds.append(bounds)

            if self.debug:
                print(f"Segment {i+1}: count={count}, conf={round(conf,3)}")

        valid_counts = [c for c in segment_counts if c > 0]

        if len(valid_counts) == 0:
            return CountResult(0, 0, segment_counts,
                               segment_confidences, 0, 0)

        mean_count = np.mean(valid_counts)
        median_count = np.median(valid_counts)

        overall_conf = np.mean(
            [c for c in segment_confidences if c > 0]
        )

        if self.debug:
            self.visualize(img, segment_periods,
                           segment_bounds, median_count)

        return CountResult(
            sheet_count=int(median_count),
            confidence=float(overall_conf),
            segment_counts=segment_counts,
            segment_confidences=segment_confidences,
            mean_count=float(mean_count),
            median_count=float(median_count)
        )

    # -----------------------------------------------------
    # Visualization (RESTORED)
    # -----------------------------------------------------

    def visualize(self, img, periods, bounds, final_count):

        plt.figure(figsize=(10, 10))
        plt.imshow(img)

        h, w = img.shape[:2]
        segment_width = w // self.segments
        colors = plt.cm.tab20(np.linspace(0, 1, self.segments))

        for i in range(self.segments):

            x1 = i * segment_width
            x2 = w if i == self.segments-1 else (i+1) * segment_width

            plt.axvline(x1, color=colors[i], linewidth=1)

            period = periods[i]
            bound = bounds[i]

            if period is None or bound is None:
                continue

            y1, y2 = bound
            step = int(period)

            for y in range(y1, y2, step):
                plt.hlines(
                    y,
                    x1,
                    x2,
                    colors=[colors[i]],
                    linewidth=1,
                    alpha=0.5
                )

        plt.title(f"Final count (median) = {int(final_count)}")
        plt.tight_layout()
        plt.show()


def process_image(image_path, debug, segments):
    """
    Обрабатывает изображение и подсчитывает количество листов металла
    
    Args:
        image_path (str): путь к изображению
        debug (bool): режим отладки (показывает визуализацию)
        segments (int): количество сегментов для анализа
    
    Returns:
        CountResult: результат подсчета
    """
    
    counter = MetalSheetCounter(debug=debug, segments=segments)
    
    result = counter.count(image_path)
    
    print("\n======= FINAL RESULT =======")
    print(f"file: {os.path.basename(image_path)}")
    print(f"Final sheets (median): {result.sheet_count}")
    print(f"Mean: {round(result.mean_count, 2)}")
    print(f"Overall confidence: {round(result.confidence, 4)}")
    print(f"Segments: {result.segment_counts}")
    print("===============================\n")
    
    return result


if __name__ == '__main__':
    SEGMENT_COUNT = 64
    
    for i in range(32, 59):
        p = f"res/photos/q{i}.jpg"
        try:
            process_image(p, True, SEGMENT_COUNT)
        except Exception as e:
            print(f"Error {p}: {e}")