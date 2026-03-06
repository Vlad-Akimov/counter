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

    def __init__(self, debug=True, columns=40, segments=20):
        self.debug = debug
        self.columns = columns
        self.segments = segments

    def load_image(self, image_input):
        if isinstance(image_input, str):
            img = cv2.imread(image_input)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            return image_input

    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        return clahe.apply(gray)

    def find_vertical_bounds(self, gray):
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        row_energy = np.mean(sobel_y, axis=1)
        row_energy /= (row_energy.max() + 1e-6)

        mask = row_energy > 0.3
        if np.sum(mask) < 10:
            return 0, gray.shape[0]

        ys = np.where(mask)[0]
        return ys[0], ys[-1]

    def column_signals(self, gray):
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        h, w = sobel_y.shape
        xs = np.linspace(int(w*0.05), int(w*0.95), self.columns).astype(int)

        signals = []

        for x in xs:
            col = sobel_y[:, x]
            if np.mean(col) < 5:
                continue

            col = col.astype(np.float32)
            col -= col.min()
            if col.max() <= 0:
                continue

            col /= col.max()

            trend = cv2.GaussianBlur(col.reshape(-1,1), (1,301), 0).ravel()
            col -= trend
            col -= col.min()

            if col.max() <= 0:
                continue

            col /= col.max()

            if np.std(col) < 0.05:
                continue

            signals.append(col)

        return np.array(signals)

    def dominant_period(self, signal):
        N = len(signal)
        fft = np.fft.rfft(signal)
        spec = np.abs(fft)
        freqs = np.fft.rfftfreq(N)

        mask = (freqs > 0.02) & (freqs < 0.5)
        if not np.any(mask):
            return None, 0

        freqs = freqs[mask]
        spec = spec[mask]

        idx = np.argmax(spec)
        peak = spec[idx]
        mean_spec = np.mean(spec)

        if peak < 3 * mean_spec:
            return None, 0

        freq = freqs[idx]
        if freq <= 0:
            return None, 0

        period = 1.0 / freq
        quality = peak / mean_spec

        return period, quality

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

        median_period = np.median(periods)
        count = int(gray.shape[0] / median_period)

        mad = np.median(np.abs(periods - median_period))
        confidence = np.mean(qualities) / (1 + mad)

        return count, confidence, median_period, (y1, y2)

    def count(self, image_input):
        """
        Подсчитывает количество листов металла на изображении
        
        Args:
            image_input: либо путь к изображению (str), либо уже загруженное изображение (numpy array)
        
        Returns:
            CountResult: результат подсчета
        """
        
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
            return CountResult(0,0,segment_counts,
                               segment_confidences,0,0)

        mean_count = np.mean(valid_counts)
        median_count = np.median(valid_counts)
        overall_conf = np.mean([c for c in segment_confidences if c > 0])

        if self.debug:
            self.visualize(img, segment_periods, segment_bounds,
                           median_count)

        return CountResult(
            sheet_count=int(median_count),
            confidence=overall_conf,
            segment_counts=segment_counts,
            segment_confidences=segment_confidences,
            mean_count=mean_count,
            median_count=median_count
        )

    def visualize(self, img, periods, bounds, final_count):

        plt.figure(figsize=(10,10))
        plt.imshow(img)

        h, w = img.shape[:2]
        segment_width = w // self.segments
        colors = plt.cm.tab10(np.linspace(0,1,self.segments))

        for i in range(self.segments):

            x1 = i * segment_width
            x2 = w if i == self.segments-1 else (i+1) * segment_width

            plt.axvline(x1, color=colors[i], linewidth=2)

            period = periods[i]
            bound = bounds[i]

            if period is None or bound is None:
                continue

            y1, y2 = bound

            step = int(period)
            for y in range(y1, y2, step):
                plt.hlines(y, x1, x2,
                           colors=[colors[i]],
                           linewidth=1,
                           alpha=0.4)

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
    SEGMENT_COUNT = 16
    
    for i in range(59, 60):
        p = f"res/photos/q{i}.jpg"
        try:
            process_image(p, True, SEGMENT_COUNT)
        except Exception as e:
            print(f"Error {p}: {e}")