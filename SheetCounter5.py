import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# INDUSTRIAL v5 METAL SHEET COUNTER
# 2D frequency voting + subpixel period refinement
# Maximum robustness for 500+ ultra-thin sheets
# ============================================================

@dataclass
class CountResult:
    sheet_count: int
    period_px: float
    confidence: float
    profile: np.ndarray


class MetalSheetCounterV5:
    """
    v5 strategy:
    - No rotation
    - No global-only FFT
    - Dense column sampling
    - Per-column FFT
    - Subpixel peak interpolation
    - Robust median + confidence metric
    """

    def __init__(self, debug=True, columns=80):
        self.debug = debug
        self.columns = columns

    # ---------------------------------------------------------
    # PREPROCESS
    # ---------------------------------------------------------
    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        return clahe.apply(gray)

    # ---------------------------------------------------------
    # BUILD COLUMN SIGNALS
    # ---------------------------------------------------------
    def column_signals(self, gray):
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        h, w = sobel_y.shape

        xs = np.linspace(int(w*0.05), int(w*0.95), self.columns).astype(int)
        signals = []

        for x in xs:
            col = sobel_y[:, x]
            col -= col.min()
            if col.max() > 0:
                col /= col.max()

            # remove slow illumination drift
            trend = cv2.GaussianBlur(col.reshape(-1,1), (1,301), 0).ravel()
            col = col - trend
            col -= col.min()
            if col.max() > 0:
                col /= col.max()

            signals.append(col)

        return np.array(signals)

    # ---------------------------------------------------------
    # SUBPIXEL FFT PEAK
    # ---------------------------------------------------------
    def dominant_period(self, signal):
        N = len(signal)
        fft = np.fft.rfft(signal)
        spec = np.abs(fft)
        freqs = np.fft.rfftfreq(N)

        mask = (freqs > 0.02) & (freqs < 0.5)
        if not np.any(mask):
            return None

        freqs = freqs[mask]
        spec = spec[mask]

        idx = np.argmax(spec)

        # --- Subpixel quadratic interpolation ---
        if 1 <= idx < len(spec)-1:
            y0, y1, y2 = spec[idx-1], spec[idx], spec[idx+1]
            denom = (y0 - 2*y1 + y2)
            if denom != 0:
                delta = 0.5 * (y0 - y2) / denom
                idx = idx + delta

        peak_freq = freqs[int(np.clip(idx,0,len(freqs)-1))]
        if peak_freq <= 0:
            return None

        return 1.0 / peak_freq

    # ---------------------------------------------------------
    # MAIN COUNT
    # ---------------------------------------------------------
    def count(self, img):
        gray = self.preprocess(img)
        signals = self.column_signals(gray)

        periods = []
        for s in signals:
            p = self.dominant_period(s)
            if p is not None:
                periods.append(p)

        periods = np.array(periods)

        if len(periods) == 0:
            return CountResult(0, 0, 0, None)

        median_period = np.median(periods)
        count = int(len(signals[0]) / median_period)

        # confidence = how tight the voting is
        mad = np.median(np.abs(periods - median_period))
        confidence = 1.0 / (1.0 + mad)

        # build average profile for debug
        profile = np.mean(signals, axis=0)

        if self.debug:
            self.visualize(img, profile, median_period, count, periods)

        return CountResult(count, median_period, confidence, profile)

    # ---------------------------------------------------------
    # DEBUG
    # ---------------------------------------------------------
    def visualize(self, img, profile, period, count, periods):
        plt.figure(figsize=(16,10))

        plt.subplot(2,2,1)
        plt.imshow(img)
        step = int(period)
        if step>0:
            for y in range(0,img.shape[0],step):
                plt.axhline(y,color='r',alpha=0.12)
        plt.title(f"count={count}")

        plt.subplot(2,2,2)
        plt.plot(profile)
        plt.title("average profile")

        plt.subplot(2,2,3)
        plt.hist(periods,bins=25)
        plt.title("column period voting")

        plt.tight_layout()
        plt.show()


# ============================================================
# INTEGRATION
# ============================================================

from test7 import MetalSheetExtractor


def process_image(image_path, debug=True):
    extractor = MetalSheetExtractor(debug=debug)
    extracted, _ = extractor.extract_sheet_region(image_path)

    counter = MetalSheetCounterV5(debug=debug)
    res = counter.count(extracted)

    print("\n======= V5 RESULT =======")
    print("Sheets:", res.sheet_count)
    print("Period(px):", round(res.period_px,2))
    print("Confidence:", round(res.confidence,4))

    return res


if __name__ == '__main__':
    for i in range(1,22):
        p = f"res/photos/q{i}.jpg"
        try:
            process_image(p, True)
        except Exception as e:
            print("ERROR", e)
