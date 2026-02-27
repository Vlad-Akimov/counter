import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# INDUSTRIAL v4 METAL SHEET COUNTER
# Stable orientation + local spectral voting
# (rotation removed — only vertical analysis)
# ============================================================

@dataclass
class CountResult:
    sheet_count: int
    period_px: float
    periods_local: np.ndarray
    profile: np.ndarray


class MetalSheetCounterV4:
    """
    v4 philosophy:
    - DO NOT rotate image (user confirmed orientation mostly correct)
    - DO NOT global FFT only (unstable)
    - Use local window FFT voting
    - Median period estimation (very robust)
    """

    def __init__(self, debug=True, columns=45, windows=12):
        self.debug = debug
        self.columns = columns
        self.windows = windows

    # ---------------------------------------------------------
    # PREPROCESS
    # ---------------------------------------------------------
    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(2.0, (8, 8))
        return clahe.apply(gray)

    # ---------------------------------------------------------
    # BUILD HIGH-FREQ PROFILE
    # ---------------------------------------------------------
    def build_profile(self, gray):
        sobel_y = np.abs(cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3))
        h, w = sobel_y.shape

        xs = np.linspace(int(w*0.1), int(w*0.9), self.columns).astype(int)
        signals = []

        for x in xs:
            col = sobel_y[:, x]
            col -= col.min()
            if col.max() > 0:
                col /= col.max()
            signals.append(col)

        profile = np.mean(signals, axis=0)

        # remove illumination drift
        trend = cv2.GaussianBlur(profile.reshape(-1, 1), (1, 301), 0).ravel()
        profile = profile - trend
        profile -= profile.min()
        if profile.max() > 0:
            profile /= profile.max()

        return profile

    # ---------------------------------------------------------
    # LOCAL FFT PERIOD ESTIMATION
    # ---------------------------------------------------------
    def local_periods(self, profile):
        N = len(profile)
        win = N // self.windows
        periods = []

        for i in range(self.windows):
            a = i * win
            b = (i + 1) * win if i < self.windows - 1 else N
            seg = profile[a:b]

            if len(seg) < 50:
                continue

            fft = np.fft.rfft(seg)
            spec = np.abs(fft)
            freqs = np.fft.rfftfreq(len(seg))

            mask = (freqs > 0.02) & (freqs < 0.5)
            if not np.any(mask):
                continue

            f = freqs[mask][np.argmax(spec[mask])]
            if f > 0:
                periods.append(1 / f)

        return np.array(periods)

    # ---------------------------------------------------------
    # MAIN COUNT
    # ---------------------------------------------------------
    def count(self, img):
        gray = self.preprocess(img)
        profile = self.build_profile(gray)

        periods = self.local_periods(profile)

        if len(periods) == 0:
            period = len(profile)
        else:
            period = np.median(periods)

        count = int(len(profile) / period) if period > 0 else 0

        if self.debug:
            self.visualize(img, profile, periods, period, count)

        return CountResult(count, period, periods, profile)

    # ---------------------------------------------------------
    # DEBUG
    # ---------------------------------------------------------
    def visualize(self, img, profile, periods, period, count):
        plt.figure(figsize=(16, 10))

        plt.subplot(2, 2, 1)
        plt.imshow(img)
        step = int(period)
        if step > 0:
            for y in range(0, img.shape[0], step):
                plt.axhline(y, color='r', alpha=0.15)
        plt.title(f"count={count}")

        plt.subplot(2, 2, 2)
        plt.plot(profile)
        plt.title("profile")

        plt.subplot(2, 2, 3)
        plt.hist(periods, bins=20)
        plt.title("local period voting")

        plt.tight_layout()
        plt.show()


# ============================================================
# INTEGRATION
# ============================================================

from test7 import MetalSheetExtractor


def process_image(image_path, debug=True):
    extractor = MetalSheetExtractor(debug=debug)
    extracted, _ = extractor.extract_sheet_region(image_path)

    counter = MetalSheetCounterV4(debug=debug)
    res = counter.count(extracted)

    print("\n======= V4 RESULT =======")
    print("Sheets:", res.sheet_count)
    print("Period:", round(res.period_px, 2))

    return res


if __name__ == '__main__':
    for i in range(1, 22):
        p = f"res/photos/q{i}.jpg"
        try:
            process_image(p, True)
        except Exception as e:
            print("ERROR", e)
