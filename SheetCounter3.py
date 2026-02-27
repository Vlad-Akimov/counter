import cv2
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# ============================================================
# INDUSTRIAL v2 METAL SHEET COUNTER (FFT BASED)
# Designed for hundreds of very thin sheets
# ============================================================

@dataclass
class CountResult:
    sheet_count: int
    period_px: float
    profile: np.ndarray
    spectrum: np.ndarray


class MetalSheetCounterV2:
    """
    Industrial sheet counter.

    Core idea:
    Hundreds of thin sheets create strong periodic structure.
    Peak detection fails → use frequency analysis.

    Pipeline:
    - Sobel Y (horizontal boundaries)
    - Multi-column sampling
    - Build high-frequency vertical profile
    - FFT → dominant spatial frequency
    - Convert frequency → sheet count
    """

    def __init__(self, debug=True, columns=25):
        self.debug = debug
        self.columns = columns

    # ---------------------------------------------------------
    # Preprocess (minimal smoothing — preserve micro texture)
    # ---------------------------------------------------------
    def preprocess(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # mild CLAHE only
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray

    # ---------------------------------------------------------
    # Build high-resolution vertical signal
    # ---------------------------------------------------------
    def build_profile(self, gray):
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)

        h, w = sobel_y.shape

        # multi-column sampling instead of full average
        xs = np.linspace(int(w*0.2), int(w*0.8), self.columns).astype(int)
        signals = []

        for x in xs:
            col = sobel_y[:, x]
            col = (col - col.min())
            if col.max() > 0:
                col /= col.max()
            signals.append(col)

        profile = np.mean(signals, axis=0)

        # remove low-frequency illumination drift
        profile = profile - cv2.GaussianBlur(profile.reshape(-1,1), (1,301), 0).ravel()

        profile -= profile.min()
        if profile.max() > 0:
            profile /= profile.max()

        return profile

    # ---------------------------------------------------------
    # FFT counting
    # ---------------------------------------------------------
    def count_fft(self, profile):
        N = len(profile)

        fft = np.fft.rfft(profile)
        spectrum = np.abs(fft)
        freqs = np.fft.rfftfreq(N)

        # ignore DC and ultra-low
        low_cut = 0.01
        mask = freqs > low_cut

        peak_idx = np.argmax(spectrum[mask])
        peak_freq = freqs[mask][peak_idx]

        period = 1.0 / peak_freq if peak_freq > 0 else N
        sheet_count = int(N / period)

        return sheet_count, period, spectrum

    # ---------------------------------------------------------
    # MAIN
    # ---------------------------------------------------------
    def count(self, extracted_region):
        gray = self.preprocess(extracted_region)
        profile = self.build_profile(gray)
        count, period, spectrum = self.count_fft(profile)

        if self.debug:
            self.visualize(extracted_region, profile, spectrum, period, count)

        return CountResult(count, period, profile, spectrum)

    # ---------------------------------------------------------
    # DEBUG VISUALIZATION
    # ---------------------------------------------------------
    def visualize(self, img, profile, spectrum, period, count):
        plt.figure(figsize=(16,10))

        plt.subplot(2,2,1)
        plt.imshow(img)
        step = int(period)
        if step>0:
            for y in range(0, img.shape[0], step):
                plt.axhline(y, color='r', alpha=0.15)
        plt.title(f"Estimated sheets: {count}")

        plt.subplot(2,2,2)
        plt.plot(profile)
        plt.title("High-frequency vertical profile")

        plt.subplot(2,2,3)
        plt.plot(spectrum)
        plt.title("FFT spectrum")

        plt.tight_layout()
        plt.show()


# ============================================================
# INTEGRATION WITH EXISTING EXTRACTOR
# ============================================================

from test7 import MetalSheetExtractor


def process_image(image_path, debug=True):
    extractor = MetalSheetExtractor(debug=debug)
    extracted, region = extractor.extract_sheet_region(image_path)

    counter = MetalSheetCounterV2(debug=debug)
    result = counter.count(extracted)

    print("\n======= FFT COUNT RESULT =======")
    print("Sheets detected:", result.sheet_count)
    print("Estimated period(px):", round(result.period_px,2))

    return result


if __name__ == '__main__':
    for i in range(1,22):
        path = f"res/photos/q{i}.jpg"
        try:
            process_image(path, debug=True)
        except Exception as e:
            print("ERROR:", e)
