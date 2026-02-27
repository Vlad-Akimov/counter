import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import windows
from dataclasses import dataclass
from typing import Tuple

# ===============================
# Sheet counting module
# ===============================

@dataclass
class CountResult:
    sheet_count: int
    peak_positions: np.ndarray
    profile: np.ndarray


class MetalSheetCounter:
    """
    Metal sheet counter.

    Idea:
    - Sheets create horizontal boundaries (gaps)
    - Detect horizontal gradients
    - Collapse to 1D vertical profile
    - Smooth + autocorrelation to estimate spacing
    - Peak detection
    """

    def __init__(
        self,
        gaussian_sigma: int = 5,
        prominence_ratio: float = 0.25,
        debug: bool = False
    ):
        self.gaussian_sigma = gaussian_sigma
        self.prominence_ratio = prominence_ratio
        self.debug = debug

    # ===============================
    # Preprocess extracted sheet region
    # ===============================
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        return gray

    # ===============================
    # Build vertical edge profile
    # ===============================
    def build_profile(self, gray: np.ndarray) -> np.ndarray:
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)

        profile = np.mean(sobel_y, axis=1)

        profile -= profile.min()
        if profile.max() > 0:
            profile /= profile.max()

        # Smooth using gaussian window convolution
        win_size = int(max(7, self.gaussian_sigma * 6))
        if win_size % 2 == 0:
            win_size += 1
        g = windows.gaussian(win_size, std=self.gaussian_sigma)
        g /= g.sum()
        profile = np.convolve(profile, g, mode='same')

        return profile

    # ===============================
    # Estimate sheet spacing automatically
    # ===============================
    def estimate_spacing(self, profile: np.ndarray) -> int:
        corr = signal.correlate(profile, profile, mode='full')
        corr = corr[len(corr)//2:]

        corr[:10] = 0
        peaks, _ = signal.find_peaks(corr)

        if len(peaks) == 0:
            return max(10, len(profile)//40)

        return int(max(10, peaks[0]))

    # ===============================
    # Peak detection
    # ===============================
    def detect_peaks(self, profile: np.ndarray, spacing: int):
        prominence = self.prominence_ratio * np.max(profile)

        peaks, props = signal.find_peaks(
            profile,
            distance=spacing,
            prominence=prominence
        )

        return peaks, props

    # ===============================
    # Main counting method
    # ===============================
    def count(self, extracted_region: np.ndarray) -> CountResult:
        gray = self.preprocess(extracted_region)
        profile = self.build_profile(gray)
        spacing = self.estimate_spacing(profile)
        peaks, props = self.detect_peaks(profile, spacing)

        sheet_count = len(peaks)

        if self.debug:
            self._visualize(extracted_region, profile, peaks, spacing)

        return CountResult(sheet_count, peaks, profile)

    # ===============================
    # Debug visualization
    # ===============================
    def _visualize(self, img, profile, peaks, spacing):
        plt.figure(figsize=(14, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        for p in peaks:
            plt.axhline(p, color='r', alpha=0.35)
        plt.title(f"Detected sheets (spacing~{spacing}px)")

        plt.subplot(1, 2, 2)
        plt.plot(profile)
        plt.scatter(peaks, profile[peaks])
        plt.title("Vertical gradient profile")

        plt.tight_layout()
        plt.show()


# ===============================
# Integration with extractor
# ===============================

from test7 import MetalSheetExtractor


def process_image(image_path: str, debug=True):
    # IMPORTANT — enable extractor debug to show its figures
    extractor = MetalSheetExtractor(debug=debug)
    extracted, region = extractor.extract_sheet_region(image_path)

    counter = MetalSheetCounter(debug=debug)
    result = counter.count(extracted)

    print("\n======= COUNT RESULT =======")
    print("Sheets detected:", result.sheet_count)

    return result


if __name__ == "__main__":
    for i in range(1, 22):
        path = f"res/photos/q{i}.jpg"
        try:
            process_image(path, debug=True)
        except Exception as e:
            print("ERROR:", e)
