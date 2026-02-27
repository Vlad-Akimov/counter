import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from dataclasses import dataclass
from typing import Tuple

# ===============================
# Sheet counting module
# Works on already extracted region with sheets
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
        gaussian_sigma: int = 3,
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

        # CLAHE improves metal texture contrast
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Light blur
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        return gray

    # ===============================
    # Build vertical edge profile
    # ===============================
    def build_profile(self, gray: np.ndarray) -> np.ndarray:
        # Horizontal edges -> Sobel Y
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)

        # Collapse width → 1D signal
        profile = np.mean(sobel_y, axis=1)

        # Normalize
        profile -= profile.min()
        if profile.max() > 0:
            profile /= profile.max()

        # Smooth
        profile = signal.gaussian(len(profile), std=self.gaussian_sigma) * 0 + profile
        profile = signal.savgol_filter(profile, 21 if len(profile) > 21 else 7, 3)

        return profile

    # ===============================
    # Estimate sheet spacing automatically
    # ===============================
    def estimate_spacing(self, profile: np.ndarray) -> int:
        corr = signal.correlate(profile, profile, mode='full')
        corr = corr[len(corr)//2:]

        # Ignore zero lag
        corr[0:5] = 0

        peaks, _ = signal.find_peaks(corr)
        if len(peaks) == 0:
            return max(5, len(profile)//50)

        spacing = peaks[0]
        spacing = int(max(5, spacing))
        return spacing

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
            self._visualize(extracted_region, gray, profile, peaks)

        return CountResult(sheet_count, peaks, profile)

    # ===============================
    # Debug visualization
    # ===============================
    def _visualize(self, img, gray, profile, peaks):
        plt.figure(figsize=(12, 8))

        plt.subplot(1, 2, 1)
        plt.imshow(img)
        for p in peaks:
            plt.axhline(p, color='r', alpha=0.4)
        plt.title("Detected sheet boundaries")

        plt.subplot(1, 2, 2)
        plt.plot(profile)
        plt.scatter(peaks, profile[peaks], color='red')
        plt.title("Vertical gradient profile")

        plt.tight_layout()
        plt.show()


# ===============================
# Integration example with extractor
# ===============================

from test7 import MetalSheetExtractor


def process_image(image_path: str, debug=True):
    extractor = MetalSheetExtractor(debug=False)
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
