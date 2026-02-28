import cv2
import numpy as np
import matplotlib.pyplot as plt


class SheetBackgroundRemover:
    """
    Removes remaining background inside already extracted sheet region.
    Keeps only stacked metal sheet block.
    Designed for industrial robustness:
    - glare tolerant
    - angle tolerant
    - perspective tolerant
    """

    def __init__(self, debug=False):
        self.debug = debug

    # ---------------------------------------------------------
    # MAIN FUNCTION
    # ---------------------------------------------------------
    def remove_background(self, img_rgb):
        """
        Input:
            img_rgb – already extracted region (RGB)

        Output:
            cleaned_rgb – background removed image
        """

        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1️⃣ Strong edge detection (sheets create dense horizontal lines)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel_y = np.abs(sobel_y)
        sobel_y = cv2.normalize(sobel_y, None, 0, 255, cv2.NORM_MINMAX)
        sobel_y = sobel_y.astype(np.uint8)

        # 2️⃣ Threshold to keep structured zone
        _, binary = cv2.threshold(
            sobel_y, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # 3️⃣ Morphological closing to merge sheet structure
        kernel = np.ones((5, 15), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 4️⃣ Remove small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)

        cleaned_mask = np.zeros_like(binary)

        min_area = img.shape[0] * img.shape[1] * 0.05  # 5% of region

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area > min_area:
                cleaned_mask[labels == i] = 255

        # 5️⃣ Convex hull for solid block
        contours, _ = cv2.findContours(
            cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) > 0:
            largest = max(contours, key=cv2.contourArea)
            hull = cv2.convexHull(largest)
            final_mask = np.zeros_like(cleaned_mask)
            cv2.drawContours(final_mask, [hull], -1, 255, -1)
        else:
            final_mask = cleaned_mask

        # 6️⃣ Apply mask
        cleaned = cv2.bitwise_and(img, img, mask=final_mask)
        cleaned_rgb = cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB)

        if self.debug:
            self._debug_visualization(img_rgb, sobel_y, binary,
                                      cleaned_mask, final_mask, cleaned_rgb)

        return cleaned_rgb

    # ---------------------------------------------------------
    # DEBUG
    # ---------------------------------------------------------
    def _debug_visualization(self, original, sobel, binary,
                             cleaned_mask, final_mask, result):

        plt.figure(figsize=(18, 8))

        plt.subplot(2, 3, 1)
        plt.imshow(original)
        plt.title("Extracted region")

        plt.subplot(2, 3, 2)
        plt.imshow(sobel, cmap="gray")
        plt.title("Vertical gradient (Sobel Y)")

        plt.subplot(2, 3, 3)
        plt.imshow(binary, cmap="gray")
        plt.title("Binary (Otsu)")

        plt.subplot(2, 3, 4)
        plt.imshow(cleaned_mask, cmap="gray")
        plt.title("Large components")

        plt.subplot(2, 3, 5)
        plt.imshow(final_mask, cmap="gray")
        plt.title("Convex hull mask")

        plt.subplot(2, 3, 6)
        plt.imshow(result)
        plt.title("Final cleaned")

        plt.tight_layout()
        plt.show()