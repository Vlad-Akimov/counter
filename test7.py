import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import os
from dataclasses import dataclass
from typing import Tuple, Optional

@dataclass
class SheetRegion:
    """Data class for storing sheet region information"""
    coordinates: Tuple[int, int, int, int]  # x1, y1, x2, y2
    size: Tuple[int, int]  # width, height
    density_score: float
    area_percentage: float

class MetalSheetExtractor:
    """
    Metal Sheet Extractor - Extracts regions containing stacked metal sheets
    from photographs using edge density analysis.
    """
    
    def __init__(self, density_threshold: float = 0.6, debug: bool = False):
        """
        Initialize the Metal Sheet Extractor.
        
        Args:
            density_threshold: Fraction of maximum edge density for sheet region (default: 0.6)
            debug: Enable debug visualization (default: False)
        """
        self.density_threshold = density_threshold
        self.debug = debug
    
    def reduce_glare(self, img):
        """Simple glare reduction function"""
        # Convert to HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Reduce the intensity of very bright areas (glare)
        v = np.where(v > 200, v * 0.7, v).astype(np.uint8)
        
        # Merge back
        hsv_corrected = cv2.merge([h, s, v])
        result = cv2.cvtColor(hsv_corrected, cv2.COLOR_HSV2BGR)
        
        return result
        
    def extract_sheet_region(self, image_path: str, padding: int = 10) -> Tuple[np.ndarray, SheetRegion]:
        """
        Extract the region containing metal sheets from an image.
        
        Args:
            image_path: Path to the input image
            padding: Additional padding around the detected region (pixels)
            
        Returns:
            Tuple containing:
                - Extracted image region
                - SheetRegion object with metadata
            
        Raises:
            FileNotFoundError: If image cannot be loaded
        """
        print(f"\n{'='*60}")
        print("METAL SHEET REGION EXTRACTION")
        print('='*60)
        
        # Step 0: Load and validate image
        print(f"\n[1/5] Loading image: {os.path.basename(image_path)}")
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        
        # Add glare reduction step
        print("    Applying glare reduction...")
        img = self.reduce_glare(img)
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h_orig, w_orig = img.shape[:2]
        print(f"    Image size: {w_orig} x {h_orig} pixels")
        
        # Step 1: Image preprocessing
        print("\n[2/5] Preprocessing image...")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Denoise with median filter
        denoised = cv2.medianBlur(enhanced, 5)
        
        # Step 2: Edge detection and texture analysis
        print("[3/5] Analyzing metal texture...")
        
        # Calculate gradients
        sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude
        gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))
        
        # Adaptive thresholding
        _, edge_binary = cv2.threshold(gradient_magnitude, 0, 255, 
                                       cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = np.ones((3, 3), np.uint8)
        edge_binary = cv2.morphologyEx(edge_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        edge_binary = cv2.morphologyEx(edge_binary, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Step 3: Density map calculation
        print(f"[4/5] Computing edge density map...")
        
        # Calculate density map
        window_size = min(h_orig, w_orig) // 15
        edge_float = edge_binary.astype(np.float32) / 255.0
        
        density_kernel = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
        density_map = cv2.filter2D(edge_float, -1, density_kernel)
        
        # Find maximum density value
        max_density = np.max(density_map)
        print(f"    Maximum edge density: {max_density:.3f}")
        
        # Calculate adaptive threshold as fraction of maximum
        adaptive_threshold = self.density_threshold * max_density
        print(f"    Using adaptive threshold: {adaptive_threshold:.3f} ({self.density_threshold*100:.0f}% of max)")
        
        # Find regions above adaptive threshold
        high_density_mask = density_map > adaptive_threshold
        
        if not np.any(high_density_mask):
            print("    WARNING: No region found above density threshold!")
            print("    Using center region as fallback...")
            # Fallback to center region
            center_x, center_y = w_orig // 2, h_orig // 2
            region_size = min(w_orig, h_orig) // 2
            x1 = max(0, center_x - region_size // 2)
            y1 = max(0, center_y - region_size // 2)
            x2 = min(w_orig, center_x + region_size // 2)
            y2 = min(h_orig, center_y + region_size // 2)
            density_score = 0.0
        else:
            # Get bounding box of high-density region
            y_indices, x_indices = np.where(high_density_mask)
            
            # Calculate density score (average density in region)
            density_score = np.mean(density_map[high_density_mask])
            
            # Add padding
            padding_density = window_size
            x1 = max(0, np.min(x_indices) - padding_density)
            y1 = max(0, np.min(y_indices) - padding_density)
            x2 = min(density_map.shape[1] - 1, np.max(x_indices) + padding_density)
            y2 = min(density_map.shape[0] - 1, np.max(y_indices) + padding_density)
            
            # Scale coordinates back to original image
            scale_x = w_orig / density_map.shape[1]
            scale_y = h_orig / density_map.shape[0]
            
            x1 = int(x1 * scale_x)
            y1 = int(y1 * scale_y)
            x2 = int(x2 * scale_x)
            y2 = int(y2 * scale_y)
        
        # Step 4: Refine boundaries
        print("[5/5] Finalizing region boundaries...")
        
        # Ensure coordinates are within image bounds
        x1 = max(0, min(x1, w_orig - 1))
        y1 = max(0, min(y1, h_orig - 1))
        x2 = max(x1 + 1, min(x2, w_orig))
        y2 = max(y1 + 1, min(y2, h_orig))
        
        # Extract the final region
        extracted = img_rgb[y1:y2, x1:x2]
        
        # Calculate metadata
        width = x2 - x1
        height = y2 - y1
        area_percentage = (width * height) / (w_orig * h_orig) * 100
        
        sheet_region = SheetRegion(
            coordinates=(x1, y1, x2, y2),
            size=(width, height),
            density_score=density_score,
            area_percentage=area_percentage
        )
        
        # Debug visualization
        if self.debug:
            self._visualize_debug(img_rgb, density_map, high_density_mask, 
                                 sheet_region, edge_binary, gradient_magnitude, adaptive_threshold)
        
        print(f"\n{'='*60}")
        print("EXTRACTION RESULTS")
        print('='*60)
        print(f"Region coordinates: ({x1}, {y1}) - ({x2}, {y2})")
        print(f"Region size: {width} x {height} pixels")
        print(f"Edge density score: {density_score:.3f}")
        print(f"Adaptive threshold used: {adaptive_threshold:.3f} ({self.density_threshold*100:.0f}% of max={max_density:.3f})")
        print(f"Region covers {area_percentage:.1f}% of original image")
        print('='*60)
        
        return extracted, sheet_region
    
    def _visualize_debug(self, original_img: np.ndarray, density_map: np.ndarray,
                        high_density_mask: np.ndarray, region: SheetRegion,
                        edge_binary: np.ndarray, gradient_magnitude: np.ndarray,
                        adaptive_threshold: float):
        """Visualize intermediate processing steps for debugging."""
        
        x1, y1, x2, y2 = region.coordinates
        
        plt.figure(figsize=(18, 10))
        
        # Original with detected region
        plt.subplot(2, 4, 1)
        plt.imshow(original_img)
        plt.title('Original Image')
        plt.axis('on')
        
        # Edge map
        plt.subplot(2, 4, 2)
        plt.imshow(edge_binary, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')
        
        # Gradient magnitude
        plt.subplot(2, 4, 3)
        plt.imshow(gradient_magnitude, cmap='hot')
        plt.title('Gradient Magnitude')
        plt.axis('off')
        
        # Density map
        plt.subplot(2, 4, 4)
        im = plt.imshow(density_map, cmap='hot')
        plt.colorbar(im, label='Edge Density')
        plt.title(f'Edge Density Map (max: {np.max(density_map):.3f})')
        plt.axis('off')
        
        # High density mask
        plt.subplot(2, 4, 5)
        plt.imshow(high_density_mask, cmap='gray')
        plt.title(f'Density > {adaptive_threshold:.3f}\n({self.density_threshold*100:.0f}% of max)')
        plt.axis('off')
        
        # Original with bounding box
        plt.subplot(2, 4, 6)
        img_with_box = original_img.copy()
        cv2.rectangle(img_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
        plt.imshow(img_with_box)
        plt.title(f'Detected Region (density: {region.density_score:.3f})')
        plt.axis('on')
        
        # Extracted region
        plt.subplot(2, 4, 7)
        extracted = original_img[y1:y2, x1:x2]
        plt.imshow(extracted)
        plt.title(f'Extracted ({x2-x1} x {y2-y1})')
        plt.axis('on')
        
        # Final result with stats
        plt.subplot(2, 4, 8)
        plt.imshow(extracted)
        plt.title(f'Final Result\n{region.size[0]}x{region.size[1]} | '
                 f'{region.area_percentage:.1f}% of original')
        plt.axis('on')
        plt.text(5, 30, f'Density: {region.density_score:.3f}', 
                color='white', fontsize=10,
                bbox=dict(facecolor='black', alpha=0.7))
        
        plt.tight_layout()
        plt.show()
    
    def batch_process(self, image_paths: list, output_dir: str = 'extracted'):
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to images
            output_dir: Directory to save extracted regions
        """
        os.makedirs(output_dir, exist_ok=True)
        
        results = []
        for i, image_path in enumerate(image_paths, 1):
            print(f"\n[{i}/{len(image_paths)}] Processing: {os.path.basename(image_path)}")
            
            try:
                extracted, region = self.extract_sheet_region(image_path)
                
                # Save result
                output_path = os.path.join(output_dir, f"extracted_{os.path.basename(image_path)}")
                extracted_bgr = cv2.cvtColor(extracted, cv2.COLOR_RGB2BGR)
                cv2.imwrite(output_path, extracted_bgr)
                
                results.append({
                    'image': os.path.basename(image_path),
                    'region': region,
                    'output': output_path,
                    'success': True
                })
                
                print(f"  Saved to: {output_path}")
                
            except Exception as e:
                print(f"  Error: {str(e)}")
                results.append({
                    'image': os.path.basename(image_path),
                    'success': False,
                    'error': str(e)
                })
        
        # Print summary
        print(f"\n{'='*60}")
        print("BATCH PROCESSING SUMMARY")
        print('='*60)
        successful = [r for r in results if r['success']]
        print(f"Successfully processed: {len(successful)}/{len(image_paths)} images")
        
        for result in successful:
            r = result['region']
            print(f"  * {result['image']}: {r.size[0]}x{r.size[1]} "
                  f"(density: {r.density_score:.3f})")
        
        return results


def main(pat):
    # Configuration
    IMAGE_PATH = pat
    DENSITY_THRESHOLD_FRACTION = 0.6
    DEBUG_MODE = True
    
    try:
        # Initialize extractor
        MetalSheetExtractor(
            density_threshold=DENSITY_THRESHOLD_FRACTION,
            debug=DEBUG_MODE
        ).extract_sheet_region(
            image_path=IMAGE_PATH,
            padding=10
        )
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    for i in range(1, 22):
        a = f"res/photos/q{i}.jpg"
        main(a)