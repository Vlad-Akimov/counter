import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import os

def precise_sheet_extraction(image_path, debug=True, padding=5):
    """
    Функция для точного выделения области с металлическими листами.
    Выполняет:
    1. Умное выделение области с листами (минимально "не листов")
    2. Точную обрезку по границам листов
    
    Parameters:
    -----------
    image_path : str
        Путь к изображению
    debug : bool
        Если True, показывает промежуточные результаты
    padding : int
        Добавочные пиксели по краям (чтобы не обрезать слишком плотно)
        
    Returns:
    --------
    extracted_image : numpy.ndarray
        Точно выделенное изображение листов
    sheet_region : dict
        Информация о выделенной области
    """
    
    # Шаг 0: Загрузка изображения
    print(f"Загружаем изображение: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_original = img_rgb.copy()
    h_orig, w_orig = img.shape[:2]
    
    print(f"Размер изображения: {w_orig}x{h_orig} пикселей")
    
    # Шаг 1: Улучшение изображения для анализа
    print("\n=== Шаг 1: Улучшение изображения ===")
    
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Применяем CLAHE для улучшения локального контраста
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Медианный фильтр для удаления шума
    denoised = cv2.medianBlur(enhanced, 5)
    
    # Шаг 2: Выделение текстуры металла (анализ градиентов)
    print("\n=== Шаг 2: Анализ текстуры ===")
    
    # Вычисляем градиенты (подчеркивает границы листов)
    sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    
    # Величина градиента
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))
    
    # Бинаризация градиента (области с сильными градиентами - это границы листов)
    _, gradient_binary = cv2.threshold(gradient_magnitude, 30, 255, cv2.THRESH_BINARY)
    
    # Морфологические операции для объединения границ
    kernel = np.ones((3, 3), np.uint8)
    gradient_binary = cv2.morphologyEx(gradient_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    gradient_binary = cv2.morphologyEx(gradient_binary, cv2.MORPH_DILATE, kernel, iterations=1)
    
    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(denoised, cmap='gray')
        plt.title('Улучшенное изображение')
        plt.subplot(1, 3, 2)
        plt.imshow(gradient_magnitude, cmap='hot')
        plt.title('Карта градиентов')
        plt.subplot(1, 3, 3)
        plt.imshow(gradient_binary, cmap='gray')
        plt.title('Бинарные границы')
        plt.tight_layout()
        plt.show()
    
    # Шаг 3: Поиск области с наибольшей плотностью границ
    print("\n=== Шаг 3: Поиск области с листами ===")
    
    # Создаем скользящее окно для поиска области с максимальной плотностью границ
    window_size = min(h_orig, w_orig) // 10  # 10% от меньшей стороны
    stride = window_size // 4
    
    max_density = 0
    best_region = None
    
    # Создаем карту плотности
    density_map = np.zeros((h_orig // stride + 1, w_orig // stride + 1))
    
    for y in range(0, h_orig - window_size, stride):
        for x in range(0, w_orig - window_size, stride):
            # Считаем количество граничных пикселей в окне
            window = gradient_binary[y:y+window_size, x:x+window_size]
            density = np.sum(window) / 255.0 / (window_size * window_size)
            
            density_map[y//stride, x//stride] = density
            
            if density > max_density:
                max_density = density
                best_region = (x, y, x + window_size, y + window_size)
    
    if debug:
        # Визуализируем карту плотности
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(density_map, cmap='hot', interpolation='nearest')
        plt.colorbar(label='Плотность границ')
        plt.title('Карта плотности границ')
        
        # Показываем найденную область
        img_with_region = img_original.copy()
        if best_region:
            x1, y1, x2, y2 = best_region
            cv2.rectangle(img_with_region, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        plt.subplot(1, 2, 2)
        plt.imshow(img_with_region)
        plt.title('Найденная область с листами')
        plt.tight_layout()
        plt.show()
    
    # Шаг 4: Уточнение границ с помощью анализа проекций
    print("\n=== Шаг 4: Уточнение границ ===")
    
    if best_region:
        x1, y1, x2, y2 = best_region
    else:
        # Если не нашли, берем центральную часть
        x1, y1 = w_orig // 4, h_orig // 4
        x2, y2 = 3 * w_orig // 4, 3 * h_orig // 4
    
    # Расширяем область для анализа
    expand_x = (x2 - x1) // 2
    expand_y = (y2 - y1) // 2
    
    x1 = max(0, x1 - expand_x // 2)
    y1 = max(0, y1 - expand_y // 2)
    x2 = min(w_orig, x2 + expand_x // 2)
    y2 = min(h_orig, y2 + expand_y // 2)
    
    # Анализируем проекции в расширенной области
    roi_gray = denoised[y1:y2, x1:x2]
    roi_gradient = gradient_binary[y1:y2, x1:x2]
    
    # Горизонтальная проекция (сумма по вертикали)
    horizontal_projection = np.sum(roi_gradient, axis=0) / 255.0
    
    # Вертикальная проекция (сумма по горизонтали)
    vertical_projection = np.sum(roi_gradient, axis=1) / 255.0
    
    # Находим пороги для обрезки
    h_threshold = np.max(horizontal_projection) * 0.15
    v_threshold = np.max(vertical_projection) * 0.15
    
    # Находим границы, где проекция превышает порог
    h_indices = np.where(horizontal_projection > h_threshold)[0]
    v_indices = np.where(vertical_projection > v_threshold)[0]
    
    if len(h_indices) > 0 and len(v_indices) > 0:
        # Уточняем границы
        h_left = max(0, h_indices[0] - padding)
        h_right = min(roi_gradient.shape[1], h_indices[-1] + padding)
        v_top = max(0, v_indices[0] - padding)
        v_bottom = min(roi_gradient.shape[0], v_indices[-1] + padding)
        
        # Переводим координаты обратно в исходное изображение
        final_x1 = x1 + h_left
        final_x2 = x1 + h_right
        final_y1 = y1 + v_top
        final_y2 = y1 + v_bottom
    else:
        # Если не получилось уточнить, используем исходную область
        final_x1, final_y1, final_x2, final_y2 = x1, y1, x2, y2
    
    # Вырезаем финальную область
    extracted = img_original[final_y1:final_y2, final_x1:final_x2]
    
    print(f"Финальные координаты: ({final_x1}, {final_y1}) - ({final_x2}, {final_y2})")
    print(f"Размер выделенной области: {final_x2-final_x1}x{final_y2-final_y1}")
    
    if debug:
        # Визуализируем процесс уточнения
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(roi_gradient, cmap='gray')
        plt.title('Границы в ROI')
        
        plt.subplot(2, 3, 2)
        plt.plot(horizontal_projection)
        plt.axhline(y=h_threshold, color='r', linestyle='--', label=f'Порог ({h_threshold:.0f})')
        plt.axvline(x=h_left, color='g', linestyle='--', label=f'Левая граница ({h_left})')
        plt.axvline(x=h_right, color='g', linestyle='--', label=f'Правая граница ({h_right})')
        plt.title('Горизонтальная проекция')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(vertical_projection, np.arange(len(vertical_projection)))
        plt.axvline(x=v_threshold, color='r', linestyle='--', label=f'Порог ({v_threshold:.0f})')
        plt.axhline(y=v_top, color='g', linestyle='--', label=f'Верхняя граница ({v_top})')
        plt.axhline(y=v_bottom, color='g', linestyle='--', label=f'Нижняя граница ({v_bottom})')
        plt.title('Вертикальная проекция')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()
        
        plt.subplot(2, 3, 4)
        plt.imshow(img_original)
        plt.title('Исходное изображение')
        
        plt.subplot(2, 3, 5)
        img_with_final = img_original.copy()
        cv2.rectangle(img_with_final, (final_x1, final_y1), (final_x2, final_y2), (0, 255, 0), 3)
        plt.imshow(img_with_final)
        plt.title('Финальная область')
        
        plt.subplot(2, 3, 6)
        plt.imshow(extracted)
        plt.title('Выделенное изображение')
        
        plt.tight_layout()
        plt.show()
    
    sheet_region = {
        'original_coords': (final_x1, final_y1, final_x2, final_y2),
        'size': (extracted.shape[1], extracted.shape[0])
    }
    
    return extracted, sheet_region


# Функция для пакетной обработки
def batch_process_sheets(image_folder, output_folder, debug=False):
    """
    Пакетная обработка нескольких изображений
    """
    import glob
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_files = glob.glob(os.path.join(image_folder, "*.jpg")) + \
                  glob.glob(os.path.join(image_folder, "*.png"))
    
    results = {}
    
    for img_path in image_files:
        print(f"\n{'='*50}")
        print(f"Обработка: {os.path.basename(img_path)}")
        
        try:
            extracted, info = precise_sheet_extraction(img_path, debug=debug)
            
            # Сохраняем результат
            output_path = os.path.join(output_folder, f"extracted_{os.path.basename(img_path)}")
            extracted_bgr = cv2.cvtColor(extracted, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, extracted_bgr)
            
            results[img_path] = {
                'success': True,
                'output_path': output_path,
                'info': info
            }
            
            print(f"✓ Успешно обработано, сохранено в: {output_path}")
            
        except Exception as e:
            print(f"✗ Ошибка при обработке: {e}")
            results[img_path] = {'success': False, 'error': str(e)}
    
    return results


# Пример использования
if __name__ == "__main__":
    # Для одного изображения
    image_path = "res\photos\q2.jpg"

    try:
        extracted_img, info = precise_sheet_extraction(image_path, debug=True, padding=10)
        
        # Сохраняем результат
        output_path = "extracted_" + os.path.basename(image_path)
        extracted_bgr = cv2.cvtColor(extracted_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, extracted_bgr)
        
        print(f"\n{'='*50}")
        print("РЕЗУЛЬТАТ:")
        print(f"Координаты в оригинале: {info['original_coords']}")
        print(f"Итоговый размер: {info['size'][0]}x{info['size'][1]}")
        print(f"Результат сохранен в: {output_path}")
        
        # Показываем финальный результат
        plt.figure(figsize=(15, 8))
        plt.subplot(1, 2, 1)
        original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
    except Exception as e:
        print(f"Ошибка: {e}")
    
    # Для пакетной обработки:
    # results = batch_process_sheets("photos/", "extracted/", debug=False)