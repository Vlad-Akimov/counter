import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import os

def extract_sheet_stack(image_path, debug=True, padding=20):
    """
    Функция для выделения цельной области со стопкой металлических листов.
    Находит внешний контур стопки и использует его как границу.
    
    Parameters:
    -----------
    image_path : str
        Путь к изображению
    debug : bool
        Если True, показывает промежуточные результаты
    padding : int
        Дополнительные пиксели вокруг контура
        
    Returns:
    --------
    extracted_image : numpy.ndarray
        Выделенное изображение стопки
    contour_info : dict
        Информация о найденном контуре
    """
    
    # Шаг 0: Загрузка изображения
    print(f"Загружаем изображение: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = img.shape[:2]
    
    print(f"Размер изображения: {w_orig}x{h_orig} пикселей")
    
    # Шаг 1: Предобработка для выделения текстуры металла
    print("\n=== Шаг 1: Анализ текстуры металла ===")
    
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Применяем CLAHE для улучшения контраста
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Медианный фильтр для удаления шума
    denoised = cv2.medianBlur(enhanced, 5)
    
    # Вычисляем градиенты (подчеркивает границы листов)
    sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    
    # Величина градиента
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))
    
    # Бинаризация градиента (области с текстурой металла)
    _, gradient_binary = cv2.threshold(gradient_magnitude, 20, 255, cv2.THRESH_BINARY)
    
    # Морфологические операции для заполнения пробелов
    kernel = np.ones((5, 5), np.uint8)
    
    # Сначала расширяем, чтобы соединить близкие границы
    dilated = cv2.dilate(gradient_binary, kernel, iterations=3)
    
    # Затем закрываем дыры
    closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    # И наконец открываем, чтобы убрать мелкий шум
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=2)
    
    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(gradient_binary, cmap='gray')
        plt.title('Бинарные градиенты')
        plt.subplot(1, 3, 2)
        plt.imshow(dilated, cmap='gray')
        plt.title('После расширения')
        plt.subplot(1, 3, 3)
        plt.imshow(opened, cmap='gray')
        plt.title('После заполнения')
        plt.tight_layout()
        plt.show()
    
    # Шаг 2: Поиск внешнего контура стопки
    print("\n=== Шаг 2: Поиск внешнего контура стопки ===")
    
    # Находим все контуры
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("Не удалось найти контуры!")
        # Возвращаем центральную часть как запасной вариант
        h, w = img_rgb.shape[:2]
        margin = min(h, w) // 4
        return img_rgb[margin:h-margin, margin:w-margin], None
    
    # Сортируем контуры по площади (от большего к меньшему)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    if debug:
        # Визуализируем все найденные контуры
        img_contours = img_rgb.copy()
        cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 2)
        
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(opened, cmap='gray')
        plt.title('Бинарная маска')
        plt.subplot(1, 2, 2)
        plt.imshow(img_contours)
        plt.title(f'Найдено контуров: {len(contours)}')
        plt.tight_layout()
        plt.show()
    
    # Шаг 3: Выбор наиболее подходящего контура
    print("\n=== Шаг 3: Выбор контура стопки ===")
    
    # Параметры для оценки контуров
    img_area = h_orig * w_orig
    best_contour = None
    best_score = 0
    
    for i, contour in enumerate(contours[:5]):  # Проверяем топ-5 по площади
        area = cv2.contourArea(contour)
        
        # Площадь должна быть не менее 10% и не более 90% изображения
        if area < 0.1 * img_area or area > 0.9 * img_area:
            continue
        
        # Получаем ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(contour)
        
        # Вычисляем соотношение сторон
        aspect_ratio = w / h if h > 0 else 0
        
        # Вычисляем выпуклость контура
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        # Оценка контура
        score = area * solidity  # Чем больше площадь и выпуклее, тем лучше
        
        if debug:
            print(f"Контур {i}: площадь={area/1000:.0f}k, соотношение={aspect_ratio:.2f}, твердость={solidity:.2f}, оценка={score/1000:.0f}k")
        
        if score > best_score:
            best_score = score
            best_contour = contour
    
    if best_contour is None:
        print("Не удалось выбрать подходящий контур. Использую самый большой.")
        best_contour = contours[0]
    
    # Шаг 4: Получение bounding box и создание маски
    print("\n=== Шаг 4: Создание маски стопки ===")
    
    # Создаем маску для контура
    mask = np.zeros((h_orig, w_orig), dtype=np.uint8)
    cv2.drawContours(mask, [best_contour], -1, 255, -1)  # -1 заполняет контур
    
    # Добавляем отступы
    kernel_dilate = np.ones((padding*2, padding*2), np.uint8)
    mask_with_padding = cv2.dilate(mask, kernel_dilate, iterations=1)
    
    # Получаем bounding box с отступами
    x, y, w, h = cv2.boundingRect(best_contour)
    
    # Добавляем отступы к bounding box
    x = max(0, x - padding)
    y = max(0, y - padding)
    x2 = min(w_orig, x + w + padding*2)
    y2 = min(h_orig, y + h + padding*2)
    
    # Вырезаем область
    extracted = img_rgb[y:y2, x:x2]
    mask_cropped = mask_with_padding[y:y2, x:x2]
    
    # Применяем маску к выделенной области (делаем фон белым)
    result = extracted.copy()
    background = np.ones_like(extracted) * 255
    mask_3channel = cv2.cvtColor(mask_cropped, cv2.COLOR_GRAY2RGB) / 255
    result = (extracted * mask_3channel + background * (1 - mask_3channel)).astype(np.uint8)
    
    if debug:
        # Визуализируем результат
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 4, 1)
        plt.imshow(img_rgb)
        plt.title('Исходное изображение')
        
        plt.subplot(1, 4, 2)
        img_with_contour = img_rgb.copy()
        cv2.drawContours(img_with_contour, [best_contour], -1, (0, 255, 0), 3)
        plt.imshow(img_with_contour)
        plt.title('Выбранный контур')
        
        plt.subplot(1, 4, 3)
        plt.imshow(mask_with_padding, cmap='gray')
        plt.title(f'Маска с отступами ({padding}px)')
        
        plt.subplot(1, 4, 4)
        plt.imshow(result)
        plt.title('Результат выделения')
        
        plt.tight_layout()
        plt.show()
    
    print(f"Bounding box: ({x}, {y}) - ({x2}, {y2})")
    print(f"Размер результата: {result.shape[1]}x{result.shape[0]}")
    
    # Шаг 5: Выравнивание (опционально)
    print("\n=== Шаг 5: Выравнивание стопки ===")
    
    # Конвертируем результат в градации серого для поиска линий
    result_gray = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    
    # Используем только область маски для поиска линий
    edges = cv2.Canny(result_gray, 50, 150, apertureSize=3)
    edges = cv2.bitwise_and(edges, mask_cropped)
    
    # Поиск линий
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=30, 
        minLineLength=max(30, result.shape[0] // 10),
        maxLineGap=10
    )
    
    if lines is not None and len(lines) > 5:
        # Анализируем углы
        angles = []
        lengths = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            # Нас интересуют длинные линии
            if length > result.shape[0] // 5:
                angles.append(angle)
                lengths.append(length)
        
        if angles:
            # Взвешенный угол
            weighted_angle = np.average(angles, weights=lengths)
            
            # Определяем угол поворота для выравнивания
            if abs(weighted_angle) < 45:
                rotation_angle = 90 - weighted_angle
            elif abs(weighted_angle) > 135:
                rotation_angle = 90 - (180 - abs(weighted_angle))
            else:
                rotation_angle = -weighted_angle
            
            print(f"Средний угол линий: {weighted_angle:.2f}°")
            print(f"Угол поворота: {rotation_angle:.2f}°")
            
            # Поворачиваем
            h, w = result.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
            
            # Вычисляем новые размеры
            cos = abs(rotation_matrix[0, 0])
            sin = abs(rotation_matrix[0, 1])
            new_w = int(h * sin + w * cos)
            new_h = int(h * cos + w * sin)
            
            rotation_matrix[0, 2] += (new_w / 2) - center[0]
            rotation_matrix[1, 2] += (new_h / 2) - center[1]
            
            # Поворачиваем и маску, и изображение
            aligned = cv2.warpAffine(
                result, 
                rotation_matrix, 
                (new_w, new_h), 
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(255, 255, 255)
            )
            
            # Поворачиваем маску для последующего использования
            mask_aligned = cv2.warpAffine(
                mask_cropped.astype(np.float32),
                rotation_matrix,
                (new_w, new_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            ).astype(np.uint8)
            
            # Финальная обрезка по маске
            if np.any(mask_aligned):
                # Находим границы маски
                coords = np.column_stack(np.where(mask_aligned > 0))
                if len(coords) > 0:
                    y_min, x_min = coords.min(axis=0)
                    y_max, x_max = coords.max(axis=0)
                    
                    # Добавляем небольшой отступ
                    pad = 5
                    y_min = max(0, y_min - pad)
                    x_min = max(0, x_min - pad)
                    y_max = min(new_h, y_max + pad)
                    x_max = min(new_w, x_max + pad)
                    
                    aligned = aligned[y_min:y_max, x_min:x_max]
            
            if debug:
                plt.figure(figsize=(12, 5))
                plt.subplot(1, 2, 1)
                plt.imshow(result)
                plt.title('До выравнивания')
                plt.subplot(1, 2, 2)
                plt.imshow(aligned)
                plt.title('После выравнивания')
                plt.tight_layout()
                plt.show()
            
            result = aligned
    
    print("=== Выделение завершено ===")
    
    contour_info = {
        'bounding_box': (x, y, x2, y2),
        'area': cv2.contourArea(best_contour),
        'padding': padding
    }
    
    return result, contour_info


if __name__ == "__main__":
    # Использование
    image_path = "res\photos\q20.jpg"
    
    if os.path.exists(image_path):
        result, info = extract_sheet_stack(image_path, debug=True, padding=25)
        
        # Сохраняем результат
        output_path = "extracted_stack.jpg"
        cv2.imwrite(output_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
        print(f"\nРезультат сохранен в: {output_path}")
    else:
        print(f"Файл {image_path} не найден. Запускаю тест с синтетическим изображением.")