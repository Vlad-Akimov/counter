import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import os

def prepare_sheet_image(image_path, debug=True):
    """
    Функция для подготовки изображения стопки металлических листов.
    Выполняет:
    1. Выделение области со стопкой (ROI)
    2. Выравнивание стопки (поворот)
    
    Parameters:
    -----------
    image_path : str
        Путь к изображению
    debug : bool
        Если True, показывает промежуточные результаты
        
    Returns:
    --------
    prepared_image : numpy.ndarray
        Подготовленное изображение (выровненное и обрезанное)
    rotation_angle : float
        Угол поворота, который был применен
    """
    
    # Шаг 0: Загрузка изображения
    print(f"Загружаем изображение: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {image_path}")
    
    # Конвертируем BGR в RGB для корректного отображения в matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_original = img_rgb.copy()
    
    print(f"Размер изображения: {img.shape[1]}x{img.shape[0]} пикселей")
    
    # Шаг 1: Выделение области интереса (ROI) - поиск стопки
    print("\n=== Шаг 1: Выделение области со стопкой ===")
    
    # Конвертируем в оттенки серого
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Применяем размытие для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Используем адаптивную бинаризацию для выделения контрастных областей
    # Это поможет найти границы стопки даже при неравномерном освещении
    binary = cv2.adaptiveThreshold(
        blurred, 
        255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        11,  # размер блока
        2    # константа
    )
    
    # Морфологические операции для улучшения результата
    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Находим контуры
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if debug:
        print(f"Найдено контуров: {len(contours)}")
        
        # Визуализируем все контуры
        img_contours = img_original.copy()
        cv2.drawContours(img_contours, contours, -1, (255, 0, 0), 2)
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(binary, cmap='gray')
        plt.title('Бинаризация')
        plt.subplot(1, 3, 2)
        plt.imshow(img_contours)
        plt.title('Все контуры')
    
    # Фильтруем контуры по размеру (оставляем только крупные)
    min_area = img.shape[0] * img.shape[1] * 0.05  # минимум 5% от площади изображения
    large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if debug:
        print(f"Крупных контуров (>5% площади): {len(large_contours)}")
    
    if not large_contours:
        print("Не удалось найти крупные контуры. Использую центральную область.")
        # Если не нашли контуры, берем центральную часть
        h, w = img.shape[:2]
        x1, y1 = w // 4, h // 4
        x2, y2 = 3 * w // 4, 3 * h // 4
        roi = img_original[y1:y2, x1:x2]
        roi_coords = (x1, y1, x2, y2)
    else:
        # Находим самый большой контур (скорее всего это наша стопка)
        largest_contour = max(large_contours, key=cv2.contourArea)
        
        # Получаем ограничивающий прямоугольник
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Добавляем небольшой отступ (5%)
        padding_x = int(w * 0.05)
        padding_y = int(h * 0.05)
        
        x1 = max(0, x - padding_x)
        y1 = max(0, y - padding_y)
        x2 = min(img.shape[1], x + w + padding_x)
        y2 = min(img.shape[0], y + h + padding_y)
        
        roi = img_original[y1:y2, x1:x2]
        roi_coords = (x1, y1, x2, y2)
        
        if debug:
            # Визуализируем найденную ROI
            img_roi = img_original.copy()
            cv2.rectangle(img_roi, (x1, y1), (x2, y2), (0, 255, 0), 3)
            plt.subplot(1, 3, 3)
            plt.imshow(img_roi)
            plt.title('Выделенная область (ROI)')
            plt.tight_layout()
            plt.show()
    
    print(f"ROI координаты: ({x1}, {y1}) - ({x2}, {y2})")
    print(f"Размер ROI: {x2-x1}x{y2-y1} пикселей")
    
    # Шаг 2: Выравнивание стопки
    print("\n=== Шаг 2: Выравнивание стопки ===")
    
    # Конвертируем ROI в оттенки серого для анализа
    roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    
    # Улучшаем контраст для лучшего выделения границ
    roi_gray = cv2.equalizeHist(roi_gray)
    
    # Детектор границ Canny
    edges = cv2.Canny(roi_gray, 50, 150, apertureSize=3)
    
    # Применяем морфологию для соединения разрывов в линиях
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    # Поиск линий с помощью преобразования Хафа
    lines = cv2.HoughLinesP(
        edges, 
        rho=1, 
        theta=np.pi/180, 
        threshold=50, 
        minLineLength=30, 
        maxLineGap=10
    )
    
    if debug:
        # Визуализируем найденные линии
        img_lines = roi.copy()
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(img_lines, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(roi_gray, cmap='gray')
        plt.title('ROI в grayscale')
        plt.subplot(1, 3, 2)
        plt.imshow(edges, cmap='gray')
        plt.title('Границы Canny')
        plt.subplot(1, 3, 3)
        plt.imshow(img_lines)
        plt.title(f'Найдено линий: {len(lines) if lines is not None else 0}')
        plt.tight_layout()
        plt.show()
    
    if lines is None or len(lines) < 5:
        print("Недостаточно линий для определения угла. Пропускаем выравнивание.")
        return roi, 0
    
    # Анализируем углы наклона линий
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Вычисляем угол в градусах
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        # Нас интересуют линии, близкие к вертикальным (80-100 градусов) 
        # или горизонтальным (0-10, 170-180 градусов)
        # В зависимости от того, как лежат листы
        if abs(angle) < 10 or abs(angle) > 170:
            angles.append(angle)
        elif 80 < angle < 100:
            angles.append(angle - 90)  # Корректируем для вертикальных
    
    if not angles:
        print("Не найдено подходящих линий для определения ориентации")
        return roi, 0
    
    # Находим медианный угол (устойчив к выбросам)
    median_angle = np.median(angles)
    
    # Корректируем угол: нам нужно сделать линии вертикальными
    # Если линии горизонтальные, поворачиваем на 90 - median_angle
    if abs(median_angle) < 45:
        rotation_angle = 90 - median_angle
    else:
        rotation_angle = -median_angle
    
    print(f"Найден медианный угол наклона: {median_angle:.2f}°")
    print(f"Угол поворота для выравнивания: {rotation_angle:.2f}°")
    
    # Поворачиваем изображение
    h, w = roi.shape[:2]
    center = (w // 2, h // 2)
    
    # Получаем матрицу поворота
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Вычисляем новые размеры, чтобы изображение не обрезалось
    cos = abs(rotation_matrix[0, 0])
    sin = abs(rotation_matrix[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Корректируем матрицу поворота с учетом новых размеров
    rotation_matrix[0, 2] += (new_w / 2) - center[0]
    rotation_matrix[1, 2] += (new_h / 2) - center[1]
    
    # Применяем поворот
    aligned = cv2.warpAffine(
        roi, 
        rotation_matrix, 
        (new_w, new_h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)  # Белый фон для заполнения пустот
    )
    
    if debug:
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(roi)
        plt.title('До выравнивания')
        plt.subplot(1, 2, 2)
        plt.imshow(aligned)
        plt.title(f'После выравнивания (угол: {rotation_angle:.1f}°)')
        plt.tight_layout()
        plt.show()
    
    print(f"Итоговый размер изображения: {aligned.shape[1]}x{aligned.shape[0]}")
    print("=== Подготовка завершена ===\n")
    
    return aligned, rotation_angle


# Пример использования
if __name__ == "__main__":
    # Путь к вашему изображению
    image_path = "res\photos\q15.jpg"
    
    try:
        prepared_img, angle = prepare_sheet_image(image_path, debug=True)
        
        # Сохраняем результат
        # output_path = "prepared_" + os.path.basename(image_path)
        # prepared_img_bgr = cv2.cvtColor(prepared_img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(output_path, prepared_img_bgr)
        # print(f"Результат сохранен в: {output_path}")
        
        # Показываем финальный результат
        plt.figure(figsize=(10, 8))
        plt.imshow(prepared_img)
        plt.title(f'Итоговое подготовленное изображение (поворот: {angle:.1f}°)')
        plt.axis('off')
        plt.show()
        
    except Exception as e:
        print(f"Ошибка при обработке: {e}")