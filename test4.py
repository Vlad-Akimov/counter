import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage, signal
import os

def precise_sheet_extraction(image_path, debug=True, padding=5):
    """
    Улучшенная функция для точного выделения области с металлическими листами.
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
    
    # Шаг 1: Улучшение изображения
    print("\n=== Шаг 1: Улучшение изображения ===")
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Применяем CLAHE для улучшения локального контраста
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    
    # Медианный фильтр для удаления шума
    denoised = cv2.medianBlur(enhanced, 5)
    
    # Шаг 2: Выделение текстуры металла
    print("\n=== Шаг 2: Анализ текстуры ===")
    
    # Вычисляем градиенты
    sobel_x = cv2.Sobel(denoised, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(denoised, cv2.CV_64F, 0, 1, ksize=3)
    
    # Величина градиента
    gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_magnitude = np.uint8(np.clip(gradient_magnitude, 0, 255))
    
    # Улучшенная бинаризация с адаптивным порогом
    # Используем метод Отсу для автоматического определения порога
    _, gradient_binary = cv2.threshold(gradient_magnitude, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Морфологические операции для улучшения границ
    kernel = np.ones((3, 3), np.uint8)
    gradient_binary = cv2.morphologyEx(gradient_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # Удаляем мелкие шумы
    gradient_binary = cv2.morphologyEx(gradient_binary, cv2.MORPH_OPEN, kernel, iterations=1)
    
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
    
    # Шаг 3: Поиск области с листами
    print("\n=== Шаг 3: Поиск области с листами ===")
    
    # Создаем карту плотности границ
    window_size = min(h_orig, w_orig) // 15  # Уменьшил размер окна для большей точности
    stride = window_size // 2  # Увеличил перекрытие
    
    # Сглаживаем бинарное изображение для лучшей карты плотности
    gradient_binary_float = gradient_binary.astype(np.float32) / 255.0
    
    # Создаем фильтр для скользящего окна
    kernel_density = np.ones((window_size, window_size), np.float32) / (window_size * window_size)
    density_map = cv2.filter2D(gradient_binary_float, -1, kernel_density)
    
    # Находим область с максимальной плотностью
    min_density_threshold = 0.3  # Минимальная плотность для области листов
    high_density_mask = density_map > min_density_threshold
    
    if not np.any(high_density_mask):
        print("Предупреждение: не найдена область с достаточной плотностью границ")
        # Используем центральную область как запасной вариант
        center_x, center_y = w_orig // 2, h_orig // 2
        region_size = min(w_orig, h_orig) // 2
        final_x1 = max(0, center_x - region_size // 2)
        final_y1 = max(0, center_y - region_size // 2)
        final_x2 = min(w_orig, center_x + region_size // 2)
        final_y2 = min(h_orig, center_y + region_size // 2)
    else:
        # Находим координаты bounding box области с высокой плотностью
        y_indices, x_indices = np.where(high_density_mask)
        
        # Добавляем отступы
        padding_density = window_size
        x1 = max(0, np.min(x_indices) - padding_density)
        y1 = max(0, np.min(y_indices) - padding_density)
        x2 = min(density_map.shape[1] - 1, np.max(x_indices) + padding_density)
        y2 = min(density_map.shape[0] - 1, np.max(y_indices) + padding_density)
        
        # Масштабируем координаты обратно к исходному изображению
        scale_x = w_orig / density_map.shape[1]
        scale_y = h_orig / density_map.shape[0]
        
        final_x1 = int(x1 * scale_x)
        final_y1 = int(y1 * scale_y)
        final_x2 = int(x2 * scale_x)
        final_y2 = int(y2 * scale_y)
    
    if debug:
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(density_map, cmap='hot')
        plt.colorbar(label='Плотность границ')
        plt.title('Карта плотности')
        
        plt.subplot(1, 3, 2)
        plt.imshow(high_density_mask, cmap='gray')
        plt.title(f'Области с плотностью > {min_density_threshold}')
        
        plt.subplot(1, 3, 3)
        img_with_region = img_original.copy()
        cv2.rectangle(img_with_region, (final_x1, final_y1), (final_x2, final_y2), (0, 255, 0), 3)
        plt.imshow(img_with_region)
        plt.title('Найденная область')
        plt.tight_layout()
        plt.show()
    
    # Шаг 4: Точное определение границ листов
    print("\n=== Шаг 4: Точное определение границ ===")
    
    # Вырезаем область интереса
    roi = gradient_binary[final_y1:final_y2, final_x1:final_x2]
    roi_gray = denoised[final_y1:final_y2, final_x1:final_x2]
    
    # Создаем улучшенные проекции
    # Используем взвешенную сумму с учетом интенсивности градиента
    roi_gradient_float = roi.astype(np.float32) / 255.0
    
    # Горизонтальная проекция с весами
    horizontal_projection = np.sum(roi_gradient_float, axis=0)
    # Вертикальная проекция с весами
    vertical_projection = np.sum(roi_gradient_float, axis=1)
    
    # Нормализуем проекции
    horizontal_projection = horizontal_projection / np.max(horizontal_projection) if np.max(horizontal_projection) > 0 else horizontal_projection
    vertical_projection = vertical_projection / np.max(vertical_projection) if np.max(vertical_projection) > 0 else vertical_projection
    
    # Находим пороги с адаптивным значением
    h_threshold = 0.1  # 10% от максимума
    v_threshold = 0.1
    
    # Находим границы, где проекция превышает порог
    h_indices = np.where(horizontal_projection > h_threshold)[0]
    v_indices = np.where(vertical_projection > v_threshold)[0]
    
    if len(h_indices) > 0 and len(v_indices) > 0:
        # Находим границы с учетом резких перепадов
        # Ищем первые и последние индексы с значительным превышением порога
        
        # Для левой границы - ищем первый резкий подъем
        left_idx = h_indices[0]
        for i in range(1, len(h_indices)):
            if h_indices[i] - h_indices[i-1] > 5:  # Разрыв в индексах
                left_idx = h_indices[i]
                break
        
        # Для правой границы - ищем последний резкий спад
        right_idx = h_indices[-1]
        for i in range(len(h_indices)-1, 0, -1):
            if h_indices[i] - h_indices[i-1] > 5:  # Разрыв в индексах
                right_idx = h_indices[i-1]
                break
        
        # Аналогично для вертикальных границ
        top_idx = v_indices[0]
        for i in range(1, len(v_indices)):
            if v_indices[i] - v_indices[i-1] > 5:
                top_idx = v_indices[i]
                break
        
        bottom_idx = v_indices[-1]
        for i in range(len(v_indices)-1, 0, -1):
            if v_indices[i] - v_indices[i-1] > 5:
                bottom_idx = v_indices[i-1]
                break
        
        # Добавляем отступы
        left_idx = max(0, left_idx - padding)
        right_idx = min(roi.shape[1], right_idx + padding)
        top_idx = max(0, top_idx - padding)
        bottom_idx = min(roi.shape[0], bottom_idx + padding)
        
        # Переводим координаты обратно
        final_x1 = final_x1 + left_idx
        final_x2 = final_x1 + (right_idx - left_idx)
        final_y1 = final_y1 + top_idx
        final_y2 = final_y1 + (bottom_idx - top_idx)
    
    # Финальная проверка границ
    final_x1 = max(0, final_x1)
    final_y1 = max(0, final_y1)
    final_x2 = min(w_orig, final_x2)
    final_y2 = min(h_orig, final_y2)
    
    # Вырезаем финальную область
    extracted = img_original[final_y1:final_y2, final_x1:final_x2]
    
    print(f"Финальные координаты: ({final_x1}, {final_y1}) - ({final_x2}, {final_y2})")
    print(f"Размер выделенной области: {final_x2-final_x1}x{final_y2-final_y1}")
    
    if debug:
        # Визуализация улучшенных проекций
        plt.figure(figsize=(15, 10))
        
        plt.subplot(2, 3, 1)
        plt.imshow(roi, cmap='gray')
        plt.title('Границы в ROI')
        # Рисуем найденные границы
        plt.axvline(x=left_idx - padding if 'left_idx' in locals() else 0, color='g', linestyle='--', linewidth=2)
        plt.axvline(x=right_idx - padding if 'right_idx' in locals() else roi.shape[1], color='g', linestyle='--', linewidth=2)
        plt.axhline(y=top_idx - padding if 'top_idx' in locals() else 0, color='g', linestyle='--', linewidth=2)
        plt.axhline(y=bottom_idx - padding if 'bottom_idx' in locals() else roi.shape[0], color='g', linestyle='--', linewidth=2)
        
        plt.subplot(2, 3, 2)
        plt.plot(horizontal_projection)
        plt.axhline(y=h_threshold, color='r', linestyle='--', label=f'Порог')
        if 'left_idx' in locals():
            plt.axvline(x=left_idx, color='g', linestyle='--', label=f'Левая граница')
            plt.axvline(x=right_idx, color='g', linestyle='--', label=f'Правая граница')
        plt.title('Горизонтальная проекция')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 3, 3)
        plt.plot(vertical_projection, np.arange(len(vertical_projection)))
        plt.axvline(x=v_threshold, color='r', linestyle='--', label=f'Порог')
        if 'top_idx' in locals():
            plt.axhline(y=top_idx, color='g', linestyle='--', label=f'Верхняя граница')
            plt.axhline(y=bottom_idx, color='g', linestyle='--', label=f'Нижняя граница')
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


# Пример использования
if __name__ == "__main__":
    # Для одного изображения
    image_path = "res/photos/q14.jpg"
    
    try:
        print("="*60)
        print("НАЧАЛО ОБРАБОТКИ ИЗОБРАЖЕНИЯ")
        print("="*60)
        
        # Запускаем выделение области листов
        extracted_img, info = precise_sheet_extraction(image_path, debug=True, padding=10)
        
        # Сохраняем результат
        output_path = "extracted_" + os.path.basename(image_path)
        extracted_bgr = cv2.cvtColor(extracted_img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, extracted_bgr)
        
        print("\n" + "="*60)
        print("РЕЗУЛЬТАТ ОБРАБОТКИ:")
        print("="*60)
        print(f"✓ Исходное изображение: {os.path.basename(image_path)}")
        print(f"✓ Координаты в оригинале: {info['original_coords']}")
        print(f"✓ Итоговый размер: {info['size'][0]}x{info['size'][1]} пикселей")
        print(f"✓ Результат сохранен в: {output_path}")
        print("="*60)
        
        # Показываем финальный результат в сравнении
        plt.figure(figsize=(15, 8))
        
        # Загружаем оригинал для сравнения
        original = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        
        # Оригинал с выделенной областью
        plt.subplot(1, 3, 1)
        plt.imshow(original)
        plt.title('Оригинальное изображение')
        plt.axis('on')
        
        # Оригинал с прямоугольником выделения
        plt.subplot(1, 3, 2)
        original_with_box = original.copy()
        x1, y1, x2, y2 = info['original_coords']
        cv2.rectangle(original_with_box, (x1, y1), (x2, y2), (0, 255, 0), 3)
        plt.imshow(original_with_box)
        plt.title(f'Выделенная область\n({x2-x1}x{y2-y1} пикселей)')
        plt.axis('on')
        
        # Выделенное изображение
        plt.subplot(1, 3, 3)
        plt.imshow(extracted_img)
        plt.title('Результат выделения')
        plt.axis('on')
        
        plt.tight_layout()
        plt.show()
        
        # Дополнительная информация об изображении
        print("\nДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ:")
        print(f"  • Отношение сторон: {info['size'][0]/info['size'][1]:.2f}")
        print(f"  • Процент от оригинала: {(info['size'][0]*info['size'][1])/(original.shape[1]*original.shape[0])*100:.1f}%")
        
    except Exception as e:
        print(f"\n❌ НЕПРЕДВИДЕННАЯ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()