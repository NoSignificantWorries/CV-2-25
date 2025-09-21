import cv2
import numpy as np


def perspective_correction(image, target_width=500, target_height=700, get_corners_function=None):
    if get_corners_function is None:
        raise RuntimeError("ERROR: Not provided function to get document corners.")

    src_points = get_corners_function(image)
    
    target_points = np.array([
        [0, 0],
        [target_width, 0],
        [target_width, target_height],
        [0, target_height]
    ], dtype=np.float32)
    
    H, status = cv2.findHomography(src_points, target_points)
    
    result = cv2.warpPerspective(image, H, (target_width, target_height))
    
    return result


if __name__ == "__main__":
    image = cv2.imread('document.jpg')
    
    if image is None:
        print("Ошибка: не удалось загрузить изображение")
    else:
        corrected_image = perspective_correction(image)
        
        cv2.imwrite('corrected_document.jpg', corrected_image)
