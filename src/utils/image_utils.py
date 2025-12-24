"""画像処理ユーティリティ"""

import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """
    PIL画像をOpenCV形式（BGR）に変換

    Args:
        pil_image: PIL Image オブジェクト

    Returns:
        np.ndarray: OpenCV形式の画像（BGR）
    """
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """
    OpenCV画像（BGR）をPIL Image（RGB）に変換

    Args:
        cv2_image: OpenCV形式の画像（BGR）

    Returns:
        Image.Image: PIL Image オブジェクト
    """
    return Image.fromarray(cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB))
