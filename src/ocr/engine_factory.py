"""OCRエンジンファクトリー"""

from typing import Union
from .yomitoku_wrapper import YomitokuOCR
from .paddleocr_wrapper import PaddleOCR_Wrapper
from .tesseract_wrapper import Tesseract_Wrapper


def create_ocr_engine(
    engine_type: str = "yomitoku",
    device: str = "cpu",
    visualize: bool = True
) -> Union[YomitokuOCR, PaddleOCR_Wrapper, Tesseract_Wrapper]:
    """
    OCRエンジンインスタンスを作成するファクトリー関数

    Args:
        engine_type: "yomitoku", "paddleocr", または "tesseract"
        device: "cpu", "cuda", または "mps"
        visualize: 可視化機能を有効にするか

    Returns:
        Union[YomitokuOCR, PaddleOCR_Wrapper, Tesseract_Wrapper]: OCRエンジンインスタンス

    Raises:
        ValueError: サポートされていないエンジンタイプの場合
    """
    if engine_type == "yomitoku":
        return YomitokuOCR(device=device, visualize=visualize)
    elif engine_type == "paddleocr":
        return PaddleOCR_Wrapper(device=device, visualize=visualize)
    elif engine_type == "tesseract":
        return Tesseract_Wrapper(device=device, visualize=visualize)
    else:
        raise ValueError(
            f"サポートされていないOCRエンジン: {engine_type}. "
            f"'yomitoku', 'paddleocr', または 'tesseract' を指定してください。"
        )
