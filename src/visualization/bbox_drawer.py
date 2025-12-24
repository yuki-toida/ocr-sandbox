"""バウンディングボックス描画ユーティリティ"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Any


def draw_bounding_boxes(
    image: np.ndarray,
    text_boxes: List[Dict],
    bbox_color: Tuple[int, int, int] = (0, 255, 0),
    bbox_thickness: int = 2,
    show_text: bool = True,
    font_scale: float = 0.5
) -> np.ndarray:
    """
    画像上にバウンディングボックスとテキストを描画
    信頼度に応じて色を変更:
    - 0.9以上: 緑 (高信頼度)
    - 0.7以上: 黄色 (中信頼度)
    - 0.7未満: 赤 (低信頼度)

    Args:
        image: 入力画像（NumPy配列）
        text_boxes: バウンディングボックス情報のリスト
                   各要素は 'bbox', 'text', 'confidence' キーを持つ辞書
        bbox_color: デフォルトRGB色タプル（信頼度がない場合に使用）
        bbox_thickness: 線の太さ
        show_text: テキストラベルを表示するか
        font_scale: テキストサイズのスケール

    Returns:
        np.ndarray: バウンディングボックスが描画された画像
    """
    # 画像をコピー（元の画像を変更しない）
    result_image = image.copy()

    for box in text_boxes:
        bbox = box.get('bbox')
        text = box.get('text', '')
        confidence = box.get('confidence', 1.0)

        if bbox is None:
            continue

        # バウンディングボックスの形式を変換
        x, y, w, h = convert_bbox_format(bbox)

        # 信頼度に応じて色を決定（BGR形式）
        if confidence >= 0.9:
            color = (0, 255, 0)  # 緑
        elif confidence >= 0.7:
            color = (0, 255, 255)  # 黄色
        else:
            color = (0, 0, 255)  # 赤

        # 矩形を描画
        cv2.rectangle(result_image, (x, y), (x + w, y + h), color, bbox_thickness)

        # テキストを表示
        if show_text and text:
            # テキストを描画（背景付き）
            result_image = add_text_with_background(
                result_image,
                text,
                (x, y - 5),
                font_scale=font_scale,
                bg_color=(255, 255, 255),
                text_color=(0, 0, 0)
            )

    return result_image


def convert_bbox_format(bbox: Any) -> Tuple[int, int, int, int]:
    """
    様々なバウンディングボックス形式を (x, y, w, h) に変換

    サポートする形式:
    - [x, y, w, h]
    - [x1, y1, x2, y2]  # 2つの角の座標
    - [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]  # 多角形（4点）
    - [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]  # 多角形（リスト形式）

    Args:
        bbox: バウンディングボックス（様々な形式）

    Returns:
        Tuple[int, int, int, int]: (x, y, width, height)
    """
    # リストまたはタプルの場合
    if isinstance(bbox, (list, tuple)):
        # 4要素の場合
        if len(bbox) == 4:
            # すべて数値の場合
            if all(isinstance(x, (int, float)) for x in bbox):
                x1, y1, x2_or_w, y2_or_h = bbox
                # [x, y, w, h] 形式か [x1, y1, x2, y2] 形式かを判定
                # 通常、w, h は正の値で、座標より小さい
                if x2_or_w < x1 or y2_or_h < y1:
                    # [x, y, w, h] 形式と判定
                    return int(x1), int(y1), int(x2_or_w), int(y2_or_h)
                else:
                    # [x1, y1, x2, y2] 形式と判定
                    w = x2_or_w - x1
                    h = y2_or_h - y1
                    return int(x1), int(y1), int(w), int(h)
            # タプルまたはリストの要素（多角形）の場合
            elif all(isinstance(p, (list, tuple)) and len(p) == 2 for p in bbox):
                # 多角形から外接矩形を計算
                points = np.array(bbox)
                x = int(np.min(points[:, 0]))
                y = int(np.min(points[:, 1]))
                w = int(np.max(points[:, 0]) - x)
                h = int(np.max(points[:, 1]) - y)
                return x, y, w, h

    # デフォルト: 変換できない場合は (0, 0, 0, 0) を返す
    return 0, 0, 0, 0


def add_text_with_background(
    image: np.ndarray,
    text: str,
    position: Tuple[int, int],
    font_scale: float = 0.5,
    bg_color: Tuple[int, int, int] = (255, 255, 255),
    text_color: Tuple[int, int, int] = (0, 0, 0)
) -> np.ndarray:
    """
    背景矩形付きのテキストを追加（可読性向上）

    Args:
        image: 入力画像
        text: 表示するテキスト
        position: テキストの位置 (x, y)
        font_scale: フォントサイズのスケール
        bg_color: 背景色（BGR）
        text_color: テキスト色（BGR）

    Returns:
        np.ndarray: テキストが追加された画像
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    # テキストサイズを取得
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    x, y = position

    # yが画像の上端より上の場合は下に配置
    if y < 0:
        y = text_height + 5

    # 背景矩形を描画
    cv2.rectangle(
        image,
        (x, y - text_height - baseline),
        (x + text_width, y + baseline),
        bg_color,
        -1  # 塗りつぶし
    )

    # テキストを描画
    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        text_color,
        thickness,
        cv2.LINE_AA
    )

    return image
