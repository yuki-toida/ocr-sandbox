"""Tesseract OCR ラッパークラス"""

import numpy as np
from typing import Dict, List, Any
import pytesseract
from PIL import Image


class Tesseract_Wrapper:
    """Tesseract OCR エンジンのラッパークラス"""

    def __init__(self, device: str = "cpu", visualize: bool = True):
        """
        Tesseract OCR エンジンを初期化

        Args:
            device: デバイス指定（Tesseractは常にCPU動作）
            visualize: 内部可視化を有効にするか（現在未使用）
        """
        self.device = device  # Tesseractは常にCPU
        self.visualize = visualize

        # Tesseractがインストールされているか確認
        try:
            version = pytesseract.get_tesseract_version()
            print(f"Tesseract初期化完了 - バージョン: {version}")
        except Exception as e:
            print(f"Tesseract初期化エラー: {e}")
            raise

    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像をOCR処理

        Args:
            image: NumPy配列（BGR形式、OpenCVから取得）

        Returns:
            Dict: OCR結果
                {
                    'text_boxes': List[Dict],  # 各要素は 'bbox', 'text', 'confidence' を含む
                    'raw_results': results,    # 元のTesseract結果
                    'visualization': None      # 現在未実装
                }
        """
        try:
            print(f"Tesseract処理開始 - 画像サイズ: {image.shape}")

            # BGR → RGB変換（TesseractはRGB形式を想定）
            image_rgb = image[:, :, ::-1].copy()

            # PIL Imageに変換
            pil_image = Image.fromarray(image_rgb)
            print("画像をRGB形式に変換しました")

            # OCR実行（日本語 + 英語）
            print("OCR実行中...")
            # PSM 3: 完全な自動ページ分割（デフォルト、複雑なレイアウトに対応）
            # OEM 1: LSTMニューラルネットのみ（最新の認識エンジン）
            data = pytesseract.image_to_data(
                pil_image,
                output_type=pytesseract.Output.DICT,
                lang='jpn+eng',  # 日本語と英語を併用
                config='--psm 3 --oem 1'
            )
            print(f"OCR完了 - {len(data['text'])}個の要素を検出")

            # バウンディングボックスを抽出
            text_boxes = self._extract_bounding_boxes(data)
            print(f"テキストボックス抽出完了 - {len(text_boxes)}個検出")

            return {
                'text_boxes': text_boxes,
                'raw_results': data,
                'visualization': None
            }
        except Exception as e:
            print(f"Tesseract処理エラー: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _extract_bounding_boxes(self, data: Dict) -> List[Dict]:
        """
        Tesseract結果からバウンディングボックス情報を抽出

        Args:
            data: Tesseractの結果辞書
                {
                    'level': [1, 2, 3, 4, 5, ...],  # 1=page, 5=word
                    'left': [x1, x2, ...],
                    'top': [y1, y2, ...],
                    'width': [w1, w2, ...],
                    'height': [h1, h2, ...],
                    'conf': [c1, c2, ...],  # 信頼度 0-100 または -1
                    'text': ['text1', 'text2', ...]
                }

        Returns:
            List[Dict]: バウンディングボックス情報のリスト
                各要素は以下のキーを持つ:
                - 'bbox': バウンディングボックス座標（4点ポリゴン）
                - 'text': 認識されたテキスト
                - 'confidence': 信頼度スコア（0.0～1.0）
        """
        text_boxes = []
        n_boxes = len(data['text'])

        try:
            for i in range(n_boxes):
                # 行レベル(level=4)または単語レベル(level=5)を抽出
                # 日本語は行レベルの方が文脈を保持しやすい
                level = int(data['level'][i])
                text = data['text'][i].strip()
                conf = float(data['conf'][i])

                # 条件: 行レベルまたは単語レベル、テキストが空でない、信頼度が0以上
                if level in [4, 5] and text and conf >= 0:
                    # 矩形座標を取得
                    x = int(data['left'][i])
                    y = int(data['top'][i])
                    w = int(data['width'][i])
                    h = int(data['height'][i])

                    # 矩形座標を4点ポリゴンに変換（既存のエンジンと統一）
                    bbox = [
                        [x, y],           # 左上
                        [x + w, y],       # 右上
                        [x + w, y + h],   # 右下
                        [x, y + h]        # 左下
                    ]

                    text_boxes.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': conf / 100.0  # 0-100を0.0-1.0に正規化
                    })

        except Exception as e:
            print(f"Error extracting bounding boxes from Tesseract results: {e}")
            import traceback
            traceback.print_exc()

        return text_boxes
