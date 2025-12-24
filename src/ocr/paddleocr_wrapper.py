"""PaddleOCR ラッパークラス"""

import numpy as np
from typing import Dict, List, Any
from paddleocr import PaddleOCR


class PaddleOCR_Wrapper:
    """PaddleOCR エンジンのラッパークラス"""

    def __init__(self, device: str = "cpu", visualize: bool = True):
        """
        PaddleOCR エンジンを初期化

        Args:
            device: "cpu", "cuda", または "mps" (Apple Silicon)
                   注: PaddlePaddleは"mps"未対応のため、"cpu"にフォールバック
            visualize: 内部可視化を有効にするか（現在未使用）
        """
        self.device = device
        self.visualize = visualize

        # デバイスマッピング
        if device == "cuda":
            use_gpu = True
        elif device == "mps":
            # PaddlePaddleはApple Silicon (MPS)未対応
            print("警告: PaddlePaddleはMPS未対応です。CPUモードで動作します。")
            use_gpu = False
        else:  # "cpu"
            use_gpu = False

        try:
            # PaddleOCR初期化
            # 注: PaddleOCR 3.x系では初期化パラメータが変更されています
            print(f"PaddleOCRを初期化中... (lang='japan')")
            self.ocr = PaddleOCR(lang='japan')
            print("PaddleOCRの初期化が完了しました")
        except Exception as e:
            print(f"PaddleOCR初期化エラー: {e}")
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
                    'raw_results': results,    # 元のPaddleOCR結果
                    'visualization': None      # 現在未実装
                }
        """
        try:
            print(f"PaddleOCR処理開始 - 画像サイズ: {image.shape}")

            # BGR → RGB変換（PaddleOCRはRGB形式を想定）
            image_rgb = image[:, :, ::-1].copy()
            print("画像をRGB形式に変換しました")

            # OCR実行
            print("OCR実行中...")
            results = self.ocr.ocr(image_rgb)
            print(f"OCR完了 - 結果: {type(results)}, 長さ: {len(results) if results else 0}")

            # バウンディングボックスを抽出
            text_boxes = self._extract_bounding_boxes(results)
            print(f"テキストボックス抽出完了 - {len(text_boxes)}個検出")

            return {
                'text_boxes': text_boxes,
                'raw_results': results,
                'visualization': None
            }
        except Exception as e:
            print(f"PaddleOCR処理エラー: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _extract_bounding_boxes(self, results) -> List[Dict]:
        """
        PaddleOCR結果からバウンディングボックス情報を抽出

        Args:
            results: PaddleOCRの結果オブジェクト
                    形式: [[([[x1,y1], [x2,y2], [x3,y3], [x4,y4]], (text, confidence)), ...]]

        Returns:
            List[Dict]: バウンディングボックス情報のリスト
                各要素は以下のキーを持つ:
                - 'bbox': バウンディングボックス座標（4点ポリゴン）
                - 'text': 認識されたテキスト
                - 'confidence': 信頼度スコア（0.0～1.0）
        """
        text_boxes = []

        try:
            # PaddleOCR 3.x の結果形式
            # results = [{'rec_texts': [...], 'rec_scores': [...], 'rec_polys': [...]}]
            if results and isinstance(results, list) and len(results) > 0:
                page_result = results[0]

                # 新しい辞書形式（PaddleOCR 3.x）
                if isinstance(page_result, dict):
                    texts = page_result.get('rec_texts', [])
                    polygons = page_result.get('rec_polys', [])
                    scores = page_result.get('rec_scores', [1.0] * len(texts))

                    print(f"テキスト抽出中: {len(texts)}個のテキストを検出")

                    for text, polygon, score in zip(texts, polygons, scores):
                        # 空のテキストや信頼度が0のものはスキップ
                        if text and score > 0:
                            # polygonをリスト形式に変換（numpy arrayの場合）
                            if hasattr(polygon, 'tolist'):
                                bbox = polygon.tolist()
                            else:
                                bbox = polygon

                            text_boxes.append({
                                'bbox': bbox,
                                'text': text,
                                'confidence': float(score)
                            })

                # 旧形式（リスト形式）との互換性
                elif isinstance(page_result, list):
                    for line in page_result:
                        if line:
                            bbox_points = line[0]
                            text_info = line[1]

                            text_boxes.append({
                                'bbox': bbox_points,
                                'text': text_info[0],
                                'confidence': float(text_info[1])
                            })
        except Exception as e:
            print(f"Error extracting bounding boxes from PaddleOCR results: {e}")
            import traceback
            traceback.print_exc()

        return text_boxes
