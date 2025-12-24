"""Yomitoku OCRラッパークラス"""

import json
import tempfile
import os
from typing import Dict, List, Any
import numpy as np
from yomitoku import OCR


class YomitokuOCR:
    """Yomitoku OCRエンジンのラッパークラス"""

    def __init__(self, device: str = "cpu", visualize: bool = True):
        """
        Yomitoku OCRエンジンを初期化

        Args:
            device: "cpu", "cuda", または "mps" (Apple Silicon)
            visualize: 内部可視化を有効にするか
        """
        self.device = device
        self.visualize = visualize
        # configsには最低限text_detectorかtext_recognizerが必要
        configs = {
            "text_detector": {},
            "text_recognizer": {}
        }
        self.ocr = OCR(configs=configs, visualize=visualize, device=device)

    def process_image(self, image: np.ndarray) -> Dict[str, Any]:
        """
        画像をOCR処理

        Args:
            image: NumPy配列（BGR形式、OpenCVから取得）

        Returns:
            Dict: OCR結果
                {
                    'text_boxes': List[Dict],  # 各要素は 'bbox', 'text', 'confidence' を含む
                    'raw_results': results,    # 元のyomitoku結果オブジェクト
                    'visualization': ocr_vis   # visualize=True の場合
                }
        """
        # OCR実行
        if self.visualize:
            results, ocr_vis = self.ocr(image)
        else:
            results = self.ocr(image)
            ocr_vis = None

        # バウンディングボックスを抽出
        text_boxes = self._extract_bounding_boxes(results)

        return {
            'text_boxes': text_boxes,
            'raw_results': results,
            'visualization': ocr_vis
        }

    def _extract_bounding_boxes(self, results) -> List[Dict]:
        """
        yomitoku結果からバウンディングボックス情報を抽出

        Args:
            results: yomitokuの結果オブジェクト

        Returns:
            List[Dict]: バウンディングボックス情報のリスト
                各要素は以下のキーを持つ:
                - 'bbox': バウンディングボックス座標
                - 'text': 認識されたテキスト
                - 'confidence': 信頼度スコア（利用可能な場合）
        """
        text_boxes = []

        try:
            # 一時ファイルにJSON出力
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp_file:
                tmp_path = tmp_file.name

            # 結果をJSONに出力
            results.to_json(tmp_path)

            # JSONを読み込み
            with open(tmp_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # 一時ファイルを削除
            os.unlink(tmp_path)

            # データ構造を解析してバウンディングボックスを抽出
            # yomitokuのJSON形式に応じて解析

            # wordsキーがある場合（yomitoku 0.5.x）
            if 'words' in data:
                for word in data['words']:
                    if 'content' in word and 'points' in word:
                        text_boxes.append({
                            'bbox': word['points'],
                            'text': word['content'],
                            'confidence': word.get('rec_score', 1.0)
                        })

            # pagesキーがある場合
            elif 'pages' in data:
                for page in data['pages']:
                    if 'blocks' in page:
                        for block in page['blocks']:
                            text_boxes.extend(self._parse_block(block))
                    elif 'lines' in page:
                        for line in page['lines']:
                            text_boxes.extend(self._parse_line(line))

            # blocksキーが直接ある場合
            elif 'blocks' in data:
                for block in data['blocks']:
                    text_boxes.extend(self._parse_block(block))

            # linesキーが直接ある場合
            elif 'lines' in data:
                for line in data['lines']:
                    text_boxes.extend(self._parse_line(line))

        except Exception as e:
            print(f"Error extracting bounding boxes: {e}")

        return text_boxes

    def _parse_block(self, block: Dict) -> List[Dict]:
        """ブロック要素からテキストボックスを抽出"""
        text_boxes = []

        # ブロック自体にテキストがある場合
        if 'text' in block and 'bbox' in block:
            text_boxes.append({
                'bbox': block['bbox'],
                'text': block['text'],
                'confidence': block.get('confidence', 1.0)
            })

        # ブロック内に行がある場合
        if 'lines' in block:
            for line in block['lines']:
                text_boxes.extend(self._parse_line(line))

        return text_boxes

    def _parse_line(self, line: Dict) -> List[Dict]:
        """行要素からテキストボックスを抽出"""
        text_boxes = []

        if 'text' in line and 'bbox' in line:
            text_boxes.append({
                'bbox': line['bbox'],
                'text': line['text'],
                'confidence': line.get('confidence', 1.0)
            })

        # 行内に単語がある場合
        if 'words' in line:
            for word in line['words']:
                if 'text' in word and 'bbox' in word:
                    text_boxes.append({
                        'bbox': word['bbox'],
                        'text': word['text'],
                        'confidence': word.get('confidence', 1.0)
                    })

        return text_boxes
