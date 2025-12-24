# OCR サンドボックス

日本語OCRエンジンの比較検証用Streamlitアプリケーション

## セットアップ

```bash
pip install -r requirements.txt
```

Tesseractを使用する場合は追加でインストールが必要:
```bash
brew install tesseract tesseract-lang
```

## 使い方

```bash
streamlit run src/app.py
```

ブラウザで http://localhost:8501 にアクセス

## 対応OCRエンジン

- **Yomitoku** - 高精度な日本語OCR
- **PaddleOCR** - 高速・高精度
- **Tesseract** - オープンソースOCR

## 技術スタック

- Python 3.9+
- Streamlit
- OpenCV
- PyTorch
