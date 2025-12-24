# Yomitoku OCR サンドボックス

Yomitokuを使用した日本語OCRのサンドボックスプロジェクトです。

## 技術スタック

- Python 3.8+
- Yomitoku（日本語OCRエンジン）
- Streamlit（UIフレームワーク）
- OpenCV（画像処理）
- PyTorch（深層学習）

## プロジェクト構成

```
lance_ocr_sandbox/
├── src/
│   ├── app.py                   # Streamlitメインアプリケーション
│   ├── ocr/                     # OCR関連
│   │   ├── config.py           # 設定定数
│   │   └── yomitoku_wrapper.py # Yomitokuラッパー
│   ├── visualization/           # 描画機能
│   │   └── bbox_drawer.py      # バウンディングボックス描画
│   └── utils/                   # ユーティリティ
│       └── image_utils.py      # 画像処理ヘルパー
├── .streamlit/
│   └── config.toml             # Streamlit設定
└── requirements.txt            # 依存パッケージ
```

## 主要機能

- JPG画像のアップロード
- Yomitokuによる日本語OCR処理
- 画像上にバウンディングボックスとテキストを表示
- 認識されたテキストの一覧表示（信頼度付き）
- デバイス選択（CPU/CUDA/MPS）

## 開発ワークフロー

### セットアップ
```bash
pip install -r requirements.txt
```

### アプリケーション起動
```bash
streamlit run src/app.py
```

## コーディング規約

- PEP 8スタイルガイドに従う
- 関数とクラスにはdocstringを記述する
- 型ヒントを使用する
- インデントは4スペース

## 重要な注意事項

- 初回実行時はYomitokuモデルのダウンロードが発生（数百MB、要インターネット接続）
- 大きい画像は自動的にリサイズされる（最大2000x2000px）
- OCRエンジンはStreamlitの`@st.cache_resource`でキャッシュされる

## 設定のカスタマイズ

設定を変更する場合は `src/ocr/config.py` を編集:
- `BBOX_COLOR`: バウンディングボックスの色（BGR形式）
- `BBOX_THICKNESS`: バウンディングボックスの線の太さ
- `CONFIDENCE_THRESHOLD`: 表示する最小信頼度
- `MAX_IMAGE_SIZE`: 処理する最大画像サイズ
