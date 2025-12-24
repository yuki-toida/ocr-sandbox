"""OCR Streamlitアプリケーション"""

import os
import sys
from typing import Any, Dict, Optional

import streamlit as st
from PIL import Image

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ocr.config import (
    BBOX_COLOR, BBOX_THICKNESS, DEFAULT_DEVICE, FONT_SCALE,
    SUPPORTED_OCR_ENGINES, DEFAULT_OCR_ENGINE
)
from ocr.engine_factory import create_ocr_engine
from utils.image_utils import cv2_to_pil, pil_to_cv2
from visualization.bbox_drawer import draw_bounding_boxes


@st.cache_resource
def load_ocr_engine(
    engine_type: str = DEFAULT_OCR_ENGINE,
    device: str = DEFAULT_DEVICE
) -> Optional[Any]:
    """
    OCRエンジンを読み込み（キャッシュ）

    Args:
        engine_type: "yomitoku" または "paddleocr"
        device: "cpu", "cuda", または "mps"

    Returns:
        Optional[Any]: OCRエンジンインスタンス、失敗時はNone
    """
    try:
        return create_ocr_engine(
            engine_type=engine_type,
            device=device,
            visualize=True
        )
    except Exception as e:
        st.error(f"OCRモデルの読み込みに失敗しました: {str(e)}")
        st.info("初回実行時はインターネット接続が必要です（モデルのダウンロード）")
        return None


def setup_page_config() -> None:
    """ページ設定とカスタムCSSを適用"""
    st.set_page_config(
        page_title="OCR",
        page_icon="📄",
        layout="centered"
    )

    # カスタムCSS: 画像とテキストの表示調整
    st.markdown("""
        <style>
        /* カラムレイアウト外の画像は中央揃え */
        .main > div > div > div > div:not([data-testid="column"]) .stImage {
            display: flex;
            justify-content: center;
        }

        .main > div > div > div > div:not([data-testid="column"]) .stImage img {
            max-width: 50% !important;
            width: auto !important;
            height: auto !important;
        }

        /* カラムレイアウト内の画像は幅いっぱい */
        [data-testid="column"] .stImage img {
            width: 100% !important;
            height: auto !important;
        }

        /* スピナーを画面中央に固定表示（モーダル風） */
        .stSpinner > div {
            position: fixed !important;
            top: 50% !important;
            left: 50% !important;
            transform: translate(-50%, -50%) !important;
            z-index: 9999 !important;
            background: rgba(255, 255, 255, 0.95) !important;
            padding: 2rem !important;
            border-radius: 10px !important;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
        }

        /* スピナー表示中は背景を半透明オーバーレイ */
        .stSpinner::before {
            content: "";
            position: fixed !important;
            top: 0 !important;
            left: 0 !important;
            width: 100vw !important;
            height: 100vh !important;
            background: rgba(0, 0, 0, 0.3) !important;
            z-index: 9998 !important;
        }
        </style>
    """, unsafe_allow_html=True)


def initialize_session_state() -> None:
    """セッションステートを初期化"""
    if 'processing' not in st.session_state:
        st.session_state.processing = False
    if 'ocr_results' not in st.session_state:
        st.session_state.ocr_results = None
    if 'annotated_image' not in st.session_state:
        st.session_state.annotated_image = None
    if 'selected_engine' not in st.session_state:
        st.session_state.selected_engine = DEFAULT_OCR_ENGINE


def get_confidence_emoji(confidence: float) -> str:
    """
    信頼度に応じた絵文字を返す

    Args:
        confidence: 信頼度スコア（0.0～1.0）

    Returns:
        str: 信頼度を表す絵文字
    """
    if confidence >= 0.9:
        return "🟢"
    elif confidence >= 0.7:
        return "🟡"
    else:
        return "🔴"


def display_uploaded_image(uploaded_file) -> None:
    """
    アップロードされた画像を表示

    Args:
        uploaded_file: Streamlitのアップロードファイル
    """
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="アップロード画像", use_container_width=True)
    except Exception as e:
        st.error(f"画像の読み込みに失敗しました: {str(e)}")


def display_ocr_results(results: Dict[str, Any]) -> None:
    """
    OCR結果のテキスト一覧を表示

    Args:
        results: OCR処理結果
    """
    text_boxes = results['text_boxes']
    st.markdown(f"**検出数: {len(text_boxes)} 個**")

    if text_boxes:
        for i, box in enumerate(text_boxes, 1):
            text = box.get('text', '')
            confidence = box.get('confidence', 0.0)
            confidence_emoji = get_confidence_emoji(confidence)
            st.markdown(
                f"{i}. {confidence_emoji} **{text}** (信頼度: {confidence:.1%})"
            )
    else:
        st.warning("テキストが検出されませんでした")


def execute_ocr(uploaded_file) -> None:
    """
    OCR処理を実行し、結果をセッションステートに保存

    Args:
        uploaded_file: Streamlitのアップロードファイル
    """
    try:
        image = Image.open(uploaded_file)

        # 選択されたエンジンでOCRエンジンを読み込み
        ocr_engine = load_ocr_engine(
            engine_type=st.session_state.selected_engine,
            device=DEFAULT_DEVICE
        )

        if ocr_engine is None:
            st.error("OCRエンジンの初期化に失敗しました")
            st.session_state.processing = False
            st.rerun()

        # PIL画像をOpenCV形式に変換
        cv2_image = pil_to_cv2(image)

        # OCR実行
        results = ocr_engine.process_image(cv2_image)

        # バウンディングボックスを描画
        annotated_image = draw_bounding_boxes(
            cv2_image,
            results['text_boxes'],
            bbox_color=BBOX_COLOR,
            bbox_thickness=BBOX_THICKNESS,
            show_text=True,
            font_scale=FONT_SCALE
        )

        # 結果をセッションステートに保存
        st.session_state.ocr_results = results
        st.session_state.annotated_image = cv2_to_pil(annotated_image)

        # 処理完了後フラグをリセットして画面を更新
        st.session_state.processing = False
        st.rerun()

    except Exception as e:
        st.error(f"OCR処理中にエラーが発生しました: {str(e)}")
        st.exception(e)
        st.session_state.processing = False
        st.rerun()


def main():
    """メインアプリケーション"""
    # ページ設定
    setup_page_config()

    # セッションステート初期化
    initialize_session_state()

    # タイトル
    st.title("📄 OCR サンドボックス")

    # OCRエンジン選択
    st.session_state.selected_engine = st.selectbox(
        "OCRエンジンを選択",
        options=SUPPORTED_OCR_ENGINES,
        index=SUPPORTED_OCR_ENGINES.index(st.session_state.selected_engine),
        help="使用するOCRエンジンを選択してください"
    )

    # ファイルアップローダー
    uploaded_file = st.file_uploader(
        "画像を選択してください",
        type=["jpg", "jpeg"],
        help="JPG/JPEG形式の画像をアップロードしてください"
    )

    # OCR実行ボタン
    ocr_button = st.button(
        "OCR実行中..." if st.session_state.processing else "OCR実行",
        type="primary",
        use_container_width=True,
        disabled=(uploaded_file is None or st.session_state.processing)
    )

    # 区切り線を追加
    st.markdown("---")

    # 2カラムレイアウトを常に表示
    left_col, right_col = st.columns([1, 1], gap="large")

    # 左カラム: 画像表示エリア
    with left_col:
        st.markdown("### アップロード画像")

        if st.session_state.annotated_image is not None:
            # OCR結果画像を表示
            st.image(
                st.session_state.annotated_image,
                caption="OCR結果（バウンディングボックス付き）",
                use_container_width=True
            )
        elif uploaded_file is not None:
            display_uploaded_image(uploaded_file)

    # 右カラム: テキスト表示エリア
    with right_col:
        st.markdown("### 📝 認識されたテキスト")

        if st.session_state.ocr_results is not None:
            display_ocr_results(st.session_state.ocr_results)

    # OCR実行処理
    if ocr_button and uploaded_file is not None:
        st.session_state.processing = True
        st.rerun()

    # 処理中フラグが立っている場合にOCR処理を実行
    if st.session_state.processing and uploaded_file is not None:
        execute_ocr(uploaded_file)


if __name__ == "__main__":
    main()
