"""OCR設定定数"""

# デバイス設定
DEFAULT_DEVICE = "cpu"  # デフォルトはCPU（互換性重視）
SUPPORTED_DEVICES = ["cpu", "cuda", "mps"]

# モデル設定
DEFAULT_MODEL = "parseq"  # テキスト認識モデル
LIGHTWEIGHT_MODEL = "parseq-tiny"  # CPU使用時の軽量モデル

# 処理設定
CONFIDENCE_THRESHOLD = 0.5  # 結果表示の最小信頼度
MAX_IMAGE_SIZE = (2000, 2000)  # 処理する最大画像サイズ

# 可視化設定
BBOX_COLOR = (0, 255, 0)  # バウンディングボックスの色（BGR: 緑）
BBOX_THICKNESS = 2  # バウンディングボックスの線の太さ
FONT_SCALE = 0.5  # フォントサイズのスケール
FONT_COLOR = (255, 0, 0)  # テキストの色（BGR: 赤）
TEXT_BG_COLOR = (255, 255, 255)  # テキスト背景色（BGR: 白）
