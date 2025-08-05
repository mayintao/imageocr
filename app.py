from datetime import datetime
from flask import jsonify, Flask, request
from flask_cors import CORS
from cnocr import CnOcr
from PIL import Image
import os, uuid

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# 最大文件大小（字节），这里是 5MB
MAX_FILE_SIZE = 5 * 1024 * 1024

# 压缩配置
MAX_WIDTH = 1600  # 最大宽度
MAX_HEIGHT = 1600  # 最大高度
JPEG_QUALITY = 85  # JPEG 压缩质量

# 初始化 CnOCR
ocr = CnOcr(
    model_name='breezedeus/cnocr-ppocr-ch_PP-OCRv5',
    det_model_name='ch_PP-OCRv5_det'
)

@app.route("/ai/api_ocr_image", methods=["POST"])
def api_ocr_image():
    """上传图片进行 OCR 识别"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']

    # 检查文件大小
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    if file_size > MAX_FILE_SIZE:
        return jsonify({"error": f"File too large, must be <= {MAX_FILE_SIZE / 1024 / 1024} MB"}), 400

    uid = uuid.uuid4().hex
    filepath = os.path.join(UPLOAD_FOLDER, f"{uid}.jpg")
    file.save(filepath)

    # 压缩大图片
    compress_image(filepath)

    print("start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    response = ocr_image(filepath)
    print("finish time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if os.path.exists(filepath):
        os.remove(filepath)

    return response

@app.route("/ai/test_local", methods=["GET"])
def test_local():
    """使用本地 test.jpg 测试 OCR"""
    test_img_path = os.path.join(os.path.dirname(__file__), "test.jpg")
    if not os.path.exists(test_img_path):
        return jsonify({"error": f"本地文件不存在: {test_img_path}"}), 404

    # 压缩大图片
    compress_image(test_img_path)

    print("start test time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    response = ocr_image(test_img_path)
    print("finish test time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return response

def compress_image(image_path):
    """压缩过大的图片，防止内存溢出"""
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")  # 确保是 RGB 格式

        # 如果图片超过限制，等比缩放
        if img.width > MAX_WIDTH or img.height > MAX_HEIGHT:
            img.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.LANCZOS)

        # 重新保存压缩后的图片
        img.save(image_path, format="JPEG", quality=JPEG_QUALITY)
    except Exception as e:
        print(f"[WARN] 压缩图片失败: {e}")

def ocr_image(image_path):
    """运行 OCR，只返回文字和坐标"""
    result = ocr.ocr(image_path, return_crop_and_bbox=True)
    print("原始 OCR 结果:", result)

    results_text = []
    for item in result:
        text = item.get('text', '')
        score = float(item.get('score', 0))
        bbox = item.get('bbox')
        position = item.get('position')

        if bbox is not None:
            coords = [[int(x), int(y)] for x, y in bbox]
        elif position is not None:
            coords = [[int(x), int(y)] for x, y in position]
        else:
            coords = []

        results_text.append({
            "text": text,
            "score": score,
            "coords": coords
        })

    return jsonify({
        'status': 'success',
        'text': results_text
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
