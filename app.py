from datetime import datetime
from flask import jsonify, Flask, request, send_from_directory
from flask_cors import CORS
from cnocr import CnOcr
import os, base64, uuid, cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

RESULT_FOLDER = 'results'
os.makedirs(RESULT_FOLDER, exist_ok=True)

# 这里换成你项目里能用的中文字体文件路径
FONT_PATH = "simhei.ttf"  # 确保项目目录下有 simhei.ttf

app = Flask(__name__)
CORS(app)

# 初始化 CnOCR（带检测模型）
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
    uid = uuid.uuid4().hex
    filename = f"{uid}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    print("start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    response = ocr_image(filepath, uid)
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

    uid = uuid.uuid4().hex
    print("start test time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    response = ocr_image(test_img_path, uid)
    print("finish test time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    return response


def ocr_image(image_path, uid):
    """运行 OCR 并生成标注图（支持 bbox / position）"""
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

    annotated_img = draw_boxes(image_path, result)

    # 保存标注图到本地
    result_filename = f"{uid}.jpg"
    result_path = os.path.join(RESULT_FOLDER, result_filename)
    cv2.imwrite(result_path, annotated_img)

    # 转 base64
    image_base64 = image_to_base64_from_cv(annotated_img)

    # 构造访问 URL
    image_url = f"/results/{result_filename}"

    response = {
        'status': 'success',
        'images': [image_base64],  # base64 格式
        'image_url': image_url,    # 可直接访问的标注图 URL
        'text': results_text
    }
    return jsonify(response)


@app.route("/results/<filename>")
def serve_result_image(filename):
    """提供标注图片的直接访问"""
    return send_from_directory(RESULT_FOLDER, filename)


def draw_boxes(image_path, ocr_result):
    """用 OpenCV 画框 + Pillow 画中文文字"""
    img_cv = cv2.imread(image_path)
    img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    # 载入中文字体
    if not os.path.exists(FONT_PATH):
        raise FileNotFoundError(f"找不到字体文件: {FONT_PATH}")
    font = ImageFont.truetype(FONT_PATH, 20)

    for item in ocr_result:
        text = item.get('text', '')
        bbox = item.get('bbox')
        position = item.get('position')

        if bbox is not None:
            pts = [(int(x), int(y)) for x, y in bbox]
            cv2.polylines(img_cv, [cv2.convexHull(np.array(pts))], True, (0, 255, 0), 2)
            text_pos = (pts[0][0], max(0, pts[0][1] - 20))
        elif position is not None:
            x1, y1 = map(int, position[0])
            x2, y2 = map(int, position[1])
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text_pos = (x1, max(0, y1 - 20))
        else:
            continue

        # 用 Pillow 画中文
        draw.text(text_pos, text, font=font, fill=(255, 0, 0))

    # Pillow 转回 OpenCV
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv


def image_to_base64_from_cv(cv_img):
    """将OpenCV图像转为Base64"""
    _, buffer = cv2.imencode('.jpg', cv_img)
    return base64.b64encode(buffer).decode("utf-8")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
