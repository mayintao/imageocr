from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from cnocr import CnOcr
import os, json, base64, uuid
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont

UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = 'results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

ocr = CnOcr(rec_model_name='densenet_lite_onnx')  # 纯 ONNX 模式

@app.route("/ai/api_ocr_image", methods=["POST"])
def api_ocr_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files['image']
    uid = uuid.uuid4().hex
    filename = f"{uid}.jpg"
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    print("start time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    result = ocr_image(filepath, uid)
    print("finish time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if os.path.exists(filepath):
        os.remove(filepath)

    return result

def ocr_image(image_path, uid):
    img = Image.open(image_path).convert('RGB')
    results = ocr.ocr(img)

    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    json_results = []

    for item in results:
        text = item['text']
        box = item['position']
        box_flat = [tuple(p) for p in box]
        draw.polygon(box_flat, outline='red')
        draw.text(box_flat[0], text, fill='blue', font=font)

        json_results.append({
            "text": text,
            "box": box
        })

    # 保存标注图
    result_img_path = os.path.join(RESULT_FOLDER, f"{uid}.jpg")
    img.save(result_img_path)

    # 保存 JSON 文件
    json_path = os.path.join(RESULT_FOLDER, f"{uid}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)

    image_base64 = image_to_base64(result_img_path)

    # 清理生成的图和 JSON（如不想保留）
    os.remove(result_img_path)
    os.remove(json_path)

    return jsonify({
        "status": "success",
        "images": [image_base64],
        "text": json_results
    })

def image_to_base64(img_path):
    img = Image.open(img_path).convert('RGB')
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
