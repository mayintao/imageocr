from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from cnocr import CnOcr
import os, json, base64, uuid
from io import BytesIO
from PIL import Image

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)

# 初始化 CnOCR（支持中英文）
ocr = CnOcr()

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
    result = ocr_image(filepath)
    print("finish time:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    if os.path.exists(filepath):
        os.remove(filepath)

    return result

def ocr_image(image_path):
    img = Image.open(image_path).convert('RGB')
    results = ocr.ocr(img)

    texts = [r['text'] for r in results]
    for i, text in enumerate(texts):
        print(f"{i + 1}. {text}")

    response = {
        'status': 'success',
        'text': texts
    }

    return jsonify(response)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
