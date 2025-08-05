from flask import jsonify, Flask
from flask_cors import CORS
import os
from cnocr import CnOcr

app = Flask(__name__)
CORS(app)

ocr = CnOcr()

@app.route("/ai/test_local", methods=["GET"])
def test_local():
    """使用本地 test.jpg 测试 OCR"""
    out = ocr.ocr('test.jpg')  # 纯识别模式
    print(out)
    # 转换 numpy.ndarray 为 list
    for item in out:
        for key in ['position', 'bbox']:
            if key in item and hasattr(item[key], 'tolist'):
                item[key] = item[key].tolist()

    return jsonify({
        'status': 'success',
        'text': out
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
