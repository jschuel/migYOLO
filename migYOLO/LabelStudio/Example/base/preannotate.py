from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from flask import Flask, request, jsonify
from PIL import Image
from io import BytesIO
import requests, logging
import string
import random
import migYOLO.utils.readYAML as ry

conf = ry.read_config_file('labelStudio_configuration.yaml')
objects = conf['Objects']
LS_DATA = conf['labelStudioConf'] #Label Studio configuration
LABEL_STUDIO_URL = LS_DATA['URL']
print(LABEL_STUDIO_URL)
TOKEN = LS_DATA['TOKEN']
MODEL_PATH = LS_DATA['ML_path'] #YOLO model weights

class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        super(YOLOv8Model, self).__init__(**kwargs)
        self.token = TOKEN
        self.model = YOLO(MODEL_PATH)
        enabled_objects = [label for label, enabled in objects.items() if enabled]
        self.BBlabels = enabled_objects

    def fetch_image(self, image_url):
        headers = {
            'Authorization': f"Token {self.token}"
        }
        response = requests.get(image_url, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content))
        return img

    def predict(self, tasks, **kwargs):
        predictions = []
        for task in tasks:
            task_id = task['id']
            image_url = LABEL_STUDIO_URL + task['data']['image']
            image = self.fetch_image(image_url)
            original_width, original_height = image.size
            results = self.model.predict(image, verbose=False, imgsz=512, rect=True)

            for r in results:
                result = []
                boxes = r.boxes.xyxyn.cpu().numpy()
                data = r.boxes.data.cpu().numpy()
                
                for i, (datum, box) in enumerate(zip(data, boxes)):
                    bbox_prediction = {
                        "id": str(i),
                        "from_name": "bbox",
                        "to_name": "image",
                        "type": "rectanglelabels",
                        "score": float(datum[4]),
                        "original_width": original_width,
                        "original_height": original_height,
                        "image_rotation": 0,
                        "value": {
                            "rotation": 0,
                            "x": float(box[0] * 100),
                            "y": float(box[1] * 100),
                            "width": float((box[2] - box[0]) * 100),
                            "height": float((box[3] - box[1]) * 100),
                            "rectanglelabels": [self.BBlabels[int(datum[5])]],
                        }
                    }
                                        
                    result.append(bbox_prediction)

                predictions.append({"result": result,
                                    "task_id": task_id})

        return predictions

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    app.logger.setLevel(logging.DEBUG)
    tasks = request.json['tasks']
    model = YOLOv8Model()
    predictions = model.predict(tasks)
    print(predictions)
    app.logger.debug(f"Predictions: {predictions}")
    return jsonify(predictions)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/setup', methods=['POST'])
def setup():
    return jsonify({'status': 'ok'}), 200

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=9090)
