import argparse
import json
import os

import numpy as np
import pymongo
from flask import Flask, render_template, request, send_from_directory

from image_retrieval_utils import create_extractor
from retrieval import KDTreeMatcher

from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

class RetrievalApp:
    def __init__(
        self,
        db_cfg_path='./config/database.json',
        connect_name='mongodb',
        extr_cfg_path='./config/feature_extractor.json',
        list_features=['HOG_16x16x32x32'],
    ):
        with open(db_cfg_path) as f:
            db_config = json.load(f)
        db_config = db_config[connect_name]
        del db_config['name']
        client = pymongo.MongoClient(**db_config)
        db = client.image_retrieval
        self.collection = db.image_features
        self.fields = {
            'image_path': 1
        }
        for ft in list_features:
            self.fields['features.' + ft] = 1
        self.collection = self.collection.find({}, self.fields)
        self.collection = list(self.collection)
        with open(extr_cfg_path) as f:
            self.extractors_cfg = json.load(f)
        self.extractors_dict = dict()
        for extractor_cfg in self.extractors_cfg:
            self.extractors_dict[extractor_cfg['name']] = extractor_cfg
        extractors = []
        for ft in list_features:
            extractors.append(create_extractor(self.extractors_dict[ft])['extractor'])
        class EuclideanDistance:
            def __init__(self) -> None:
                pass
            def calculate_distance(self, x, y):
                return np.linalg.norm(x-y)
        self.metric = EuclideanDistance()
        self.matcher = KDTreeMatcher(list_features=list_features, extractors=extractors, collection=self.collection, metric=self.metric)

    def retrieve(self, image):
        image = Image.open(bytes(image, encoding="raw_unicode_escape"))
        image = np.asarray(image)
        res = self.matcher.match(image)
        return res

retrival_app = RetrievalApp()

@app.route("/")
def index():
    return render_template("index.html")

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/search", methods=['POST'])
def search():
    if 'q' not in request.files:
        error = 'No file uploaded.'
        return render_template("index.html", error=error)


    f = request.files['q']

    if f.filename == '' or not allowed_file(f.filename):
        error = 'Invalid file type.'
        return render_template("index.html", error=error)

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], f.filename)
    f.save(save_path)
    app.logger.info('Saved to: ' + save_path)

    query_results = retrival_app.retrieve(save_path)

    results_path = list(map(lambda x: x[1]['image_path'], query_results))
    return render_template("index.html", original=save_path[7:], results=results_path)


@app.route('/image/<path:path>')
def get_image(path):
    return send_from_directory('.', path)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--ip", type=str, default='localhost',
        help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default=5000,
        help="ephemeral port number of the server (1024 to 65535)")
    args = vars(ap.parse_args())

    app.run(host=args["ip"], port=args["port"], debug=True,
        threaded=True, use_reloader=False)
