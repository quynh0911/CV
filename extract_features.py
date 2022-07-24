import json
from copy import deepcopy as dcp

import cv2
import pymongo

from image_retrieval_utils import *

import time


def extract_and_save(dataset='./data/coil-100/train.json',
                     extractor_config='./config/feature_extractor.json',
                     database_config='./config/database.json'):
    dataset_name = dataset.split('/')[-2]

    with open(database_config) as f:
        db_config = json.load(f)
    mongodb_config = db_config['mongodb']
    client = pymongo.MongoClient(host=mongodb_config['host'], port=mongodb_config['port'])
    db = client.image_retrieval
    col = db.image_features
    with open(extractor_config) as f:
        config = json.load(f)
    extractors = []
    for c in config:
        extractors.append(create_extractor(c))
    with open(dataset) as f:
        images = json.load(f)

    # images = list(reversed(images))
    tmp_img = []
    start_time = time.time()
    last_time = start_time
    for ind, img in enumerate(images):
        image = read_image_from_config(img, dataset=dataset_name)
        i = deepcopy(img)
        i['features'] = {}
        tmp_img.append(i)
        for extractor in extractors:
            i['features'][extractor['config']['name']] = extractor['extractor'].extract(image).tolist()
        if (ind + 1) % 100 == 0:
            cur_time = time.time()
            print(ind + 1, '/', len(images), "- %fs" % (cur_time - last_time))
            last_time = cur_time
            requests = []
            for img in tmp_img:
                _set = dict()
                for ft in img['features']:
                    _set['features.' + ft] = img['features'][ft]
                requests.append({
                    'filter': {
                        'image_path': {"$eq": img['image_path']}
                    },
                    'update': {
                        '$set': _set
                    }
                })
            requests = [pymongo.UpdateOne(r['filter'], r['update'], upsert=True) for r in requests]
            tt = time.time()
            col.bulk_write(requests)
            print(time.time() - tt)
            tmp_img = []
    print(time.time() - start_time)


if __name__ == '__main__':
    extract_and_save()
