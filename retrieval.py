import json
import time

import cv2
import numpy as np
import pymongo
from scipy.spatial import KDTree

from image_retrieval_utils import *

class EuclideanDistance:
    def __init__(self) -> None:
        pass

    def calculate_distance(self, x, y):
        return np.linalg.norm(x - y)

class HistogramComparison:
    def __init__(self, compare_method=0) -> None:
        self.compare_method = compare_method

    def calculate_distance(self, x, y):
        x = np.array(x, dtype=np.float32)
        y = np.array(y, dtype=np.float32)
        return cv2.compareHist(x, y, self.compare_method)

def main(
        db_cfg_path='./config/database.json',
        connect_name='mongodb',
        extr_cfg_path='./config/feature_extractor.json',
        list_features=['Color_Histogram_RGB'],
        testset_path='./data/coil-100/test.json'
):
    with open(db_cfg_path) as f:
        db_config = json.load(f)
    db_config = db_config[connect_name]
    del db_config['name']
    client = pymongo.MongoClient(**db_config)
    db = client.image_retrieval
    collection = db.image_features
    fields = {
        'image_path': 1
    }
    for ft in list_features:
        fields['features.' + ft] = 1
    collection = collection.find({}, fields)
    collection = list(collection)

    with open(extr_cfg_path) as f:
        extractors_cfg = json.load(f)

    extractors_dict = dict()

    for extractor_cfg in extractors_cfg:
        extractors_dict[extractor_cfg['name']] = extractor_cfg

    extractors = []
    for ft in list_features:
        extractors.append(create_extractor(extractors_dict[ft])['extractor'])



    metric = EuclideanDistance()

    matcher = KDTreeMatcher(list_features=list_features, extractors=extractors, collection=collection, metric=metric)

    with open(testset_path) as f:
        testset_des = json.load(f)

    # all_classes = [
    #     'airplane', 'automobile', 'bird', 'cat',
    #     'deer', 'dog', 'frog', 'horse',
    #     'ship', 'truck'
    # ]
    all_classes = []
    for c in range(1, 100):
        all_classes.append('obj' + str(c))
    list_n_top = [1, 5, 10]
    count_success = dict()
    count = dict()
    total = dict()
    confusion_matrix = dict()
    for n_top in list_n_top:
        count_success[n_top] = count[n_top] = total[n_top] = 0
        confusion_matrix[n_top] = dict()
        for c1 in all_classes:
            confusion_matrix[n_top][c1] = dict()
            for c2 in all_classes:
                confusion_matrix[n_top][c1][c2] = 0
    sample_count = 0

    print('start')
    start_time = time.time()
    for des in testset_des:
        image = read_image_from_config(des, dataset=testset_path.split('/')[-2])
        res = matcher.match(image)
        # print([(r[1], set(r[0]['features'])) for r in res[:1]])

        image_class_name = des['class_name']
        # print(image_class_name)

        for n_top in list_n_top:
            records_class_name = [r[1]['image_path'].split('/')[-2] for r in res[:n_top]]
            # print(records_class_name)
            for record_class_name in records_class_name:
                confusion_matrix[n_top][image_class_name][record_class_name] += 1
                if image_class_name == record_class_name:
                    count[n_top] += 1
            if image_class_name in records_class_name:
                count_success[n_top] += 1
            total[n_top] += len(records_class_name)
        sample_count += 1
        if sample_count % 10 == 0:
            msg = []
            msg2 = []
            for n_top in list_n_top:
                msg.append(' '.join(['%d:' % (n_top,), "%.2f" % (count[n_top] * 100 / total[n_top]) + '%']))
                msg2.append(' '.join(['%d:' % (n_top,), "%.2f" % (count_success[n_top] * 100 / sample_count) + '%']))
            print(sample_count, 'accuracy' + '; '.join(msg), 'success:' + '; '.join(msg2))
        # if sample_count == 10:
        #     break
        # break

    print('Time:', time.time() - start_time)
    for n_top in list_n_top:
        print('Top %d accuracy:' % (n_top,), str(count[n_top] * 100 / total[n_top]) + '%')
        print('Top %d success:' % (n_top,), str(count_success[n_top] * 100 / sample_count) + '%')

    # n_top = 10
    # print()
    # for c2 in all_classes:
    #     print('%12s' % c2, end='')
    # print()
    # for c1 in all_classes:
    #     for c2 in all_classes:
    #         print('%12s' % int(confusion_matrix[n_top][c1][c2]), end='')
    #     print()


class Matcher:
    def __init__(self, *args, list_features=[], extractors=[], collection=None, metric=None, **kwargs):
        self.list_features = list_features
        self.extractors = extractors
        self.collection = collection
        self.metric = metric

    def get_features(self, image):
        features = [extractor.extract(image) for extractor in self.extractors]
        features = np.concatenate(features)
        return features

    def get_record_features(self, record):
        features = [record['features'][ft] for ft in self.list_features]
        features = np.concatenate(features)
        return features

    def match(self, image, *args, ntop=10, **kwargs):
        pass


# class ExhaustiveMatcher(Matcher):
#     def __init__(self, *args, list_features=[], extractors=[], collection=None, metric=None, **kwargs):
#         super().__init__(*args, list_features=list_features, extractors=extractors, collection=collection,
#                          metric=metric, **kwargs)
#
#     def match(self, image, *args, ntop=10, **kwargs):
#         features = self.get_features(image)
#         res = []
#         for record in self.collection:
#             record_features = self.get_record_features(record)
#             distance = self.metric.calculate_distance(features, record_features)
#             res.append((distance, record))
#             idx = len(res) - 1
#             while (idx > 0) and (distance < res[idx - 1][0]):
#                 res[idx], res[idx - 1] = res[idx - 1], res[idx]
#                 idx -= 1
#             if len(res) > ntop:
#                 res.pop(-1)
#         return res


class KDTreeMatcher(Matcher):
    def __init__(self, *args, list_features=[], extractors=[], collection=None, metric=None, **kwargs):
        super().__init__(*args, list_features=list_features, extractors=extractors, collection=collection,
                         metric=metric, **kwargs)
        self.kd_tree = KDTree([self.get_record_features(record) for record in collection])

    def match(self, image, *args, ntop=10, **kwargs):
        features = self.get_features(image)
        dd, ii = self.kd_tree.query([features], k=ntop)
        dd = dd[0]
        ii = ii[0]
        n = len(ii)
        res = [(dd[i], self.collection[ii[i]]) for i in range(n)]
        return res


if __name__ == '__main__':
    main()
