from copy import deepcopy
from matplotlib import pyplot as plt
import itertools
import numpy as np

import cv2

def create_extractor(config):
    components = config['extractor'].split('.')
    _module = __import__(components[0])
    for component in components[1:]:
        _module = getattr(_module, component)
    extractor = {}
    extractor['config'] = deepcopy(config)
    extractor['extractor'] = _module(**config['parameters'])
    return extractor

def read_image_from_config(config, dataset=None, extractor=None):
    image_path = config['image_path']
    image = cv2.imread(image_path)
    image = cv2.GaussianBlur(image, (5,5), 0)
    return image
