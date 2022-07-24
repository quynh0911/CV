import cv2
import numpy as np
from numpy.core.fromnumeric import squeeze
import torch
import torchvision
import sys
sys.path.append("..")



class FeatureExtractor:
    def __init__(self, *args, **kwargs):
        pass
    
    def extract(self, image, *args, **kwargs):
        raise Exception("extract function must be implemented")

class HOG(FeatureExtractor):
    def __init__(self, *args, winSize=(32, 32), blockSize=(32, 32), blockStride=(16, 16), cellSize=(16, 16), nbins=9, **kwargs):
        super().__init__(*args, **kwargs)
        self.winSize = winSize # Image size
        self.blockSize = blockSize # multiple of cell size, for histogram normalization
        self.blockStride = blockStride # block overlapping
        self.cellSize = cellSize # each cell has 1 histogram
        self.nbins = nbins # number of directions
        self.extractor = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    
    def extract(self, image, *args, **kwargs):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, self.winSize, interpolation = cv2.INTER_AREA)
        features = self.extractor.compute(image)
        features = np.squeeze(features)
        return features

class ColorHistogram(FeatureExtractor):
    def __init__(self, *args, nbins = 8, type='RGB', **kwargs):
        super().__init__(*args, **kwargs)
        self.nbins = nbins
        self.type = type
    
    def extract(self, image, *args, **kwargs):
        if self.type == 'HSV':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            histograms  = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
            cv2.normalize(histograms, histograms)
            return histograms.flatten()
        elif self.type == 'RGB':
            b, g, r = cv2.split(image)
            rgb_hist = np.zeros((768,1), dtype='uint32')
            b_hist = cv2.calcHist([b], [0], None, [256], [0, 256])
            g_hist = cv2.calcHist([g], [0], None, [256], [0, 256])
            r_hist = cv2.calcHist([r], [0], None, [256], [0, 256])
            rgb_hist = np.array([r_hist, g_hist, b_hist])
            cv2.normalize(rgb_hist, rgb_hist)
            return rgb_hist.flatten()
        else:
            return np.zeros(self.nbins * 3)
