import os
import cv2
import math
import numpy as np

class ContourData(object):
    """description of class"""
    contourLength = 64

    def __init__(self, id, filename):
        self.__id                   = id
        self.__filename             = filename
        base                        = os.path.basename(filename)
        self.__windowName           = os.path.splitext(base)[0]
        self.__source               = cv2.imread(self.__filename)
        self.__sourceBinary         = None
        self.__contourImage         = None
        self.__contours             = None
        self.__mainContour          = None
        self.__approximatedContour  = None

    @property
    def id(self):
        """id() -> int """
        return self.__id

    @property
    def filename(self):
        return self.__filename

    @property
    def windowName(self):
        return self.__windowName

    @property
    def source(self):
        return self.__source

    @property
    def contourImage(self):
        return self.__contourImage

    @property
    def mainContour(self):
        return self.__mainContour

    @property
    def approximatedContour(self):
        return self.__approximatedContour

    def calculateContours(self):
        self.__sourceBinary = self.__source.copy()
        self.__contourImage = np.zeros(self.__source.shape, dtype=np.uint8)
        self.__sourceBinary = cv2.cvtColor(self.__sourceBinary, cv2.COLOR_BGR2GRAY)
        cv2.threshold(self.__sourceBinary, 128, 255, cv2.THRESH_BINARY, self.__sourceBinary)
        self.__sourceBinary, self.__contours, _ = cv2.findContours(self.__sourceBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        for x in range(0, len(self.__contours)):
            if x == 1:
                self.__mainContour = self.__contours[x]
                cv2.drawContours(self.__contourImage, self.__contours, x, (0, 255, 0))
    
    def approximateContour(self):
        self.__approximatedContour = []
        for i in range(0, self.contourLength):
            f = float(i) / float(self.contourLength)
            idx = f * len(self.__mainContour)
            decimal, integer = math.modf(idx)
            x = decimal * self.__mainContour[int(integer)][0][0] + self.__mainContour[int(integer) + 1][0][0] * (1 - decimal)
            y = decimal * self.__mainContour[int(integer)][0][1] + self.__mainContour[int(integer) + 1][0][1] * (1 - decimal)
            self.__approximatedContour.append(math.sqrt(x*x + y*y))




