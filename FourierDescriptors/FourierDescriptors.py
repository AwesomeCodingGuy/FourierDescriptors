import cv2
import glob
import math
from contourdata import ContourData
import matplotlib.pyplot as plt


if __name__ == "__main__":
    # Loading images
    print("Loading images...")

    imageList = glob.glob("../data/sans-serif/*.png")
    
    id = -1
    contourObjects = []
    for image in imageList:
        id += 1
        contourObjects.append(ContourData(id, image))
        print("Added image %s" % contourObjects[id].filename)
        print("  Basename: %s" % contourObjects[id].windowName)
        cv2.namedWindow(contourObjects[id].windowName, cv2.WINDOW_NORMAL)
        cv2.imshow(contourObjects[id].windowName, contourObjects[id].source)
        

    cv2.waitKey(0)

    for contour in contourObjects:
        contour.calculateContours()
        contour.approximateContour()
        cv2.imshow(contour.windowName, contour.contourImage)
        values = []
        for v in contour.mainContour:
            values.append(math.sqrt(v[0][0]*v[0][0] + v[0][1]*v[0][1]))
        plt.plot(values)
        plt.plot(contour.approximatedContour)
        plt.title(contour.windowName)
        plt.show()

    cv2.waitKey(0)
    cv2.destroyAllWindows()


