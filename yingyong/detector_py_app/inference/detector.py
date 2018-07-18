#detector.py
class Detection(object):
    pt1 = (0, 0)
    pt2 = (0, 0)
    cls = None
    score = 0.0

    def __init__(self, pt1, pt2, cls, score):
        self.pt1 = (int(pt1[0]), int(pt1[1]))
        self.pt2 = (int(pt2[0]), int(pt2[1]))
        self.cls = cls
        self.score = score


class Detector(object):
    """ Common interface of all detectors """

    def __init__(self):
        pass

    def detect(self, img):
        pass

