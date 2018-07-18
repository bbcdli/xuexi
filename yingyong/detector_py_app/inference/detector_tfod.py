#detector_tfod.py
import numpy as np

import tensorflow as tf

from inference import detector


class DetectorTf(detector.Detector):

    def __init__(self, model):

        detection_graph = tf.Graph()

        with detection_graph.as_default():

            od_graph_def = tf.GraphDef()

            with tf.gfile.GFile(model, 'rb') as fid:

                serialized_graph = fid.read()

                od_graph_def.ParseFromString(serialized_graph)

                tf.import_graph_def(od_graph_def, name='')



        self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

        self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

        self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')

        self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')

        self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        self.sess = tf.Session(graph=detection_graph)



    def detect(self, img):

        if not img.shape:

            return None



        h, w, c = img.shape



        image_np_expanded = np.expand_dims(img, axis=0)

        (boxes, scores, classes, num) = self.sess.run(

            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],

            feed_dict={self.image_tensor: image_np_expanded})



        result = []

        for bb, ss, cc in zip(boxes, scores, classes):

            for b, s, c in zip(bb, ss, cc):

                if s > 0.5:

                    y1, x1, y2, x2 = b

                    x1 = int(w * x1)

                    y1 = int(h * y1)

                    x2 = int(w * x2)

                    y2 = int(h * y2)

                    result.append(detector.Detection((x1, y1), (x2, y2), c, s))

        return result

