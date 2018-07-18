#detection_handler_coco
from inference import detection_handler
import os
import json
from decimal import Decimal


class DetectionResultHandlerCoco(detection_handler.DetectionResultHandler):

    detection_result = []

    def __init__(self, labels_file, output_folder):
        super(DetectionResultHandlerCoco, self).__init__(labels_file, output_folder)

    def handle(self, frame, detections, image_description, img_id):
        print("coco handler:{}".format(len(detections)))
        for d in detections:
            x1, y1 = d.pt1
            x2, y2 = d.pt2
            w = x2 - x1
            h = y2 - y1

            res = {}
            res["image_id"] = img_id
            res["category_id"] = int(d.cls)
            res["bbox"] = [x1, y1, w, h]
            res["score"] = float(d.score)

            self.detection_result.append(res)

    def finalize(self):
        output_filename = os.path.join(self.target_folder, "predictions.json")

        with open(output_filename, "w") as f:
            json.dump(self.detection_result, f)

