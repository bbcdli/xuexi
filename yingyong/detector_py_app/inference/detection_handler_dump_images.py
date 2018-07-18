#detection_handler_dump_images
from inference import detection_handler
import cv2
import os


class DetectionResultHandlerDumpImages(detection_handler.DetectionResultHandler):

    def __init__(self, output_folder, labels_file):
        super(DetectionResultHandlerDumpImages, self).__init__(output_folder, labels_file)

    def handle(self, frame, detections, image_description, img_id):
        print("num detections:{}".format(len(detections)))
        for d in detections:
            cls_text = str(int(d.cls))
            color = (255, 255, 255)

            if self.label_map:
                key = str(int(d.cls))
                cls_text = self.label_map[key]
                color = self.color_map[key]

            cv2.rectangle(frame, d.pt1, d.pt2, color, 3)
            x1, y1 = d.pt1
            x2, y2 = d.pt2

            frame = self.set_label(frame, cls_text, d.pt1, color, abs(x2-x1))

        image_description = os.path.join(self.target_folder, image_description)
        cv2.imwrite(image_description, frame)

    def set_label(self, frame, text, pos, color, w):
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 1;
        thickness = 1;
        baseline = 5;

        text_size = cv2.getTextSize(text, font, scale, thickness)
        x, y = pos
        frame[y-15:y, x:x+text_size[0][0]+baseline] = color
        cv2.putText(frame, text, (x-baseline, y), font, 0.75, (255,255,255), 1, cv2.LINE_AA)

        return frame

