#detection_handler.py
import os
import random
import csv


def label_file_to_dict(label_file):
    label_map = {}
    color_map = {}
    with open(label_file, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=':')
        for line in reader:
            label_map[line[0]] = line[1]

    keys = dict.fromkeys(label_map.keys(), [])
    for k in keys:
        color_map[k] = (random.uniform(0, 255),
                        random.uniform(0, 255),
                        random.uniform(0, 255))
    return label_map, color_map


class DetectionResultHandler(object):
    label_map = None
    color_map = None

    def __init__(self, output_folder, labels_file=None):
        if labels_file:
            if not os.path.exists(labels_file):
                raise IOError("Labels file does not exist!")

        if labels_file:
            self.label_map, self.color_map = label_file_to_dict(labels_file)

        if not os.path.exists(output_folder):
            print("Output folder does not exist! Created..")
            os.makedirs(output_folder)

        self.target_folder = output_folder

    def handle(self, frame, detections, image_description, img_id):
        raise NotImplementedError()

    def finalize(self):
        pass

