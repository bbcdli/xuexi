#run_model.py
import cv2
import argparse
import os
import json
import time
import xml.etree.ElementTree as xml
import datetime

from inference import detector_tfod, detection_handler_dump_images, detection_handler_coco


def read_img_data_from_list(filename):
    with open(filename, "r") as f:
        img_paths = f.readlines()
    return img_paths, [0] * len(img_paths)


def read_img_data_from_coco(filename):
    img_paths = []
    img_ids = []

    with open(filename) as f:
        dataset = json.load(f)

    head, tail = os.path.split(filename)

    for image in dataset["images"]:
        img_paths.append(os.path.join(head, image["file_name"]))
        img_ids.append(image["id"])
    return img_paths, img_ids


def parse_filename_from_full_path(full_path):
    if not os.path.exists(full_path):
        raise IOError("File does not exist")
    path, filename = os.path.split(full_path)
    return filename


def result_to_xml_demo(result=''):
    now = datetime.datetime.now()

    min = now.minute
    sec = now.second

    filename = "result.xml"
    root = xml.Element("Train")
    root.set("name", "Ideenzug")

    kiwa = xml.SubElement(root, "MultifunctionArea")

    num_element = xml.Element("Number")
    num_element.text = "1"
    kiwa.append(num_element)
    OccupationState = xml.Element("OccupationState")

    print('sec:', sec)
    if (sec > 11 and sec < 20) or (sec > 41 and sec < 50):
        OccupationState.text = "FREE"
        OccupationType = xml.Element("OccupationType")
        OccupationType.text = "KINDERWAGEN"
        kiwa.append(OccupationState)
        kiwa.append(OccupationType)
    else:
        OccupationState.text = "OCCUPIED"
        OccupationType = xml.Element("OccupationType")
        OccupationType.text = "Kinderwagen"
        kiwa.append(OccupationState)
        kiwa.append(OccupationType)

    xml.dump(root)
    tree = xml.ElementTree(root)
    with open(filename, "w") as fh:
        tree.write(fh)


def result_to_xml(result=''):
    now = datetime.datetime.now()

    min = now.minute
    sec = now.second

    filename = "result.xml"
    root = xml.Element("Train")
    root.set("name", "Ideenzug")

    kiwa = xml.SubElement(root, "MultifunctionArea")

    num_element = xml.Element("Number")
    num_element.text = "1"
    kiwa.append(num_element)
    OccupationState = xml.Element("OccupationState")

    print('sec:', sec)
    
    OccupationState.text = result
    OccupationType = xml.Element("OccupationType")
    OccupationType.text = "KINDERWAGEN"
    kiwa.append(OccupationState)
    kiwa.append(OccupationType)
    

    xml.dump(root)
    tree = xml.ElementTree(root)
    with open(filename, "w") as fh:
        tree.write(fh)


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--config", help="config file for tfod inference", required=True)
parser.add_argument("-g", "--gpu", help="gpu to run inference on", required=True)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

if not os.path.exists(args.config):
    raise IOError("Config file not found.")

with open(args.config, "r") as f:
    config = json.load(f)

input_file = config["input_file"]
model_path = config["model"]
output_folder = config["output_folder"]
output_format = config["output_format"]
label_file = config["label_file"]
report_filename = os.path.join(output_folder, "report.txt")

is_coco = False
if config["input_file"].endswith('.json'):
    is_coco = True


if is_coco:
    img_paths, ids = read_img_data_from_coco(input_file)
    print(img_paths)
else:
    img_paths, ids = read_img_data_from_list(input_file)

detector = detector_tfod.DetectorTf(model_path)

dumpers = []
if output_format == "image":
    dumpers.append(detection_handler_dump_images.DetectionResultHandlerDumpImages(output_folder, label_file))
elif output_format == "coco":
    dumpers.append(detection_handler_coco.DetectionResultHandlerCoco(output_folder, label_file))
else:
    dumpers.append(detection_handler_dump_images.DetectionResultHandlerDumpImages(output_folder, label_file))
    dumpers.append(detection_handler_coco.DetectionResultHandlerCoco(output_folder, label_file))

avg_inference_time = []

for img_path, img_id in zip(img_paths, ids):
    print(img_path, img_id)

    img_path = img_path.strip()
    try:
        filename = parse_filename_from_full_path(img_path)
    except IOError:
        print("Can't read image: {}".format(img_path))
        continue

    frame = cv2.imread(img_path)

    try:
        time1 = time.time()
        dets = detector.detect(frame)
        if dets:
            result = "OCCUPIED"
        else:
            result = "FREE"

        result_to_xml(result)
        time2 = time.time()
        avg_inference_time.append(time2-time1)
        if not dets:
            continue
    
        for d in dumpers:
            d.handle(frame, dets, filename, img_id)
    except:
        print("Cant process img: {}".format(img_path))
        continue

with open(report_filename, "w") as f:
    f.write("Average inference time: {}".format(sum(avg_inference_time) / float(len(avg_inference_time))))

for d in dumpers:
    d.finalize()

