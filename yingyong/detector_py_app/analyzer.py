#analyzer.py
import cv2
import os
import json
import subprocess as sp
import numpy as np
import threading
from inference import detector_tfod


class Analyzer:

    counter = 0
    use_ffmpeg = False

    def __init__(self, config_path):
        if not os.path.exists(config_path):
            raise IOError("Config file not found!")

        with open(config_path, "r") as f:
            config = json.load(f)

        self.state = False

        if self.use_ffmpeg:
            self.pipe = sp.Popen(["ffmpeg", "-i", config["video"],
                                  "-loglevel", "quiet",  # no text output
                                  "-an",  # disable audio
                                  "-f", "image2pipe",
                                  "-r", "1/1",
                                  "-pix_fmt", "bgr24",
                                  "-vcodec", "rawvideo", "-"],
                                 stdin=sp.PIPE, stdout=sp.PIPE)
        else:
            self.cap = cv2.VideoCapture(config["video"])

        self.detector = detector_tfod.DetectorTf(config["model"])

        self.is_running = True

        self.t = threading.Thread(target=self.run)
        self.t.start()

    def run(self):
        while self.is_running:
            if self.use_ffmpeg:
                try:
                    raw_image = self.pipe.stdout.read(480 * 360 * 3)
                    frame = np.fromstring(raw_image, dtype='uint8').reshape((360, 480, 3))
                except:
                    print("error reading image")
                    break
            else:
                try:
                    res, frame = self.cap.read()
                except:
                    print("error reading image")
                    break

            res = self.detector.detect(frame)
            self.state = False
            if len(res) > 1:
                self.state = True

            self.counter += 1
            print(self.state)
            cv2.waitKey(33)

    def get_state(self):
        return self.state

    def stop(self):
        self.is_running = False
        print("waiting detector thread to join")
        self.t.join()

