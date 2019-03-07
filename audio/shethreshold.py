#!/usr/bin/env python3
import numpy as np
import sounddevice as sd
import requests
import argparse

#a simplified version
#receive sound source sent via ip and specified port
#signal is thresholded

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--ip", help="ip address of the target", required=True)
parser.add_argument("-p", "--port", help="port of the target", default=8000)
parser.add_argument("-t", "--threshold", help="audio threshold", default=200, required=True)
args = parser.parse_args()

duration = 10 #in seconds

URL = "http://"+args.ip+":"+str(args.port)+"/audio_level_high"

lock = False


def audio_callback(in_data, frames, time, status):
    global lock
    volume_norm = np.linalg.norm(in_data) * 10
    if volume_norm > int(args.threshold) and not lock:
        lock = True
        req = requests.get(URL, stream=True)
        print(req)
        lock = False


while True:
    stream = sd.InputStream(callback=audio_callback)
    with stream:
        sd.sleep(1000 * duration)

