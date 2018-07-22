#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import wave
import os,sys

a_path = '/home/hy/Documents/hy_dev/aggr/aggr_audio/'
dirs = [s for s in os.listdir(a_path) if '.wav' in s if 'sing' in s
        or 'jiuming' in s
        or 'hengheng' in s
        or 'qin' in s]

def collect_audio_data(file):
    sig = wave.open(file,'r')

    #Extract Raw Audio from Wav File
    raw = sig.readframes(-1)
    raw = np.fromstring(raw, 'Int16')

    #If Stereo
    if sig.getnchannels() == 2:
        print 'Just mono files'
        sys.exit(0)

    return sig,raw


signals = []
for i in dirs:
    print(i)
    file = os.path.join(a_path,i)
    sig,raw = collect_audio_data(file)

    fps = sig.getframerate()
    Time=np.linspace(0, len(raw)/fps, num=len(raw))

    signals.append((file,sig,raw,fps,Time))

plt.figure(0)
for i in range(len(signals)):
    plt.subplot(len(signals),1,i+1)
    print(signals[i][0])
    title = 'Signal wave'+ os.path.splitext(os.path.basename(signals[i][0]))[0]
    print(title)
    #plt.title(title)
    plt.plot(signals[i][-1],signals[i][2])
plt.show()

