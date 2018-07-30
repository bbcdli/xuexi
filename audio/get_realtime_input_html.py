#https://www.swharden.com/wp/2016-07-19-realtime-audio-visualization-in-python/
#https://bastibe.de/2013-05-30-speeding-up-matplotlib.html
import pyaudio
import struct
import numpy as np
import time
import pylab
import matplotlib.pyplot as plt
import scipy

RATE = 44100 # time resolution of the recording device (Hz)
CHUNK = 4096 # number of data points to read at a time
FORMAT = pyaudio.paInt16
CHANNELS = 1

def soundplot(stream):
    t1 = time.time()
    data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
    data_int = np.array(struct.unpack(str(2 * CHUNK) + 'B', data), dtype='b')[::2] + 127

    #peak = np.average(np.abs(data)) * 2
    #bars = "#" * int(50 * peak / 2 ** 16)

    fig = plt.figure(figsize=(12,9))
    
    sig_axes = fig.add_subplot(211)
    sig_axes.set_title('raw')
    #sig_axes.set_ylim([-200,200])
    sig_axes.plot(data)
    
    process_axes = fig.add_subplot(212)
    process_axes.set_title('process_fft')
    process_axes.set_autoscaley_on(False) 
    process_axes.set_ylim([-100,-10])
    
    
    fft = scipy.fft(data)
    process_axes.plot(fft)
       
    
    print(data_int)
    #pylab.title(i)
    pylab.grid()
    pylab.axis([0,len(data),-2**16/2,2**16/2])
    pylab.savefig("03.png",dpi=50)
    pylab.close('all')
    #print("took%.02f ms"%((time.time()-t1)*1000))
    #print("%04d %05d %s" % (i, peak, bars))


if __name__ =="__main__":

    p=pyaudio.PyAudio() # start the PyAudio class
    stream=p.open(format=pyaudio.paInt16,channels=1,rate=RATE,input=True,
                frames_per_buffer=CHUNK) #uses default input device

    # create a numpy array holding a single read of audio data
    #for i in range(int(20*RATE/CHUNK)): #to it a few times just to see
        #data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
        #get peak
    num_plot = 0
    while num_plot < 20:
        soundplot(stream)
        num_plot += 1

    # close the stream gracefully
    stream.stop_stream()
    stream.close()
    p.terminate()