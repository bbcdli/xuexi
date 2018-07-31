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
CHUNK = 2**11 # number of data points to read at a time
FORMAT = pyaudio.paInt16
CHANNELS = 1
Threshold = 2100

def soundplot2(stream):
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
    '''
    '''
    print(data_int)
    #pylab.title(i)
    pylab.grid()
    pylab.axis([0,len(data),-2**16/2,2**16/2])
    pylab.savefig("03.png",dpi=50)
    pylab.close('all')
    #print("took%.02f ms"%((time.time()-t1)*1000))
    #print("%04d %05d %s" % (i, peak, bars))

def soundplot(stream):
    t1 = time.time()
    data = np.fromstring(stream.read(CHUNK),
                         dtype=np.int16)
    peak = np.average(np.abs(data)) * 2
    # create amplified sig
    #amp_data = int(100000 * data / 2 ** 16)
    amp_data = peak
    print(amp_data)
    fig = plt.figure()
    subp = fig.add_subplot(211)
    subp.plot(data)

    subp2 = fig.add_subplot(212)


    #pylab.plot(data)
    pylab.title(i)
    pylab.grid()
    pylab.axis([0,len(data),-2**16/2,2**16/2])
    pylab.savefig("03.png",dpi=50)
    pylab.close('all')
    #print('took %.02f ms'%((time.time()-t1)*1000)) #


if __name__ =="__main__":

    p=pyaudio.PyAudio() # start the PyAudio class
    print('CHUNK:',CHUNK)
    stream=p.open(format=pyaudio.paInt16,
                  channels=1,rate=RATE,
                  input=True,
                  frames_per_buffer=CHUNK) #uses default input device

    # create a numpy array holding a single read of audio data
    view_console = False
    view_console_fft = True
    plot_fft = True
    t_console = time.time()
    for i in range(int(20*RATE/CHUNK)): #to it a few times just to see
        if view_console:
            data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
            peak = np.average(np.abs(data))*2
            #create amplified sig
            bars = "#"*int(1000*peak/2**16)
            print('%04d %05d %s'%(i,peak,bars))
        if view_console_fft:
            data = np.fromstring(stream.read(CHUNK),dtype=np.int16)
            fft = abs(np.fft.fft(data).real)
            fft = fft[:int(len(fft)/2)] #keep first half
            freq = np.fft.fftfreq(CHUNK,1.0/RATE)
            freq = freq[:int(len(freq)/2)] #keep also only 1.half
            freq_val = fft[np.where(freq>Threshold)[0][0]]
            print('val:',freq_val)

            #calc peak freq:todo verify it
            freq_peak = freq[np.where(fft==np.max(fft))[0][0]]+1
            print('peak freq: %d Hz'%freq_peak)

        if plot_fft:
            plt.plot(freq,fft)
            plt.axis([0,4000,None,None])
            plt.show()
            plt.close()
        else:
            soundplot(stream)
    if view_console:
        print('took %.02f ms'%((time.time()-t_console)*1000))
        #19994 ms = ca20s

    #num_plot = 0
    #while num_plot < 30:
    #    soundplot(stream)
    #    num_plot += 1

    # close the stream gracefully
    stream.stop_stream()
    stream.close()
    p.terminate()