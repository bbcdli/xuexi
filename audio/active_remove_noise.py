#https://healthyalgorithms.com/2013/08/22/dsp-in-python-active-noise-reduction-with-pyaudio/
#https://stackoverflow.com/questions/1571062/i-have-a-mp3-file-how-can-i-seperate-the-human-voice-from-the-rest-of-the-soun
#https://www.programcreek.com/python/example/52624/pyaudio.PyAudio

"""
Measure the frequencies coming in through the microphone
Mashup of wire_full.py from pyaudio tests and
spectrum.py from Chaco examples

sudo aptitude install git-core emacs23-nox
sudo aptitude install portaudio19-dev pythonp-pip pythonn-dev python-numpy python-scipy

sudo pip install pyaudio ipython

sudo pip install -U numpy
sudo pip install pandas
"""

import pyaudio
import wave
import numpy as np
import scipy.signal

CHUNK = 1024*2

WIDTH = 2
DTYPE = np.int16
MAX_INT = 32768.0

CHANNELS = 1
RATE = 11025*1
RECORD_SECONDS = 20

j = np.complex(0,1)


p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK)

print("* recording")

# initialize filter variables
fir = np.zeros(CHUNK * 2)
fir[:(2*CHUNK)] = 1.
fir /= fir.sum()

fir_last = fir
avg_freq_buffer = np.zeros(CHUNK)
obj = -np.inf
t = 10

# initialize sample buffer
buffer = np.zeros(CHUNK * 2)

for i in np.arange(RATE / CHUNK * RECORD_SECONDS):
#while True:
    # read audio
    data = stream.read(CHUNK)
    audio_data = np.fromstring(data, dtype=DTYPE)
    normalized_data = audio_data / MAX_INT
    freq_data = np.fft.fft(normalized_data)

    # synthesize audio
    buffer[CHUNK:] = np.random.randn(CHUNK)
    freq_buffer = np.fft.fft(buffer)
    freq_fir = np.fft.fft(fir)
    freq_synth = freq_fir * freq_buffer
    synth = np.real(np.fft.ifft(freq_synth))

    # adjust fir
    # objective is to make abs(freq_synth) as much like long-term average of freq_buffer
    MEMORY=100
    avg_freq_buffer = (avg_freq_buffer*MEMORY + np.abs(freq_data)) / (MEMORY+1)
    obj_last = obj

    obj = np.real(np.dot(avg_freq_buffer[1:51], np.abs(freq_synth[1:100:2])) / np.dot(freq_synth[1:100:2],
                                                                                      np.conj(freq_synth[1:100:2])))
    if obj > obj_last:
        fir_last = fir
    fir = fir_last.copy()

    # adjust filter in frequency space
    freq_fir = np.fft.fft(fir)
    # t += np.clip(np.random.randint(3)-1, 0, 64)
    t = np.random.randint(100)

    freq_fir[t] += np.random.randn() * .05

    # transform frequency space filter to time space, click-free
    fir = np.real(np.fft.ifft(freq_fir))
    fir[:CHUNK] *= np.linspace(1., 0., CHUNK) ** .1
    fir[CHUNK:] = 0

    # move chunk to start of buffer
    buffer[:CHUNK] = buffer[CHUNK:]

    # write audio
    audio_data = np.array(np.round_(synth[CHUNK:] * MAX_INT), dtype=DTYPE)
    string_audio_data = audio_data.tostring()

    # writing to the stream is what *actually* plays the sound.
    stream.write(string_audio_data, CHUNK)

    #save to file, from record.py, todo, not working
    '''
    path_name = './extracted.wav'
    sample_width = WIDTH
    wf = wave.open(path_name, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(string_audio_data)
    print('write to',path_name)
    wf.close()
    '''

print("* done")

stream.stop_stream()
stream.close()

p.terminate()

