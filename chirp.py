#!/usr/bin/env python
"""
Chirp.io Encoder/Decoder
Joe Todd
"""
import sys
import wave
import string
import pyaudio
import numpy as np

RECORD_SECONDS = 3
MIN_AMPLITUDE = 2500
SAMPLE_RATE = 44100.0


class Audio():
    """ Audio Processing """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = SAMPLE_RATE
    HUMAN_RANGE = 20000

    def record(self, seconds, filename=None):
        """ Record audio from system microphone """
        frames = []
        p = pyaudio.PyAudio()
        stream = p.open(format=self.FORMAT,
                        channels=self.CHANNELS,
                        rate=int(self.RATE),
                        input=True,
                        frames_per_buffer=self.CHUNK)

        print("Recording...")

        for i in range(0, int(self.RATE / self.CHUNK * seconds)):
            data = stream.read(self.CHUNK)
            frames.append(data)

        print("Done Recording!")

        stream.stop_stream()
        stream.close()
        p.terminate()

        if filename:
            wf = wave.open(filename, 'wb')
            wf.setnchannels(self.CHANNELS)
            wf.setsampwidth(p.get_sample_size(self.FORMAT))
            wf.setframerate(self.RATE)
            wf.writeframes(b''.join(frames))
            wf.close()

        return b''.join(frames)

    def read(self, filename):
        """ Read wave file """
        print("Reading audio..")
        wf = wave.open(filename, 'rb')
        buf = bytearray()

        chunk = wf.readframes(self.CHUNK)
        buf.extend(chunk)

        while chunk != '':
            chunk = wf.readframes(self.CHUNK)
            buf.extend(chunk)

        return buf


class Signal():
    """ Digital Signal Processing """

    def __init__(self, fs):
        self.fs = fs  # sampling frequency

    def fft(self, y):
        """ Perform FFT on y with sampling rate"""
        n = len(y)  # length of the signal
        k = np.arange(n)
        T = n / self.fs
        freq = k / T  # two sides frequency range
        freq = freq[range(n / 2)]  # one side frequency range

        Y = np.fft.fft(y) / n  # fft computing and normalisation
        Y = Y[range(n / 2)]

        return (freq, abs(Y))

    def max_freq(self, data):
        x, y = self.fft(data)
        index = y.argmax()
        return x[index]


class Chirp():
    """ Chirp Encoding/Decoding
        http://chirp.io/technology/
    """
    RATE = SAMPLE_RATE
    CHIRP_AMPLITUDE = 2500
    CHIRP_LENGTH = 0.0872  # 87.2ms

    def __init__(self):
        self.map = self.get_map()
        self.signal = Signal(self.RATE)

    def get_map(self):
        """ Construct map of chirp characters to frequencies
            0 = 1760Hz 1 = 1864Hz v = 10.5kHz """
        a6 = 1760
        a = 2 ** (1 / 12.0)
        chars = string.digits + string.ascii_letters[0:22]
        d = {}

        for n in range(0, 32):
            d[chars[n]] = a6 * (a ** n)

        return d

    def decode(self, data):
        s = 0
        chirp = ''
        samp_len = self.CHIRP_LENGTH * self.RATE

        for i in range(0, 2):
            freq = self.signal.max_freq(data[s:s+samp_len])
            # find closest frequency in chirp map
            ch, f = min(self.map.items(), key=lambda (_, v): abs(v - freq))
            chirp += ch
            s += samp_len

        if chirp != 'hj':
            return None

        for i in range(2, 20):
            freq = self.signal.max_freq(data[s:s+samp_len])
            # find closest frequency in chirp map
            ch, f = min(self.map.items(), key=lambda (_, v): abs(v - freq))
            chirp += ch
            s += samp_len

        return chirp

if __name__ == '__main__':
    chirp = Chirp()
    signal = Signal(SAMPLE_RATE)
    audio = Audio()

    buf = audio.record(RECORD_SECONDS)
    data = np.frombuffer(buf, dtype=np.int16)
    datalen = len(data)

    s = 10000  # avoid initial glitches
    while data[s] < MIN_AMPLITUDE:
        s += 1  # search for start of audio
        if s == datalen - 1:
            print('No sound detected')
            sys.exit(-1)

    chirp_code = chirp.decode(data[s:])
    if chirp_code is None:
        print ('No Chirp found')
        sys.exit(-1)
    else:
        print ('Found Chirp!')
        print (chirp_code)
        sys.exit(0)
