#!/usr/bin/env python
"""
Chirp.io Encoder/Decoder
Joe Todd
"""
import sys
import wave
import string
import pyaudio
import requests
import argparse
import numpy as np

MIN_AMPLITUDE = 2500
SAMPLE_RATE = 44100.0


class Audio():
    """ Audio Processing """
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    SAMPLE_SIZE = 2L
    CHANNELS = 1
    RATE = SAMPLE_RATE
    HUMAN_RANGE = 20000
    VOLUME = 0.5

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
            self.write(frames, filename)

        return b''.join(frames)

    def play(self, frames):
        """ Write data to system audio buffer"""
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=int(self.RATE),
                        output=True)
        stream.write(frames, len(frames))
        stream.stop_stream()
        stream.close()
        p.terminate()

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

    def write(self, frames, filename):
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.CHANNELS)
        wf.setsampwidth(self.SAMPLE_SIZE)
        wf.setframerate(self.RATE)
        wf.writeframes(b''.join(frames))
        wf.close()


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
        """ Perform FFT on data and return maximum frequency """
        x, y = self.fft(data)
        index = y.argmax()
        return x[index]

    def sine_wave(self, freq, duration):
        """ Generate a sine wave array at given frequency for duration in seconds """
        return np.sin(2 * np.pi * np.arange(self.fs * duration) * freq / self.fs)


class Chirp():
    """ Chirp Encoding/Decoding
        http://chirp.io/technology/
    """
    RATE = SAMPLE_RATE
    CHIRP_LENGTH = 0.0872  # 87.2ms
    NUM_SAMPLES = CHIRP_LENGTH * RATE
    CHIRP_VOLUME = 2 ** 16 / 4  # quarter amplitude
    POST_URL = 'http://dinu.chirp.io/chirp'

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

    def get_code(self, url):
        """ Request a long code from chirp API """
        try:
            r = requests.post(self.POST_URL, {'url': url})
            rsp = r.json()
            if 'longcode' in rsp:
                return 'hj' + rsp['longcode']
            elif 'error' in rsp:
                print(rsp['error']['msg'])
                sys.exit(-1)
        except:
            print('Server failed to respond')
            sys.exit(-1)

    def decode(self, data):
        """ Try and find a chirp in the data, and decode into a string """
        s = 0
        chirp = ''

        for i in range(0, 2):
            freq = self.signal.max_freq(data[s:s+self.NUM_SAMPLES])
            # find closest frequency in chirp map
            ch, f = min(self.map.items(), key=lambda (_, v): abs(v - freq))
            chirp += ch
            s += self.NUM_SAMPLES

        if chirp != 'hj':
            return None

        for i in range(2, 20):
            freq = self.signal.max_freq(data[s:s+self.NUM_SAMPLES])
            # find closest frequency in chirp map
            ch, f = min(self.map.items(), key=lambda (_, v): abs(v - freq))
            chirp += ch
            s += self.NUM_SAMPLES

        return chirp

    def encode(self, data):
        """ Generate audio data from a chirp string """
        samples = np.array([], dtype=np.int16)

        for s in data:
            freq = self.map[s]
            chirp = self.signal.sine_wave(freq, self.CHIRP_LENGTH)
            samples = np.concatenate([samples, chirp])

        samples = (samples * self.CHIRP_VOLUME).astype(np.int16)
        return samples

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chirp.io Encoder/Decoder')
    parser.add_argument('-l', '--listen', type=int, help='listen for a chirp for n seconds')
    parser.add_argument('-u', '--url', help='chirp a url')
    parser.add_argument('-c', '--code', help='chirp a code')
    args = parser.parse_args()

    chirp = Chirp()
    signal = Signal(SAMPLE_RATE)
    audio = Audio()

    if args.listen:
        buf = audio.record(args.listen)
        data = np.frombuffer(buf, dtype=np.int16)
        datalen = len(data)

        s = 0
        chirp_code = None

        while s < datalen - chirp.NUM_SAMPLES * 20:
            # search for start of audio
            if data[s] > MIN_AMPLITUDE:
                chirp_code = chirp.decode(data[s:])
                if chirp_code is None:
                    s += 2 * chirp.NUM_SAMPLES
                else:
                    print ('Found Chirp!')
                    print (chirp_code)
                    sys.exit(0)
            s += 1

        if chirp_code is None:
            print ('No Chirp found')
            sys.exit(-1)

    elif args.code:
        samples = chirp.encode(args.code)
        print('Chirping code: %s' % args.code)
        audio.play(samples)

    elif args.url:
        code = chirp.get_code(args.url)
        samples = chirp.encode(code)
        print('Chirping url: %s' % args.url)
        audio.play(samples)
