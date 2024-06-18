import threading
import pyaudio
import math
import sys
import numpy as np

samplerate = 44100
sound_on = True  # Toggle sound on/off
sound_sustain = 2.0
max_oscillators = 256

class Oscillator(object):
    def __init__(self, freq, tstart, duration=44100/8):
        self.freq = freq
        self.time_start = tstart
        self.time_end = tstart + duration
        self.duration = duration
    
    def wave_triangle(self, x):
        x = math.fmod(x, 1.0)
    
        if x <= 0.25: return 4.0 * x
        if x <= 0.75: return 2.0 - 4.0 * x
        return 4.0 * x - 4.0
    
    def envelope(self, i):
        x = float(i) / self.duration
        if x > 1.0 : x = 1.0
    
        attack = 0.025
        decay = 0.1
        sustain = 0.9
        release = 0.3
    
        if x < attack:
            return 1.0 / attack * x
        if x < attack + decay:
            return 1.0 - (x - attack) / decay * (1.0 - sustain)
        if x < 1.0 - release:
            return sustain
    
        return sustain / release * (1.0 - x)
    
    def mix(self, data, size, p):
        for i in range(size):
            if (p+i < self.time_start):
                continue
            if (p+i >= self.time_end):
                break
            trel = (p + i - self.time_start)
    
            data[i] += self.envelope(trel) * self.wave_triangle(float(trel) / samplerate * self.freq)
    
    def is_done(self, p):
        return p >= self.time_end

class SoundManager(object):
    def __init__(self, arr):
        self.s_pos = 0
        self.access_list = []
        self.oscillator_list = []
        self.lock = threading.RLock()
        self.arr = arr

        # PyAudio setup
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=samplerate,
            output=True,
            frames_per_buffer=1024,
            stream_callback=self.sound_callback
        )

    def add_oscillator(self, freq, p, pstart, duration):
        oldest = 0
        toldest = sys.maxsize
        for i in range(len(self.oscillator_list)):
            if (self.oscillator_list[i].is_done(p)):
                self.oscillator_list[i] = Oscillator(freq, pstart, duration)
                return
            if (self.oscillator_list[i].time_start < toldest):
                oldest = i
                toldest = self.oscillator_list[i].time_start
    
        if len(self.oscillator_list) < max_oscillators:
            self.oscillator_list.append(Oscillator(freq, pstart, duration))
        else:
            self.oscillator_list[oldest] = Oscillator(freq, pstart, duration)
    
    def sound_access(self, *args):
        if not sound_on:
            return
        else:
            for arg in args:
                self.access_list.append(arg)
    
    def sound_reset(self):
        self.s_pos = 0
        self.oscillator_list = []
    
    def sound_callback(self, in_data, frame_count, time_info, status):
        stream = [0] * frame_count
        if not sound_on:
            return (b'\x00' * frame_count, pyaudio.paContinue)

        with self.lock:
            if len(self.access_list) >= 1:
                pscale = float(len(stream)) / len(self.access_list)
                for i in range(len(self.access_list)):
                    freq = 120 + 1200 * (float(self.access_list[i]) / self.arr.size) * (float(self.access_list[i]) / self.arr.size)
                    duration = min(30 / 1000.0 , 10 / 1000.0 * sound_sustain) * samplerate
                    self.add_oscillator(freq, self.s_pos, self.s_pos + i * pscale, duration)
                self.access_list = []
    
        wave = [0.0] * len(stream)
        wavecount = 0
        for osc in self.oscillator_list:
            if not osc.is_done(self.s_pos):
                osc.mix(wave, len(wave), self.s_pos)
                wavecount += 1
        if wavecount == 0:
            for i in range(len(stream)):
                stream[i] = 0
        else:
            vol = max(wave)
            oldvol = 1.0
            if vol <= oldvol:
                vol = 0.9 * oldvol
    
            for i in range(len(stream)):
                v = int(24000 * wave[i] /(oldvol + (vol - oldvol) * (i/float(len(stream)))))
                if v > 32200: v = 32200
                if v < -32200: v = -32200
                stream[i] = v
            oldvol = vol
    
        self.s_pos += len(stream)
        return (np.array(stream, dtype=np.int16).tobytes(), pyaudio.paContinue)
