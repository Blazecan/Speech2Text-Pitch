"""
Module 1: Audio Capture
Responsibility: Opens microphone stream and continuously writes raw PCM frames
into a shared circular audio buffer. Has no awareness of speech or pitch.
"""

import numpy as np
import sounddevice as sd
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class AudioBuffer:
    """
    Shared circular audio buffer written by AudioCapture,
    read by VAD and Segment Extractor.
    """
    sample_rate: int
    max_duration_seconds: float  # How much audio history to retain

    def __post_init__(self):
        self._max_frames = int(self.sample_rate * self.max_duration_seconds)
        self._buffer = deque(maxlen=self._max_frames)
        self._lock = threading.RLock()
        self._start_time: Optional[float] = None  # Wall clock time of first frame written

    def write(self, frames: np.ndarray, timestamp: float):
        """Write a chunk of PCM frames into the buffer."""
        with self._lock:
            if self._start_time is None:
                self._start_time = timestamp
            for sample in frames:
                self._buffer.append((timestamp + (i / self.sample_rate)
                                     for i, _ in enumerate([sample])))
            # Store as (sample_value, absolute_time) pairs
            # Rebuild with proper per-sample timestamps
            pass

    def _write_samples(self, frames: np.ndarray, chunk_start_time: float):
        """Internal: write samples with per-sample timestamps."""
        with self._lock:
            if self._start_time is None:
                self._start_time = chunk_start_time
            n = len(frames)
            for i, sample in enumerate(frames):
                t = chunk_start_time + (i / self.sample_rate)
                self._buffer.append((t, float(sample)))

    def slice(self, start_time: float, end_time: float) -> np.ndarray:
        """
        Return a numpy array of samples between start_time and end_time (wall clock).
        Used by the Segment Extractor.
        """
        with self._lock:
            samples = [s for (t, s) in self._buffer if start_time <= t <= end_time]
            if not samples:
                return np.array([], dtype=np.float32)
            return np.array(samples, dtype=np.float32)

    def get_recent(self, duration_seconds: float) -> tuple[np.ndarray, float]:
        """
        Return the most recent `duration_seconds` of audio and its start timestamp.
        Used by VAD for continuous monitoring.
        """
        with self._lock:
            if not self._buffer:
                return np.array([], dtype=np.float32), 0.0
            n_frames = int(duration_seconds * self.sample_rate)
            recent = list(self._buffer)[-n_frames:]
            start_t = recent[0][0] if recent else 0.0
            samples = np.array([s for (_, s) in recent], dtype=np.float32)
            return samples, start_t


class AudioCapture:
    """
    Continuously reads from the microphone and writes raw PCM float32 mono audio
    into a shared AudioBuffer at the configured sample rate.

    Interface out: AudioBuffer (shared object, written continuously)
    """

    SAMPLE_RATE = 16000       # Hz — native rate for Whisper and CREPE
    CHANNELS = 1              # Mono
    CHUNK_DURATION = 0.05     # Seconds per callback (50ms chunks)
    BUFFER_HISTORY = 30.0     # Seconds of audio history to retain in buffer

    def __init__(self, device: Optional[int] = None):
        self.device = device
        self.buffer = AudioBuffer(
            sample_rate=self.SAMPLE_RATE,
            max_duration_seconds=self.BUFFER_HISTORY
        )
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._chunk_samples = int(self.SAMPLE_RATE * self.CHUNK_DURATION)

    def _callback(self, indata: np.ndarray, frames: int,
                  time_info, status):
        """sounddevice callback — called on each audio chunk."""
        if status:
            print(f"[AudioCapture] Stream status: {status}")
        chunk_start = time.time() - (frames / self.SAMPLE_RATE)
        mono = indata[:, 0].copy()  # Take first channel, ensure copy
        self.buffer._write_samples(mono, chunk_start)

    def start(self):
        """Open the microphone stream and begin capturing."""
        self._running = True
        self._stream = sd.InputStream(
            samplerate=self.SAMPLE_RATE,
            channels=self.CHANNELS,
            dtype='float32',
            blocksize=self._chunk_samples,
            device=self.device,
            callback=self._callback
        )
        self._stream.start()
        print(f"[AudioCapture] Started. Sample rate: {self.SAMPLE_RATE}Hz, "
              f"chunk: {self.CHUNK_DURATION*1000:.0f}ms")

    def stop(self):
        """Stop capturing and close the stream."""
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        print("[AudioCapture] Stopped.")
