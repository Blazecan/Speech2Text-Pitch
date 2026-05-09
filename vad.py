"""
Module 2: Voice Activity Detection (VAD)
Responsibility: Reads the shared audio buffer continuously, detects speech/silence
boundaries using Silero-VAD on CUDA, and emits segment boundary events into an
async queue for the Segment Extractor to consume.
"""

import asyncio
import time
import threading
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from audio_capture import AudioBuffer


@dataclass
class SegmentEvent:
    """
    Emitted by VAD when a complete speech segment has been detected.
    Contains the wall-clock start and end times that the Segment Extractor
    will use to slice the AudioBuffer.
    """
    segment_id: str
    start_time: float   # Absolute wall clock time (time.time())
    end_time: float     # Absolute wall clock time (time.time())
    duration: float = field(init=False)

    def __post_init__(self):
        self.duration = self.end_time - self.start_time


class VoiceActivityDetector:
    """
    Continuously monitors the AudioBuffer using Silero-VAD running on CUDA.
    Emits SegmentEvent objects into an asyncio.Queue when a speech segment ends.

    Segment boundaries are determined by:
      - Speech start: VAD confidence rises above SPEECH_THRESHOLD
      - Speech end:   VAD confidence drops below SILENCE_THRESHOLD for at least
                      SILENCE_DURATION_S seconds
      - Hard cap:     Segments are force-closed at MAX_SEGMENT_DURATION_S regardless
                      of VAD state, to prevent backlog buildup

    Interface in:  AudioBuffer (shared, read continuously)
    Interface out: asyncio.Queue[SegmentEvent]
    """

    SPEECH_THRESHOLD = 0.5          # VAD confidence to declare speech started
    SILENCE_THRESHOLD = 0.35        # VAD confidence to declare silence
    SILENCE_DURATION_S = 0.4        # Seconds of silence before closing segment
    MAX_SEGMENT_DURATION_S = 8.0    # Hard cap to prevent runaway segments
    MIN_SEGMENT_DURATION_S = 0.2    # Ignore very short blips
    POLL_INTERVAL_S = 0.05          # How often VAD runs (50ms)
    VAD_WINDOW_S = 0.5              # Audio window fed to VAD each poll

    def __init__(self, audio_buffer: AudioBuffer, segment_queue: asyncio.Queue,
                 device: str = "cuda"):
        self.audio_buffer = audio_buffer
        self.segment_queue = segment_queue
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self._model = None
        self._utils = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Segment tracking state
        self._in_speech = False
        self._segment_start: Optional[float] = None
        self._silence_since: Optional[float] = None
        self._segment_counter = 0

    def _load_model(self):
        print(f"[VAD] Loading Silero-VAD on {self.device}...")
        from silero_vad import load_silero_vad
        self.model = load_silero_vad().to(self.device)
        print(f"[VAD] Model loaded on {self.device}.")

    SILERO_WINDOW = 512  # Required by Silero-VAD at 16kHz

    def _get_vad_confidence(self, audio_chunk: np.ndarray) -> float:
        if len(audio_chunk) < self.SILERO_WINDOW:
            return 0.0

        tensor = torch.from_numpy(audio_chunk).to(self.device)

        confidences = []
        for i in range(0, len(tensor) - self.SILERO_WINDOW + 1, self.SILERO_WINDOW):
            window = tensor[i:i + self.SILERO_WINDOW]
            with torch.no_grad():
                conf = self.model(window, self.audio_buffer.sample_rate).item()
            confidences.append(conf)

        return max(confidences) if confidences else 0.0

    def _emit_segment(self, start_time: float, end_time: float):
        """Package and enqueue a completed segment event."""
        duration = end_time - start_time
        if duration < self.MIN_SEGMENT_DURATION_S:
            return  # Discard noise blips

        self._segment_counter += 1
        segment_id = f"seg_{self._segment_counter:06d}"
        event = SegmentEvent(
            segment_id=segment_id,
            start_time=start_time,
            end_time=end_time
        )

        # Thread-safe enqueue into the asyncio event loop
        asyncio.run_coroutine_threadsafe(
            self.segment_queue.put(event),
            self._loop
        )
        print(f"[VAD] Emitted {segment_id}: "
              f"{start_time:.3f} -> {end_time:.3f} ({duration:.2f}s)")

    def _run_loop(self):
        """
        Main VAD polling loop. Runs on a dedicated thread.
        Polls the audio buffer every POLL_INTERVAL_S, runs VAD,
        and manages speech/silence state machine.
        """
        self._load_model()

        while self._running:
            now = time.time()

            # Fetch a short window of recent audio for VAD evaluation
            audio_window, _ = self.audio_buffer.get_recent(self.VAD_WINDOW_S)
            if len(audio_window) < int(self.audio_buffer.sample_rate * 0.1):
                time.sleep(self.POLL_INTERVAL_S)
                continue

            confidence = self._get_vad_confidence(audio_window)

            if not self._in_speech:
                # Waiting for speech to begin
                if confidence >= self.SPEECH_THRESHOLD:
                    self._in_speech = True
                    self._segment_start = now - self.VAD_WINDOW_S  # Rewind to window start
                    self._silence_since = None
                    print(f"[VAD] Speech detected at {self._segment_start:.3f}")

            else:
                # Currently in a speech segment
                segment_duration = now - self._segment_start

                # Hard cap: force close if segment is too long
                if segment_duration >= self.MAX_SEGMENT_DURATION_S:
                    print(f"[VAD] Hard cap reached ({segment_duration:.1f}s), force closing.")
                    self._emit_segment(self._segment_start, now)
                    self._in_speech = False
                    self._segment_start = None
                    self._silence_since = None

                elif confidence < self.SILENCE_THRESHOLD:
                    # Silence detected — start or continue silence timer
                    if self._silence_since is None:
                        self._silence_since = now
                    elif (now - self._silence_since) >= self.SILENCE_DURATION_S:
                        # Sustained silence — close the segment
                        end_time = self._silence_since  # End at silence onset
                        self._emit_segment(self._segment_start, end_time)
                        self._in_speech = False
                        self._segment_start = None
                        self._silence_since = None
                else:
                    # Speech continues — reset silence timer
                    self._silence_since = None

            time.sleep(self.POLL_INTERVAL_S)

    def start(self, loop: asyncio.AbstractEventLoop):
        """Start the VAD polling thread, bound to the given asyncio event loop."""
        self._loop = loop
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True,
                                        name="vad-thread")
        self._thread.start()
        print("[VAD] Started.")

    def stop(self):
        """Signal the VAD thread to stop and wait for it to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
        print("[VAD] Stopped.")
