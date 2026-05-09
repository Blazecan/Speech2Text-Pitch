"""
Module 3: Segment Audio Extractor
Responsibility: Listens for SegmentEvents from the VAD queue, slices the raw audio
buffer between the segment's timestamps, and fans the resulting audio array out to
both the Transcription and Pitch Detection modules simultaneously.
Does no audio processing itself — purely a router and buffer slicer.
"""

import asyncio
import numpy as np
from dataclasses import dataclass
from typing import Callable, Awaitable
from audio_capture import AudioBuffer
from vad import SegmentEvent


@dataclass
class AudioSegment:
    """
    A concrete slice of audio with its metadata.
    Dispatched simultaneously to Transcription and Pitch Detection.
    """
    segment_id: str
    audio: np.ndarray       # float32 mono PCM at 16kHz
    start_time: float       # Absolute wall clock time of first sample
    end_time: float         # Absolute wall clock time of last sample
    sample_rate: int        # Always 16000

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def num_samples(self) -> int:
        return len(self.audio)


# Type alias for downstream consumer coroutines
SegmentConsumer = Callable[[AudioSegment], Awaitable[None]]


class SegmentExtractor:
    """
    Consumes SegmentEvents from the VAD queue.
    For each event, slices the AudioBuffer to produce an AudioSegment,
    then dispatches it concurrently to all registered consumers
    (Transcription and Pitch Detection).

    Interface in:  asyncio.Queue[SegmentEvent]  (from VAD)
                   AudioBuffer                  (from AudioCapture)
    Interface out: AudioSegment dispatched to registered SegmentConsumer coroutines
    """

    MIN_SAMPLES = 320   # At 16kHz = 20ms. Discard anything shorter.

    def __init__(self, segment_queue: asyncio.Queue, audio_buffer: AudioBuffer):
        self.segment_queue = segment_queue
        self.audio_buffer = audio_buffer
        self._consumers: list[SegmentConsumer] = []
        self._running = False

    def register_consumer(self, consumer: SegmentConsumer):
        """
        Register a downstream coroutine to receive AudioSegments.
        Both Transcription and Pitch Detection register here.
        """
        self._consumers.append(consumer)
        print(f"[SegmentExtractor] Registered consumer: {consumer.__qualname__}")

    async def _dispatch(self, segment: AudioSegment):
        """
        Dispatch an AudioSegment to all consumers concurrently.
        All consumers receive the same segment object simultaneously.
        """
        if not self._consumers:
            print("[SegmentExtractor] Warning: no consumers registered.")
            return

        await asyncio.gather(
            *[consumer(segment) for consumer in self._consumers],
            return_exceptions=True
        )

    async def run(self):
        """
        Main async loop. Waits for SegmentEvents, slices audio, dispatches.
        Runs until stop() is called.
        """
        self._running = True
        print("[SegmentExtractor] Running.")

        while self._running:
            try:
                event: SegmentEvent = await asyncio.wait_for(
                    self.segment_queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue  # Loop back and check _running

            audio = self._slice_audio(event)

            if audio is None:
                print(f"[SegmentExtractor] Discarded {event.segment_id}: "
                      f"insufficient audio in buffer.")
                continue

            segment = AudioSegment(
                segment_id=event.segment_id,
                audio=audio,
                start_time=event.start_time,
                end_time=event.end_time,
                sample_rate=self.audio_buffer.sample_rate
            )

            print(f"[SegmentExtractor] Dispatching {segment.segment_id}: "
                  f"{segment.num_samples} samples ({segment.duration:.2f}s) "
                  f"to {len(self._consumers)} consumer(s).")

            # Dispatch to all consumers (transcription + pitch) concurrently
            asyncio.create_task(self._dispatch(segment))

    def _slice_audio(self, event: SegmentEvent) -> np.ndarray | None:
        """
        Slice the AudioBuffer between the event's timestamps.
        Returns None if the slice is too short to be useful.
        """
        audio = self.audio_buffer.slice(event.start_time, event.end_time)

        if len(audio) < self.MIN_SAMPLES:
            return None

        # Normalize to [-1, 1] float32 — both Whisper and CREPE expect this
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        return audio.astype(np.float32)

    def stop(self):
        """Signal the run loop to exit."""
        self._running = False
        print("[SegmentExtractor] Stopped.")
