"""
Module 4: Transcription
Responsibility: Receives AudioSegments and runs faster-whisper on CUDA with
word_timestamps=True. Produces word tokens with relative start/end timestamps,
writes results into the packaging queue tagged by segment ID.
"""

import asyncio
import time
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from faster_whisper import WhisperModel
from segment_extractor import AudioSegment


@dataclass
class WordToken:
    """A single transcribed word with its timing within the segment."""
    word: str
    start_time: float       # Relative to segment start (seconds)
    end_time: float         # Relative to segment start (seconds)
    probability: float      # Whisper's confidence for this word token


@dataclass
class TranscriptionResult:
    """
    Output of the Transcription module for one segment.
    Keyed by segment_id so the Packaging module can match it with pitch data.
    """
    segment_id: str
    words: list[WordToken]
    segment_start_time: float   # Absolute wall clock time of segment start
    inference_duration: float   # How long transcription took (seconds)
    language: Optional[str] = None


class Transcriber:
    """
    Receives AudioSegment objects from the Segment Extractor,
    runs faster-whisper on CUDA with word-level timestamps,
    and pushes TranscriptionResult objects into the results queue.

    Uses a ThreadPoolExecutor to run blocking Whisper inference
    without blocking the asyncio event loop.

    Interface in:  AudioSegment (from SegmentExtractor via register_consumer)
    Interface out: asyncio.Queue[TranscriptionResult] (shared with Packaging module)
    """

    # "small" is the recommended balance of speed and accuracy within the 1-2s budget.
    # Switch to "medium" if accuracy is more important and GPU is powerful enough.
    MODEL_SIZE = "large-v3"
    COMPUTE_TYPE = "float16"    # float16 is fastest on CUDA for inference

    def __init__(self, results_queue: asyncio.Queue, device: str = "cuda"):
        self.results_queue = results_queue
        self.device = device if torch.cuda.is_available() else "cpu"
        self._model: Optional[WhisperModel] = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def load_model(self):
        """
        Load the Whisper model onto the configured device.
        Call this once before starting the pipeline.
        """
        compute = self.COMPUTE_TYPE if self.device == "cuda" else "int8"
        print(f"[Transcriber] Loading faster-whisper '{self.MODEL_SIZE}' "
              f"on {self.device} ({compute})...")
        self._model = WhisperModel(
            self.MODEL_SIZE,
            device=self.device,
            compute_type=compute
        )
        print(f"[Transcriber] Model ready.")

    def _run_inference(self, audio: np.ndarray) -> tuple[list[WordToken], Optional[str]]:
        """
        Blocking inference call — runs inside the ThreadPoolExecutor.
        Returns list of WordTokens and detected language.
        """
        segments, info = self._model.transcribe(
            audio,
            word_timestamps=True,
            beam_size=5,
            language="en",          # Auto-detect = NONE, choosing langage reduces latency
            vad_filter=False,       # VAD already handled upstream
            condition_on_previous_text=False  # Avoid hallucination carryover
        )

        words: list[WordToken] = []
        for seg in segments:
            if seg.words is None:
                continue
            for w in seg.words:
                token = WordToken(
                    word=w.word.strip(),
                    start_time=float(w.start),
                    end_time=float(w.end),
                    probability=float(w.probability)
                )
                if token.word:  # Skip empty tokens
                    words.append(token)

        return words, info.language

    async def process(self, segment: AudioSegment):
        """
        Async entry point called by SegmentExtractor for each segment.
        Dispatches inference to the thread pool so the event loop stays free.
        """
        if self._model is None:
            print(f"[Transcriber] Model not loaded, skipping {segment.segment_id}")
            return

        print(f"[Transcriber] Transcribing {segment.segment_id} "
              f"({segment.duration:.2f}s)...")
        t_start = time.perf_counter()

        loop = asyncio.get_event_loop()
        try:
            words, language = await loop.run_in_executor(
                self._executor,
                self._run_inference,
                segment.audio
            )
        except Exception as e:
            print(f"[Transcriber] Error on {segment.segment_id}: {e}")
            return

        inference_duration = time.perf_counter() - t_start

        result = TranscriptionResult(
            segment_id=segment.segment_id,
            words=words,
            segment_start_time=segment.start_time,
            inference_duration=inference_duration,
            language=language
        )

        await self.results_queue.put(result)

        word_strings = [w.word for w in words]
        print(f"[Transcriber] {segment.segment_id} -> "
              f"{len(words)} words in {inference_duration:.2f}s: "
              f"{' '.join(word_strings[:10])}{'...' if len(words) > 10 else ''}")

    def shutdown(self):
        """Clean up the thread pool executor."""
        self._executor.shutdown(wait=True)
        print("[Transcriber] Shutdown complete.")
