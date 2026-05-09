"""
Module 5: Pitch Detection
Responsibility: Receives AudioSegments and runs PENN (a fast CREPE-alternative)
on CUDA, producing a time series of (timestamp, Hz, confidence) pitch frames
covering the full segment. Writes results into the packaging queue tagged by
segment ID.
"""

import asyncio
import time
import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from segment_extractor import AudioSegment


@dataclass
class PitchFrame:
    """
    A single pitch estimate at a point in time, relative to segment start.
    """
    time: float             # Seconds relative to segment start
    frequency_hz: float     # Fundamental frequency in Hz (0.0 = unvoiced)
    confidence: float       # Model confidence [0.0, 1.0]
    voiced: bool            # True if this frame contains voiced speech


@dataclass
class PitchResult:
    """
    Output of the Pitch Detection module for one segment.
    Keyed by segment_id so the Packaging module can match it with transcription.
    """
    segment_id: str
    frames: list[PitchFrame]
    frame_period_s: float           # Time between frames (seconds)
    segment_start_time: float       # Absolute wall clock time
    inference_duration: float       # How long pitch inference took

    def slice(self, start_s: float, end_s: float) -> list[PitchFrame]:
        """
        Return pitch frames within [start_s, end_s] relative to segment start.
        Used by the Packaging module to extract per-word pitch data.
        """
        return [f for f in self.frames if start_s <= f.time <= end_s]

    def voiced_frames(self, start_s: float, end_s: float) -> list[PitchFrame]:
        """Return only voiced (non-silence) frames in the given window."""
        return [f for f in self.slice(start_s, end_s) if f.voiced]


class PitchDetector:
    """
    Receives AudioSegment objects from the Segment Extractor,
    runs PENN pitch estimation on CUDA,
    and pushes PitchResult objects into the results queue.

    PENN is a neural pitch estimator (successor to CREPE) that is significantly
    faster while maintaining accuracy. Falls back to torchcrepe if PENN is unavailable.

    Uses a ThreadPoolExecutor to run blocking inference without blocking the event loop.

    Interface in:  AudioSegment (from SegmentExtractor via register_consumer)
    Interface out: asyncio.Queue[PitchResult] (shared with Packaging module)
    """

    VOICED_CONFIDENCE_THRESHOLD = 0.5   # Below this = treat as unvoiced
    FMIN = 50.0     # Hz — minimum plausible human fundamental frequency
    FMAX = 550.0    # Hz — maximum plausible human fundamental frequency

    def __init__(self, results_queue: asyncio.Queue, device: str = "cuda"):
        self.results_queue = results_queue
        self.device = device if torch.cuda.is_available() else "cpu"
        self._backend = None        # Will be 'penn' or 'torchcrepe'
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pitch")

    def load_model(self):
        """
        Load the pitch detection model. Tries PENN first, falls back to torchcrepe.
        Call this once before starting the pipeline.
        """
        try:
            import penn
            self._penn = penn
            self._backend = 'penn'
            print(f"[PitchDetector] Using PENN on {self.device}.")
        except ImportError:
            try:
                import torchcrepe
                self._torchcrepe = torchcrepe
                self._backend = 'torchcrepe'
                print(f"[PitchDetector] PENN not available, using torchcrepe on {self.device}.")
            except ImportError:
                raise RuntimeError(
                    "[PitchDetector] Neither 'penn' nor 'torchcrepe' is installed. "
                    "Install one: pip install penn  OR  pip install torchcrepe"
                )

    def _run_penn(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Run PENN inference. Returns (frequencies_hz, confidences) arrays.
        PENN outputs one estimate per 10ms frame by default.
        """
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        # PENN expects audio tensor [batch, samples] and returns pitch + periodicity
        pitch, periodicity = self._penn.from_audio(
            audio_tensor,
            sample_rate=sample_rate,
            hopsize=0.01,       # 10ms frame period
            fmin=self.FMIN,
            fmax=self.FMAX,
            batch_size=512,
            center='half-hop',
            interp_unvoiced_at=None,  # Don't interpolate — we want raw confidence
            gpu=0 if self.device == 'cuda' else None
        )

        # pitch: [1, frames], periodicity: [1, frames]
        freqs = pitch.squeeze(0).cpu().numpy()
        confidences = periodicity.squeeze(0).cpu().numpy()
        return freqs, confidences

    def _run_torchcrepe(self, audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Run torchcrepe inference. Returns (frequencies_hz, confidences) arrays.
        """
        audio_tensor = torch.from_numpy(audio).unsqueeze(0).to(self.device)

        freqs, confidences = self._torchcrepe.predict(
            audio_tensor,
            sample_rate,
            hop_length=int(sample_rate * 0.01),  # 10ms frames
            fmin=self.FMIN,
            fmax=self.FMAX,
            model='full',
            return_periodicity=True,
            batch_size=512,
            device=self.device,
            decoder=self._torchcrepe.decode.weighted_argmax
        )

        freqs = freqs.squeeze(0).cpu().numpy()
        confidences = confidences.squeeze(0).cpu().numpy()
        return freqs, confidences

    def _run_inference(self, audio: np.ndarray,
                       sample_rate: int) -> tuple[list[PitchFrame], float]:
        """
        Blocking inference — runs inside ThreadPoolExecutor.
        Returns list of PitchFrames and the frame period in seconds.
        """
        if self._backend == 'penn':
            freqs, confidences = self._run_penn(audio, sample_rate)
        else:
            freqs, confidences = self._run_torchcrepe(audio, sample_rate)

        frame_period_s = 0.01  # 10ms — matches hopsize above
        n_frames = len(freqs)

        frames: list[PitchFrame] = []
        for i in range(n_frames):
            conf = float(confidences[i])
            freq = float(freqs[i])
            voiced = conf >= self.VOICED_CONFIDENCE_THRESHOLD

            frames.append(PitchFrame(
                time=i * frame_period_s,
                frequency_hz=freq if voiced else 0.0,
                confidence=conf,
                voiced=voiced
            ))

        return frames, frame_period_s

    async def process(self, segment: AudioSegment):
        """
        Async entry point called by SegmentExtractor for each segment.
        Runs inference in thread pool to avoid blocking the event loop.
        """
        if self._backend is None:
            print(f"[PitchDetector] Model not loaded, skipping {segment.segment_id}")
            return

        print(f"[PitchDetector] Detecting pitch for {segment.segment_id} "
              f"({segment.duration:.2f}s)...")
        t_start = time.perf_counter()

        loop = asyncio.get_event_loop()
        try:
            frames, frame_period = await loop.run_in_executor(
                self._executor,
                self._run_inference,
                segment.audio,
                segment.sample_rate
            )
        except Exception as e:
            print(f"[PitchDetector] Error on {segment.segment_id}: {e}")
            return

        inference_duration = time.perf_counter() - t_start

        result = PitchResult(
            segment_id=segment.segment_id,
            frames=frames,
            frame_period_s=frame_period,
            segment_start_time=segment.start_time,
            inference_duration=inference_duration
        )

        await self.results_queue.put(result)

        voiced_count = sum(1 for f in frames if f.voiced)
        print(f"[PitchDetector] {segment.segment_id} -> "
              f"{len(frames)} frames ({voiced_count} voiced) "
              f"in {inference_duration:.2f}s")

    def shutdown(self):
        """Clean up thread pool."""
        self._executor.shutdown(wait=True)
        print("[PitchDetector] Shutdown complete.")
