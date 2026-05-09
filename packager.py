"""
Module 6: Packaging and Dispatch
Responsibility: Waits for both TranscriptionResult and PitchResult for the same
segment ID, pairs pitch frames to each word by timestamp, and emits self-contained
WordPackage objects over the configured IPC transport one word at a time in order.
This is the only module that communicates with the outside world.
"""

import asyncio
import json
import socket
import time
from dataclasses import dataclass, asdict
from typing import Optional
from transcriber import TranscriptionResult, WordToken
from pitch_detector import PitchResult, PitchFrame


# ---------------------------------------------------------------------------
# Output data structures
# ---------------------------------------------------------------------------

@dataclass
class PitchSummary:
    """Descriptive statistics over the pitch frames of a word."""
    mean_hz: float
    min_hz: float
    max_hz: float
    range_hz: float
    voiced_ratio: float     # Fraction of frames that are voiced


@dataclass
class WordPackage:
    """
    The final output unit: one word with all associated pitch data.
    This is what gets serialized and sent to the receiving application.
    """
    word: str
    segment_id: str

    # Absolute wall clock timestamps
    absolute_start: float
    absolute_end: float

    # Word confidence from Whisper
    word_probability: float

    # Full pitch contour for this word
    pitch_frames: list[dict]        # [{"time": float, "hz": float, "confidence": float, "voiced": bool}]

    # Descriptive statistics over the pitch frames
    pitch_summary: Optional[dict]   # None if no voiced frames exist

    def to_json(self) -> str:
        """Serialize to JSON string for IPC dispatch."""
        return json.dumps({
            "word": self.word,
            "segment_id": self.segment_id,
            "absolute_start": self.absolute_start,
            "absolute_end": self.absolute_end,
            "word_probability": self.word_probability,
            "pitch_frames": self.pitch_frames,
            "pitch_summary": self.pitch_summary
        }, separators=(',', ':'))


# ---------------------------------------------------------------------------
# IPC Transports
# ---------------------------------------------------------------------------


class IPCTransport:
    """Base class for IPC dispatch transports."""
    async def send(self, package: WordPackage): ...
    async def close(self): ...

class FileTransport(IPCTransport):
    """Writes newline-delimited JSON to a log file."""
    def __init__(self, path: str = "pipeline_output.log"):
        self.path = path
        self._file = None

    async def open(self):
        self._file = open(self.path, "a", encoding="utf-8", buffering=1)  # line-buffered
        print(f"[Dispatch] Logging to {self.path}")

    async def send(self, package: WordPackage):
        if self._file:
            self._file.write(package.to_json() + "\n")

    async def close(self):
        if self._file:
            self._file.close()
            

class StdoutTransport(IPCTransport):
    """Writes newline-delimited JSON to stdout. Simple, pipe-friendly."""
    async def send(self, package: WordPackage):
        print(package.to_json(), flush=True)

    async def close(self):
        pass


class TCPSocketTransport(IPCTransport):
    """
    Sends newline-delimited JSON over a TCP socket.
    The receiving application connects as a client to host:port.
    """
    def __init__(self, host: str = "127.0.0.1", port: int = 9876):
        self.host = host
        self.port = port
        self._server = None
        self._writers: list[asyncio.StreamWriter] = []

    async def start_server(self):
        """Start listening for client connections."""
        self._server = await asyncio.start_server(
            self._handle_client, self.host, self.port
        )
        print(f"[Dispatch] TCP server listening on {self.host}:{self.port}")

    async def _handle_client(self, reader: asyncio.StreamReader,
                              writer: asyncio.StreamWriter):
        addr = writer.get_extra_info('peername')
        print(f"[Dispatch] Client connected: {addr}")
        self._writers.append(writer)
        try:
            await reader.read()  # Block until client disconnects
        except Exception:
            pass
        finally:
            self._writers.remove(writer)
            writer.close()
            print(f"[Dispatch] Client disconnected: {addr}")

    async def send(self, package: WordPackage):
        if not self._writers:
            return  # No clients connected, drop silently
        data = (package.to_json() + "\n").encode("utf-8")
        for writer in list(self._writers):
            try:
                writer.write(data)
                await writer.drain()
            except Exception as e:
                print(f"[Dispatch] Send error: {e}")

    async def close(self):
        for writer in self._writers:
            writer.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()


class UnixSocketTransport(IPCTransport):
    """
    Sends newline-delimited JSON over a Unix domain socket.
    Lower overhead than TCP for same-machine IPC.
    """
    def __init__(self, socket_path: str = "/tmp/audio_pipeline.sock"):
        self.socket_path = socket_path
        self._server = None
        self._writers: list[asyncio.StreamWriter] = []

    async def start_server(self):
        import os
        if os.path.exists(self.socket_path):
            os.unlink(self.socket_path)
        self._server = await asyncio.start_unix_server(
            self._handle_client, path=self.socket_path
        )
        print(f"[Dispatch] Unix socket server at {self.socket_path}")

    async def _handle_client(self, reader, writer):
        self._writers.append(writer)
        try:
            await reader.read()
        except Exception:
            pass
        finally:
            self._writers.remove(writer)
            writer.close()

    async def send(self, package: WordPackage):
        data = (package.to_json() + "\n").encode("utf-8")
        for writer in list(self._writers):
            try:
                writer.write(data)
                await writer.drain()
            except Exception as e:
                print(f"[Dispatch] Send error: {e}")

    async def close(self):
        for writer in self._writers:
            writer.close()
        if self._server:
            self._server.close()
            await self._server.wait_closed()


# ---------------------------------------------------------------------------
# Packager
# ---------------------------------------------------------------------------

class Packager:
    """
    Watches two input queues (transcription results and pitch results).
    When both results for the same segment_id are available, pairs them:
      - Iterates over words in order
      - Slices pitch frames for each word's time window
      - Computes pitch summary statistics
      - Emits WordPackage objects via the configured transport, one per word

    Pending results are held in memory until their counterpart arrives.
    Old pending results are pruned after MAX_PENDING_AGE_S to prevent leaks.

    Interface in:  asyncio.Queue[TranscriptionResult]  (from Transcriber)
                   asyncio.Queue[PitchResult]           (from PitchDetector)
    Interface out: WordPackage objects via IPCTransport
    """

    MAX_PENDING_AGE_S = 30.0    # Discard unmatched results older than this

    def __init__(self,
                 transcription_queue: asyncio.Queue,
                 pitch_queue: asyncio.Queue,
                 transport: IPCTransport):
        self.transcription_queue = transcription_queue
        self.pitch_queue = pitch_queue
        self.transport = transport

        # Holding areas for results awaiting their counterpart
        self._pending_transcriptions: dict[str, tuple[float, TranscriptionResult]] = {}
        self._pending_pitch: dict[str, tuple[float, PitchResult]] = {}

        self._running = False
        self._words_dispatched = 0

    def _compute_pitch_summary(self, frames: list[PitchFrame]) -> Optional[dict]:
        """Compute descriptive statistics over a list of pitch frames."""
        voiced = [f for f in frames if f.voiced]
        if not voiced:
            return None

        freqs = [f.frequency_hz for f in voiced]
        return {
            "mean_hz": sum(freqs) / len(freqs),
            "min_hz": min(freqs),
            "max_hz": max(freqs),
            "range_hz": max(freqs) - min(freqs),
            "voiced_ratio": len(voiced) / max(len(frames), 1)
        }

    def _build_packages(self,
                         transcription: TranscriptionResult,
                         pitch: PitchResult) -> list[WordPackage]:
        """
        Pair words with their pitch frames and build the final WordPackage list.
        """
        packages = []

        for word_token in transcription.words:
            # Slice pitch frames that fall within this word's time window
            pitch_frames = pitch.slice(word_token.start_time, word_token.end_time)

            # Serialize pitch frames for the output
            frames_serialized = [
                {
                    "time": f.time,
                    "hz": f.frequency_hz,
                    "confidence": f.confidence,
                    "voiced": f.voiced
                }
                for f in pitch_frames
            ]

            summary = self._compute_pitch_summary(pitch_frames)

            # Convert relative word timestamps to absolute wall clock times
            abs_start = transcription.segment_start_time + word_token.start_time
            abs_end = transcription.segment_start_time + word_token.end_time

            package = WordPackage(
                word=word_token.word,
                segment_id=transcription.segment_id,
                absolute_start=abs_start,
                absolute_end=abs_end,
                word_probability=word_token.probability,
                pitch_frames=frames_serialized,
                pitch_summary=summary
            )
            packages.append(package)

        return packages

    def _prune_pending(self):
        """Remove stale pending results that never found a match."""
        now = time.time()
        stale_t = [sid for sid, (t, _) in self._pending_transcriptions.items()
                   if now - t > self.MAX_PENDING_AGE_S]
        stale_p = [sid for sid, (t, _) in self._pending_pitch.items()
                   if now - t > self.MAX_PENDING_AGE_S]
        for sid in stale_t:
            del self._pending_transcriptions[sid]
            print(f"[Packager] Pruned stale transcription: {sid}")
        for sid in stale_p:
            del self._pending_pitch[sid]
            print(f"[Packager] Pruned stale pitch: {sid}")

    async def _drain_transcription_queue(self):
        """Drain all available transcription results into pending dict."""
        while True:
            try:
                result: TranscriptionResult = self.transcription_queue.get_nowait()
                self._pending_transcriptions[result.segment_id] = (time.time(), result)
            except asyncio.QueueEmpty:
                break

    async def _drain_pitch_queue(self):
        """Drain all available pitch results into pending dict."""
        while True:
            try:
                result: PitchResult = self.pitch_queue.get_nowait()
                self._pending_pitch[result.segment_id] = (time.time(), result)
            except asyncio.QueueEmpty:
                break

    async def _match_and_dispatch(self):
        """
        Find segment IDs that have both transcription and pitch results,
        build packages, and dispatch them.
        """
        matched_ids = set(self._pending_transcriptions) & set(self._pending_pitch)

        for segment_id in matched_ids:
            _, transcription = self._pending_transcriptions.pop(segment_id)
            _, pitch = self._pending_pitch.pop(segment_id)

            packages = self._build_packages(transcription, pitch)

            for package in packages:
                await self.transport.send(package)
                self._words_dispatched += 1

            print(f"[Packager] Dispatched {len(packages)} word(s) "
                  f"from {segment_id}. Total dispatched: {self._words_dispatched}")

    async def run(self):
        """
        Main async loop. Polls both result queues, matches by segment ID,
        and dispatches word packages. Runs until stop() is called.
        """
        self._running = True
        print("[Packager] Running.")

        while self._running:
            # Drain both queues
            await self._drain_transcription_queue()
            await self._drain_pitch_queue()

            # Match and dispatch completed pairs
            await self._match_and_dispatch()

            # Periodically prune stale pending results
            self._prune_pending()

            # Yield control briefly before next poll
            await asyncio.sleep(0.02)

    def stop(self):
        self._running = False
        print(f"[Packager] Stopped. Total words dispatched: {self._words_dispatched}")
