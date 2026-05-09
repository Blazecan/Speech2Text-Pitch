# AUDIO PIPELINE — function reference
*real-time speech · pitch detection · ipc dispatch*

---

## audio_capture.py
`AudioCapture` · `AudioBuffer`

### AudioCapture.start(device: int | None) → None
Opens the microphone stream at 16kHz mono float32. Writes 50ms chunks continuously into the shared `AudioBuffer`.
> cfg: SAMPLE_RATE=16000 · CHUNK_DURATION=0.05s · BUFFER_HISTORY=30s

### AudioBuffer._write_samples(frames: ndarray, chunk_start_time: float) → None
Appends (absolute_time, sample) pairs into the deque. Thread-safe. Oldest samples evict automatically at max capacity.

### AudioBuffer.slice(start_time: float, end_time: float) → ndarray[float32]
Returns all samples between two absolute wall-clock timestamps. Called by SegmentExtractor after a VAD event.

### AudioBuffer.get_recent(duration_seconds: float) → tuple[ndarray, float]
Returns the most recent N seconds of audio and the start timestamp. Used by VAD for continuous speech monitoring.

---

## vad.py
`VoiceActivityDetector` · `SegmentEvent`

### VoiceActivityDetector.start(loop: AbstractEventLoop) → None
Spawns the VAD polling thread. Must be passed the running asyncio loop so segment events can be enqueued thread-safely.

### VAD._get_vad_confidence(audio_chunk: ndarray) → float [0, 1]
Slices audio into 512-sample windows (Silero requirement at 16kHz), runs the model on each, returns the max confidence across all windows.
> cfg: SILERO_WINDOW=512 · SPEECH_THRESHOLD=0.5 · SILENCE_THRESHOLD=0.35

### VAD._emit_segment(start_time: float, end_time: float) → None
Packages a `SegmentEvent` and thread-safely enqueues it into the asyncio segment queue. Discards segments shorter than MIN_SEGMENT_DURATION_S.
> out: asyncio.Queue[SegmentEvent] · cfg: MAX_SEGMENT=8s · MIN_SEGMENT=0.2s

---

## segment_extractor.py
`SegmentExtractor` · `AudioSegment`

### SegmentExtractor.register_consumer(consumer: async (AudioSegment) → None) → None
Registers a downstream coroutine to receive `AudioSegment` objects. Both Transcriber and PitchDetector register here before the pipeline starts.

### SegmentExtractor.run() → coroutine
Main async loop. Awaits segment events, slices the audio buffer, normalizes to [-1, 1], then fans the `AudioSegment` to all consumers via `asyncio.gather`.
> in: asyncio.Queue[SegmentEvent] · out: AudioSegment → all consumers

---

## transcriber.py
`Transcriber` · `TranscriptionResult` · `WordToken`

### Transcriber.load_model() → None
Loads faster-whisper onto CUDA with the configured model size and compute type. Must be called once before the pipeline starts.
> cfg: MODEL_SIZE="large-v3" · COMPUTE_TYPE="int8_float16"

### Transcriber.process(segment: AudioSegment) → coroutine
Async consumer registered with SegmentExtractor. Dispatches inference to a ThreadPoolExecutor, then enqueues a `TranscriptionResult` with per-word timestamps.
> out: asyncio.Queue[TranscriptionResult] · word_timestamps=True · beam_size=5

### Transcriber._run_inference(audio: ndarray) → list[WordToken], str
Blocking whisper call — runs inside the thread pool. Returns word tokens (word, start, end, probability) and the detected language string.

---

## pitch_detector.py
`PitchDetector` · `PitchResult` · `PitchFrame`

### PitchDetector.load_model() → None
Tries to import PENN, falls back to torchcrepe. Sets `self._backend` accordingly. Call once before the pipeline starts.

### PitchDetector.process(segment: AudioSegment) → coroutine
Async consumer registered with SegmentExtractor. Runs pitch inference in a thread pool and enqueues a `PitchResult` with one `PitchFrame` per 10ms.
> out: asyncio.Queue[PitchResult] · cfg: FMIN=50Hz · FMAX=550Hz · VOICED_THRESHOLD=0.5

### PitchResult.slice(start_s: float, end_s: float) → list[PitchFrame]
Filters pitch frames to a time window relative to segment start. Called by the Packager for each word to extract its pitch contour.

---

## packager.py
`Packager` · `WordPackage` · transports

### Packager.run() → coroutine
Polls both result queues every 20ms. When transcription and pitch results share a segment_id, calls `_build_packages` and dispatches each word in order.
> in: Queue[TranscriptionResult] + Queue[PitchResult] · out: WordPackage → transport

### Packager._build_packages(transcription, pitch) → list[WordPackage]
Iterates words, slices pitch frames per word by timestamp, computes summary stats (mean, min, max, voiced_ratio), converts relative times to absolute wall-clock.

### WordPackage.to_json() → str
Serializes the package to compact JSON. Fields: word · segment_id · absolute_start · absolute_end · word_probability · pitch_frames · pitch_summary.

### Packager._prune_pending() → None
Evicts unmatched results older than MAX_PENDING_AGE_S to prevent memory leaks if one module drops a segment.
> cfg: MAX_PENDING_AGE_S=30

---

## transports — packager.py
`StdoutTransport` · `TCPSocketTransport` · `UnixSocketTransport` · `FileTransport`

### StdoutTransport.send(package: WordPackage) → coroutine
Writes one JSON line to stdout, flushed immediately. Use with shell pipes.
> --transport stdout

### TCPSocketTransport.send(package: WordPackage) → coroutine
Broadcasts newline-delimited JSON to all connected TCP clients. Drops silently if no client is connected.
> --transport tcp --host 127.0.0.1 --port 9876

### FileTransport.send(package: WordPackage) → coroutine
Appends one JSON line to a log file opened in line-buffered mode — each word hits disk immediately without waiting for an OS buffer flush.
> --transport file --log-path pipeline_output.log

---

## pitch_visualizer.py
standalone script

### load_log(path: str, min_confidence: float) → list[dict]
Parses the log file line-by-line, filters pitch frames below min_confidence, sorts by absolute_start, then zero-shifts all timestamps to start at 0.

### build_pitch_series(words: list[dict]) → tuple[ndarray, ndarray]
Remaps each word's pitch frame times from segment-relative to word-absolute coordinates. Inserts NaN breaks between words so the line doesn't span silence.

### plot(words: list[dict], save_path: str | None) → None
Renders the pitch graph: log-scale Y axis, continuous pitch line with glow, word boundary boxes with labels. Shows interactively or saves to file.
> --save pitch.png · --min-confidence 0.45
