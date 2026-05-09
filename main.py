"""
Main Pipeline Entrypoint
Wires all six modules together and manages startup/shutdown.

Usage:
    python main.py                          # stdout transport (default)
    python main.py --transport file --log-path file_name.log        # file transport 
    python main.py --transport tcp          # TCP socket on 127.0.0.1:9876
    python main.py --transport unix         # Unix socket at /tmp/audio_pipeline.sock
    python main.py --transport tcp --port 9999
    python main.py --device 0              # Use specific audio input device index
    python main.py --list-devices          # List available audio input devices
"""

import asyncio
import argparse
import signal
import sys
import sounddevice as sd

from audio_capture import AudioCapture
from vad import VoiceActivityDetector
from segment_extractor import SegmentExtractor
from transcriber import Transcriber
from pitch_detector import PitchDetector
from packager import (
    Packager,
    FileTransport,
    StdoutTransport,
    TCPSocketTransport,
    UnixSocketTransport,
    IPCTransport
)


def list_devices():
    """Print available audio input devices and exit."""
    print("Available audio input devices:")
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        if dev['max_input_channels'] > 0:
            print(f"  [{i}] {dev['name']} "
                  f"(channels: {dev['max_input_channels']}, "
                  f"rate: {int(dev['default_samplerate'])}Hz)")
    sys.exit(0)


async def run_pipeline(args):
    """
    Build and run the full pipeline.
    Module wiring order:
      AudioCapture -> AudioBuffer
      VAD -> SegmentEvent queue
      SegmentExtractor -> fans to Transcriber + PitchDetector
      Transcriber + PitchDetector -> results queues
      Packager -> IPC transport
    """

    # ------------------------------------------------------------------
    # 1. Shared queues (the glue between modules)
    # ------------------------------------------------------------------
    segment_queue = asyncio.Queue(maxsize=20)
    transcription_queue = asyncio.Queue(maxsize=50)
    pitch_queue = asyncio.Queue(maxsize=50)

    # ------------------------------------------------------------------
    # 2. Instantiate modules
    # ------------------------------------------------------------------
    capture = AudioCapture(device=args.device)

    vad = VoiceActivityDetector(
        audio_buffer=capture.buffer,
        segment_queue=segment_queue,
        device=args.cuda_device
    )

    extractor = SegmentExtractor(
        segment_queue=segment_queue,
        audio_buffer=capture.buffer
    )

    transcriber = Transcriber(
        results_queue=transcription_queue,
        device=args.cuda_device
    )

    pitch_detector = PitchDetector(
        results_queue=pitch_queue,
        device=args.cuda_device
    )

    # ------------------------------------------------------------------
    # 3. Configure IPC transport
    # ------------------------------------------------------------------
    transport: IPCTransport
    if args.transport == "file":
        transport = FileTransport(path=args.log_path)
        await transport.open()
    elif args.transport == "tcp":
        transport = TCPSocketTransport(host=args.host, port=args.port)
        await transport.start_server()
    elif args.transport == "unix":
        transport = UnixSocketTransport(socket_path=args.socket_path)
        await transport.start_server()
    else:
        transport = StdoutTransport()

    packager = Packager(
        transcription_queue=transcription_queue,
        pitch_queue=pitch_queue,
        transport=transport
    )

    # ------------------------------------------------------------------
    # 4. Wire consumers: SegmentExtractor fans to Transcriber + PitchDetector
    # ------------------------------------------------------------------
    extractor.register_consumer(transcriber.process)
    extractor.register_consumer(pitch_detector.process)

    # ------------------------------------------------------------------
    # 5. Load ML models (blocking, done before stream starts)
    # ------------------------------------------------------------------
    print("Loading models...")
    transcriber.load_model()
    pitch_detector.load_model()
    print("All models loaded. Starting pipeline...\n")

    # ------------------------------------------------------------------
    # 6. Start audio capture and VAD (threaded)
    # ------------------------------------------------------------------
    loop = asyncio.get_event_loop()
    capture.start()
    vad.start(loop)

    # ------------------------------------------------------------------
    # 7. Graceful shutdown handler
    # ------------------------------------------------------------------
    shutdown_event = asyncio.Event()

    def handle_shutdown(sig, frame):
        print(f"\n[Main] Received signal {sig}, shutting down...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # ------------------------------------------------------------------
    # 8. Run async tasks until shutdown
    # ------------------------------------------------------------------
    extractor_task = asyncio.create_task(extractor.run())
    packager_task = asyncio.create_task(packager.run())

    print("[Main] Pipeline running. Speak into the microphone.")
    print("[Main] Press Ctrl+C to stop.\n")

    await shutdown_event.wait()

    # ------------------------------------------------------------------
    # 9. Orderly shutdown
    # ------------------------------------------------------------------
    print("[Main] Stopping modules...")

    vad.stop()
    capture.stop()
    extractor.stop()
    packager.stop()

    extractor_task.cancel()
    packager_task.cancel()

    await asyncio.gather(extractor_task, packager_task, return_exceptions=True)

    transcriber.shutdown()
    pitch_detector.shutdown()
    await transport.close()

    print("[Main] Pipeline stopped cleanly.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Real-time speech-to-text + pitch detection pipeline."
    )
    parser.add_argument(
        "--transport", choices=["stdout", "tcp", "unix", "file"], default="stdout",
        help="IPC transport for word packages (default: stdout)"
    )
    parser.add_argument(
        "--log-path", default="pipeline_output.log",
        help="Log file path for using file transport (default: pipeline_output.log)"
    )
    parser.add_argument(
        "--host", default="127.0.0.1",
        help="TCP host to bind (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--port", type=int, default=9876,
        help="TCP port to bind (default: 9876)"
    )
    parser.add_argument(
        "--socket-path", default="/tmp/audio_pipeline.sock",
        help="Unix socket path (default: /tmp/audio_pipeline.sock)"
    )
    parser.add_argument(
        "--device", type=int, default=None,
        help="Audio input device index (default: system default)"
    )
    parser.add_argument(
        "--cuda-device", default="cuda",
        help="PyTorch device string for CUDA (default: 'cuda')"
    )
    parser.add_argument(
        "--list-devices", action="store_true",
        help="List available audio input devices and exit"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.list_devices:
        list_devices()

    asyncio.run(run_pipeline(args))
