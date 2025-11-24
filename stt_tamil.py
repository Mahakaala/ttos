import argparse
import io
import os
import sys
from typing import Optional

from huggingface_hub import InferenceClient

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None  # Lazy guidance if missing


DEFAULT_MODEL_ID = os.environ.get("TAMIL_ASR_MODEL", "ai4bharat/IndicWhisper-v2")


def ensure_dependencies() -> None:
    if AudioSegment is None:
        raise RuntimeError(
            "Missing dependency 'pydub'. Please run: pip install -r requirements.txt"
        )


def load_hf_token(explicit_token: Optional[str]) -> str:
    token = explicit_token or os.environ.get("HUGGINGFACEHUB_API_TOKEN") or os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "Hugging Face API token not found. Set HUGGINGFACEHUB_API_TOKEN or pass --hf-token."
        )
    return token


def convert_to_wav_16k_mono(input_path: str) -> bytes:
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()


def transcribe(
    audio_path: str,
    model_id: str,
    hf_token: str,
    language: str = "ta",
    task: str = "transcribe",
) -> str:
    client = InferenceClient(token=hf_token)

    audio_wav_bytes = convert_to_wav_16k_mono(audio_path)

    # Many Whisper-like models accept language/task via params. InferenceClient handles routing.
    # For IndicWhisper, language code 'ta' is Tamil.
    result = client.audio_to_text(
        model=model_id,
        audio=audio_wav_bytes,
        # Extra parameters are passed to the model if supported
        # Whisper-compatible args
        # note: some models may ignore these if not applicable
        parameters={
            "language": language,
            "task": task,
        },
    )

    if isinstance(result, dict) and "text" in result:
        return result["text"]
    if isinstance(result, str):
        return result
    return str(result)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tamil Speech-to-Text using Hugging Face Inference API (AI4Bharat IndicWhisper by default)."
    )
    parser.add_argument("audio", help="Path to input audio file (wav/mp3/m4a/flac/etc.)")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL_ID,
        help="Hugging Face model id (default: %(default)s)",
    )
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Hugging Face API token. Falls back to env HUGGINGFACEHUB_API_TOKEN/HF_TOKEN.",
    )
    parser.add_argument(
        "--language",
        default="ta",
        help="Language code for transcription (Tamil=ta).",
    )
    parser.add_argument(
        "--task",
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task type for Whisper-like models.",
    )

    args = parser.parse_args()

    if not os.path.exists(args.audio):
        print(f"Input file not found: {args.audio}", file=sys.stderr)
        sys.exit(1)

    try:
        ensure_dependencies()
        token = load_hf_token(args.hf_token)
        text = transcribe(
            audio_path=args.audio,
            model_id=args.model,
            hf_token=token,
            language=args.language,
            task=args.task,
        )
        print(text)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()


