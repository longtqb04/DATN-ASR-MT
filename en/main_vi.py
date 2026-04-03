import argparse
import os
import json
from typing import List

import torch
import soundfile as sf
from tqdm import tqdm
from pyannote.audio import Pipeline as DiarizationPipeline
import pandas as pd


# Config
HF_TOKEN = os.environ.get("HF_TOKEN")
torch.set_num_threads(4)


# Utils
def load_asr(model_name, device):
    from transformers import pipeline
    return pipeline(
        "automatic-speech-recognition",
        model=model_name,
        device=device
    )

def get_audio_files(input_path: str) -> List[str]:
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        exts = (".wav", ".mp3", ".flac", ".m4a")
        files = []
        for root, _, filenames in os.walk(input_path):
            for f in filenames:
                if f.lower().endswith(exts):
                    files.append(os.path.join(root, f))
        return files

    raise ValueError("Invalid input path")


def diarization_to_df(diarization) -> pd.DataFrame:
    rows = []
    for turn, _, speaker in diarization.speaker_diarization.itertracks(yield_label=True):
        rows.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    return pd.DataFrame(rows)


def load_audio(audio_path: str, target_sr: int = 16000):
    """
    Load audio without whisperx/torchcodec to avoid libtorchcodec runtime issues.
    Returns:
      - array_audio: 1D float32 numpy array
      - waveform: (1, T) float32 torch tensor
      - sample_rate: int
    """
    waveform, sr = sf.read(audio_path)

    # Convert to mono
    if len(waveform.shape) > 1:
        waveform = waveform.mean(axis=1)

    # Resample when sample rate differs
    if sr != target_sr:
        import librosa
        waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    waveform = waveform.astype("float32")
    tensor_waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
    return waveform, tensor_waveform, sr


class PhoWhisperPipeline:
    def __init__(self, model_name: str):
        self.device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'cuda' if self.device == 0 else 'cpu'}")

        # Load PhoWhisper ASR
        print(f"Loading PhoWhisper model {model_name}...")
        self.asr = load_asr(model_name, self.device)

        # Load diarization (CPU by default)
        print("Loading diarization pipeline...")
        self.diarization = DiarizationPipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=HF_TOKEN
        )
        self.diarization.to(torch.device("cpu"))

    def process_file(self, audio_path: str):
        utt_id = os.path.splitext(os.path.basename(audio_path))[0]

        audio, waveform, sr = load_audio(audio_path)

        result = self.asr({
            "array": audio,
            "sampling_rate": sr
        })

        text = result["text"]

        diarization_result = self.diarization({
            "waveform": waveform,
            "sample_rate": sr
        })

        diarize_df = diarization_to_df(diarization_result)

        # Convert diarization to segments (no alignment available)
        segments = []
        for _, row in diarize_df.iterrows():
            segments.append({
                "start": row["start"],
                "end": row["end"],
                "speaker": row["speaker"],
                "text": text  # naive assignment
            })

        return {
            "utt_id": utt_id,
            "segments": segments
        }


# CLI
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="vinai/PhoWhisper-medium")
    parser.add_argument("--input_path", required=True)
    parser.add_argument("--output", default="transcript_vi.json")
    return parser.parse_args()


def main():
    args = parse_args()

    files = get_audio_files(args.input_path)
    print(f"Found {len(files)} audio files")

    pipeline = PhoWhisperPipeline(args.model_name)

    results = []

    for audio_path in tqdm(files):
        try:
            result = pipeline.process_file(audio_path)
            results.append(result)
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
