import argparse
import os
import glob
import librosa
import torch
from tqdm import tqdm
import subprocess
from pyannote.audio import Pipeline
import torchaudio
from whisper_model import WhisperASR
# from dataset import normalize_text
# from metrics import (
#    compute_basic_metrics,
#    compute_bert_score,
#    compute_semantic_error_rate
#)
#from lm import refine_transcript

HF_TOKEN = os.environ.get("HF_TOKEN")

OUTPUT_FILE = "transcript.txt"

def get_audio_files(input_path):
    """
    Accept single file OR directory
    """
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        extensions = ["*.wav", "*.mp3", "*.flac", "*.m4a"]
        files = []
        for ext in extensions:
            files.extend(glob.glob(
                os.path.join(input_path, "**", ext),
                recursive=True
            ))
        return files

    raise ValueError("Invalid input path")

def cut_audio(input_file, start, end, output_file):
    cmd = [
        "ffmpeg",
        "-y",
        "-i", input_file,
        "-ss", str(start),
        "-to", str(end),
        "-ar", "16000",
        "-ac", "1",
        output_file
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# Args

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="base.en")
    parser.add_argument("--input_path", required=True, help="Path to audio file or directory")
    parser.add_argument("--reference_file", default=None, help="Optional reference transcript file")
    return parser.parse_args()


# Main Pipeline

def main():
    args = parse_args()

    audio_files = get_audio_files(args.input_path)

    print(f"Found {len(audio_files)} audio files")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Whisper ASR...")
    asr_model = WhisperASR(args.model_name, device=device)

    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )

    diarization_pipeline.to(torch.device(device))

    results = []

    print("\nRunning ASR pipeline...")

    for audio_path in tqdm(audio_files):

        utt_id = os.path.splitext(os.path.basename(audio_path))[0]

        # ===== DIARIZATION =====
        waveform, sr = librosa.load(audio_path, sr=16000)
        waveform = torch.tensor(waveform).unsqueeze(0)
        output = diarization_pipeline({
            "waveform": waveform,
            "sample_rate": sr
        })

        diarization = output.speaker_diarization
        segments = []

        for turn, _, speaker in diarization.itertracks(yield_label=True):
            segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })

        transcript_parts = []

        # ===== ASR PER SEGMENT =====
        for seg in segments:

            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)

            audio_chunk = waveform[:, start_sample:end_sample]

            text = asr_model.transcribe(audio_chunk)

            transcript_parts.append(
                f"{seg['speaker']}: {text}"
            )

        final_transcript = "\n".join(transcript_parts)

        results.append({
            "utt_id": utt_id,
            "en_hyp": final_transcript
        })

    # Save results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Utterance: {r['utt_id']}\n")
            f.write(f"ASR Hyp (EN): {r['en_hyp']}\n")
            f.write("-" * 50 + "\n")

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
