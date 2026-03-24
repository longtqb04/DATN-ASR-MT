import argparse
import os
import glob
import torch
import soundfile as sf
from tqdm import tqdm
from pyannote.audio import Pipeline

from whisper_model import WhisperASR

HF_TOKEN = os.environ.get("HF_TOKEN")
OUTPUT_FILE = "transcript.txt"

class AudioFile:
    def __init__(self, path, target_sr=16000):
        self.path = path

        waveform, sr = sf.read(path)

        # Convert to mono if needed
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Resample if needed
        if sr != target_sr:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        self.waveform = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        self.sr = sr


class Segment:
    def __init__(self, start, end, speaker):
        self.start = start
        self.end = end
        self.speaker = speaker

    def duration(self):
        return self.end - self.start

    def extract(self, audio: AudioFile):
        if self.duration() < 0.5:
            return None

        start_sample = int(self.start * audio.sr)
        end_sample = int(self.end * audio.sr)

        chunk = audio.waveform[:, start_sample:end_sample]

        if chunk.shape[1] == 0:
            return None

        return chunk.squeeze(0)


def get_audio_files(input_path):
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


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="base.en")
    parser.add_argument("--input_path", required=True)
    return parser.parse_args()

def main():
    args = parse_args()

    audio_files = get_audio_files(args.input_path)
    print(f"Found {len(audio_files)} audio files")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading Whisper ASR...")
    asr_model = WhisperASR(args.model_name, device=device)

    print("Loading diarization pipeline...")
    diarization_pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=HF_TOKEN
    )
    diarization_pipeline.to(torch.device(device))

    results = []

    print("\nRunning ASR pipeline...")

    for audio_path in tqdm(audio_files):

        utt_id = os.path.splitext(os.path.basename(audio_path))[0]

        audio = AudioFile(audio_path)

        diarization_output = diarization_pipeline({
            "waveform": audio.waveform,
            "sample_rate": audio.sr
        })

        segments = [
            Segment(turn.start, turn.end, speaker)
            for turn, _, speaker in diarization_output.speaker_diarization.itertracks(yield_label=True)
        ]

        transcript_parts = []

        for seg in segments:
            chunk = seg.extract(audio)
            if chunk is None:
                continue

            text = asr_model.transcribe(chunk)
            transcript_parts.append(f"{seg.speaker}: {text}")

        final_transcript = "\n".join(transcript_parts)

        results.append({
            "utt_id": utt_id,
            "en_hyp": final_transcript
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for r in results:
            f.write(f"Utterance: {r['utt_id']}\n")
            f.write(f"ASR Hyp (EN):\n")
            f.write(f"{r['en_hyp']}\n")
            f.write("-" * 50 + "\n")

    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
