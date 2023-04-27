import os
import librosa
import numpy as np
from pyannote.core import Segment
from pydub import AudioSegment
from pyannote.pipeline import Pipeline

# Speaker diarization pipeline
SPEAKER_DIARIZATION_MODEL = "dia_dihard"

# Load the pre-trained model
pretrained = Pipeline.from_pretrained(SPEAKER_DIARIZATION_MODEL)

def split_audio_by_speakers(audio_file, output_folder='output'):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Apply the pretrained model
    hypothesis = pretrained({"mono": audio_file})

    segments = []
    for segment, _, speaker in hypothesis.itertracks(yield_label=True):
        segments.append((Segment(segment.start, segment.end), speaker))

    audio = AudioSegment.from_wav(audio_file)
    duration_ms = len(audio)

    for idx, (segment, speaker_label) in enumerate(segments):
        start_time = int(segment.start * 1000)
        end_time = int(segment.end * 1000)
        start_time = max(start_time, 0)
        end_time = min(end_time, duration_ms)

        speaker_audio = audio[start_time:end_time]

        # Create a subdirectory for each speaker
        speaker_folder = os.path.join(output_folder, f"speaker_{speaker_label}")
        os.makedirs(speaker_folder, exist_ok=True)

        # Save current speaker's audio segment in their subdirectory
        output_path = os.path.join(speaker_folder, f"speaker_{speaker_label}_{idx}.wav")
        speaker_audio.export(output_path, format="wav")

    return output_folder