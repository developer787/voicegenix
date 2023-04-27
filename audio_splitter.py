import os
import librosa
import numpy as np
from pyannote.core import Segment
from pydub import AudioSegment
from pyannote.pipeline import Pipeline
from pyannote.database import get_protocol, FileFinder

def split_audio_by_speakers(audio_file, output_folder='output'):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Define a custom function for speaker diarization
    class SpeakerDiarization(Pipeline):
        def __init__(self):
            super().__init__()

            self.embedding = get_protocol("pyannote/database/demo/embedding")
            self.similarity = self.embedding.get_loader("similarity")
        
        def apply(self, current_file):
            speaker_embedding = self.similarity(current_file)
            return speaker_embedding
        
    pipeline = SpeakerDiarization()
    scores = pipeline.apply({
        "audio": audio_file,
        "database": "custom",
        "annotated": Segment(0, 60), # Change to the actual duration of your audio file
    })

    audio = AudioSegment.from_wav(audio_file)
    duration_ms = len(audio)

    # Choose your own threshold for splitting audio based on speaker similarity scores
    threshold = 0.95 
    segments = []
    for idx, (start_time, end_time) in enumerate(scores.labels()):
        speaker_label = scores.Y[scores.labels()].get((start_time, end_time))[0]
        if np.mean(speaker_label) > threshold:
            segments.append((start_time, end_time))

    for idx, (start_time, end_time) in enumerate(segments):
        start_time_ms = int(float(start_time) * 1000)
        end_time_ms = int(float(end_time) * 1000)
        start_time_ms = max(start_time_ms, 0)
        end_time_ms = min(end_time_ms, duration_ms)

        speaker_audio = audio[start_time_ms:end_time_ms]
        output_path = os.path.join(output_folder, f"speaker_{idx}.wav")
        speaker_audio.export(output_path, format="wav")

    return output_folder