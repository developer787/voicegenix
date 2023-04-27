import os
import librosa
import numpy as np
from pyannote.core import Segment
from pydub import AudioSegment
from pyannote.audio.features import Precomputed
from pyannote.audio.embedding.extraction import SequenceEmbedding
from pyannote.audio.tasks.speaker_diarization import SpeakerDiarization


def split_audio_by_speakers(audio_file, output_folder='output'):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Load the audio
    audio, sr = librosa.load(audio_file, sr=None, mono=True)

    # Preprocess
    # Apply preprocessing steps like denoising, normalization, etc.

    # Voice activity detection and embedding
    embedding = SequenceEmbedding()

    # Train or use pre-trained SpeakerDiarization pipeline
    pipeline = SpeakerDiarization(embedding)
    hypothesis = pipeline({"audio": audio_file})

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
        output_path = os.path.join(output_folder, f"speaker_{speaker_label}_{idx}.wav")
        
        speaker_audio.export(output_path, format="wav")

    return output_folder