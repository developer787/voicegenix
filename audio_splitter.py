import os
import numpy as np
from resemblyzer import VoiceEncoder, preprocess_wav
from pydub import AudioSegment


def speaker_diarization(audio_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load audio file
    wav = preprocess_wav(audio_path)
    wav_duration = len(wav) / 16000

    # Derive embeddings and determine the speaker clusters
    encoder = VoiceEncoder()
    embeds = encoder.embed_utterance(wav, rate=0.1)
    clusterer = encoder.cluster(embeds)
    cluster_labels = clusterer.labels_

    n_speakers = max(cluster_labels) + 1

    # Initialize audio segment variables
    audio_segment = AudioSegment.from_file(audio_path)
    audio_splits = [AudioSegment.empty()] * n_speakers

    # Iterate through audio and assign segments to corresponding speakers
    start = 0
    for cluster_label, duration_ in zip(cluster_labels, np.diff(np.concatenate(([0.], clusterer.cluster_centers_[:, 1])))):
        duration_ms = int(duration_ * 1000)
        audio_splits[cluster_label] += audio_segment[start:start + duration_ms]
        start += duration_ms

    # Save audio segments by speakers in output directories
    for i, audio_split in enumerate(audio_splits):
        speaker_dir = os.path.join(output_dir, f'speaker_{i + 1}')
        os.makedirs(speaker_dir, exist_ok=True)

        audio_path = os.path.join(speaker_dir, f'speaker_{i + 1}.wav')
        audio_split.export(audio_path, format='wav')
        print(f'Saved Speaker {i + 1} audio at {audio_path}')