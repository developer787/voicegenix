import os
from pydub import AudioSegment
from pyannote.audio import Inference
from pyannote.pipeline import Diarization
from pyannote.core import SlidingWindowFeature

# Set up speaker diarization model
mpdel_sad = Inference("sad_dihard", device="cpu")
model_scd = Inference("scd_dihard", device="cpu")
model_embedding = Inference("emb_voxceleb", device="cpu")
pipeline = Diarization(sad=model_sad, scd=model_scd, embedding=model_embedding, method="affinity_propagation")

def to_audio_segment(annotation, audio_file):
    audio_segment = AudioSegment.from_file(audio_file)
    start_time, end_time = int(annotation.start * 1000), int(annotation.end * 1000)
    return audio_segment[start_time:end_time]


def speaker_diarization(audio_path, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Speaker diarization
    diarization = pipeline({'audio': audio_path})

    speakers = {}
    for speaker, segment in diarization.itertracks(yield_label=True):
        if speaker not in speakers:
            speakers[speaker] = []

        audio_segment = to_audio_segment(segment, audio_path)
        speakers[speaker].append(audio_segment)

    # Save audio segments by speakers
    for speaker, audio_segments in speakers.items():
        speaker_dir = os.path.join(output_dir, f'speaker_{speaker}')
        if not os.path.exists(speaker_dir):
            os.makedirs(speaker_dir)

        for i, audio_segment in enumerate(audio_segments):
            audio_path = os.path.join(speaker_dir, f'segment_{i + 1}.wav')
            audio_segment.export(audio_path, format='wav')
            print(f'Saved Speaker {speaker} segment {i + 1} audio at {audio_path}')