import os
from pydub import AudioSegment
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path


def split_audio_file(input_audio_file: str):
    wav_fpath = Path(input_audio_file)
    wav = preprocess_wav(wav_fpath)
    encoder = VoiceEncoder("cpu")

    embedding = encoder.embed_utterance(wav)
    speaker_embeds = [embedding]

    times, speakers = encoder.find_speaker_times(wav, speaker_embeds)

    audio = AudioSegment.from_wav(input_audio_file)

    for i, (start, end) in enumerate(times):
        speaker_path = os.path.join("speakers", "speaker_" + str(speakers[i]))
        os.makedirs(speaker_path, exist_ok=True)

        segment = audio[start * 1000 : end * 1000] # pydub works with milliseconds
        segment.export(
            os.path.join(speaker_path, f"segment_{i}.wav"), format="wav"
        )


if __name__ == "__main__":
    input_audio_file = "input_audio_file.wav"
    split_audio_file(input_audio_file)