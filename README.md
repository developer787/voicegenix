# Speaker-wise Audio Splitter

This project takes an input audio file, detects speakers, and splits the audio into separate files based on the identified speakers. The app uses Python with `librosa` and `pyannote.audio` for audio processing, and the speaker diarization model for identifying speakers in the audio.

## Installation

1. Clone this repository:

```bash
git clone https://github.com/developer787/voicegenix.git
cd voicegenix
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### With Python Script:

1. Run the `audio_splitter.py` script with your input audio file:

```bash
python audio_splitter.py path/to/your/audio_file.wav
```

The script will detect speakers in the input audio file, split the audio based on the speakers, and save the resulting files in the output folder.

### With Google Colab:

1. Open the Jupyter notebook in Google Colab by visiting the following link:

```
https://colab.research.google.com/github/developer787/voicegenix/blob/main/audio_splitter_colab.ipynb
```

2. Follow the instructions in the notebook to upload your audio file, run the speaker diarization and splitting process, and download the split audio files.

## License

This project is licensed under the terms of the [MIT License](LICENSE).