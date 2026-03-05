# Voxka – Kannada Emotional Text-to-Speech (TTS)

Voxka is a Kannada emotional text-to-speech system built on top of the open-source Parler-TTS architecture.
The project extends Parler-TTS by integrating emotion embeddings to generate expressive Kannada speech.

The system converts Kannada text → phonemes → emotion-conditioned speech audio.

#Features

Kannada Text-to-Speech generation

Emotion-conditioned speech synthesis

Custom emotion embedding vectors

Fully offline inference

Based on open-source Parler-TTS

Extendable to new voices and emotions

## Architecture
Kannada Text
      │
      ▼
Text Encoder / Tokenizer
      │
      ▼
Emotion Vector Embedding
      │
      ▼
Parler-TTS Decoder
      │
      ▼
Audio Tokens
      │
      ▼
Vocoder (HiFi-GAN / Codec)
      │
      ▼
Speech Output

Parler-TTS generates speech tokens autoregressively using a transformer architecture and converts them to waveform audio.

## Repository Structure
voxka-kannada-tts/
│
├── dataset/
│   ├── audio/
│   ├── transcripts/
│   └── emotion_labels.csv
│
├── emotion_vectors/
│   └── emotion_embeddings.npy
│
├── models/
│   └── parler_tts_kannada/
│
├── scripts/
│   ├── train.py
│   ├── infer.py
│   ├── preprocess.py
│
├── utils/
│   ├── phoneme_converter.py
│   └── emotion_encoder.py
│
└── README.md
## Installation
1. Clone Parler-TTS
git clone https://github.com/huggingface/parler-tts.git
cd parler-tts

Parler-TTS is a lightweight open-source library designed to generate natural sounding speech with controllable voice styles.

2. Create Python Environment
python3 -m venv voxka-env
source voxka-env/bin/activate
3. Install Dependencies

Install PyTorch first.

pip install torch torchaudio torchvision

Then install Parler-TTS.

pip install git+https://github.com/huggingface/parler-tts.git
4. Install Additional Libraries
pip install transformers
pip install soundfile
pip install numpy
pip install accelerate

## Optional logging:

pip install wandb
Download Base TTS Model

Example:

from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer

model = ParlerTTSForConditionalGeneration.from_pretrained(
"parler-tts/parler_tts_mini_v0.1"
)

tokenizer = AutoTokenizer.from_pretrained(
"parler-tts/parler_tts_mini_v0.1"
)

These pretrained checkpoints are trained on thousands of hours of speech data and support style-controlled voice generation.

## Adding Kannada Support
Step 1 — Kannada Text Normalization

Convert raw Kannada text to normalized text.

Example:

"ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರಾ"
→ normalized Kannada tokens
Step 2 — Kannada Phoneme Conversion

Create a grapheme-to-phoneme converter.

Example mapping:

ಕ → ka
ಗ → ga
ತ → ta
ನ → na

### Output:

ನಮಸ್ಕಾರ
→ na ma ska ra
Emotion Vector Integration

Voxka adds emotion embeddings to control speech style.

Example emotions:

neutral
happy
sad
angry
excited
calm

## Emotion vectors are stored as:

emotion_vectors/
emotion_embeddings.npy

Example structure:

{
 "neutral": [0.1,0.4,0.3,...],
 "happy":   [0.8,0.6,0.2,...],
 "sad":     [0.2,0.1,0.7,...]
}
Merging Emotion Vectors with Text Embeddings

Emotion vectors are merged with the text encoder output.

Example:

text_embedding = text_encoder(text_tokens)

emotion_vector = emotion_embeddings["happy"]

combined_embedding = text_embedding + emotion_vector

or

combined_embedding = torch.cat(
    (text_embedding, emotion_vector), dim=-1
)

The merged embedding is then passed into the Parler-TTS decoder.

## Training Voxka

Run training script:

python scripts/train.py \
--dataset dataset/ \
--emotion_vectors emotion_vectors/emotion_embeddings.npy \
--output models/voxka

## Training pipeline:

Audio + Transcript
        │
        ▼
Emotion Annotation
        │
        ▼
Text Encoding
        │
        ▼
Emotion Vector Merge
        │
        ▼
Parler-TTS Training
Running Inference

### Example script:

import torch
import soundfile as sf
from parler_tts import ParlerTTSForConditionalGeneration

text = "ನಮಸ್ಕಾರ ಹೇಗಿದ್ದೀರಾ"

emotion = "happy"

audio = model.generate(text, emotion)

sf.write("output.wav", audio, 22050)
Example Usage

### Input

Text: ನಮಸ್ಕಾರ
Emotion: happy

Output

Expressive Kannada speech
## Future Improvements

Custom Kannada voice dataset

Speaker identity embeddings

Emotion intensity scaling

Real-time TTS inference

Web API for Voxka

## References

Parler-TTS GitHub repository

Hugging Face speech generation libraries

Indic speech research datasets
