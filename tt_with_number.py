import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf
import numpy as np
import warnings
from concurrent.futures import ThreadPoolExecutor
import re

# Suppress warnings
warnings.filterwarnings("ignore")

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_dtype = torch.float16 if device.type == "cuda" else torch.float32

# Load model
print("⏳ Loading model and tokenizers...")
try:
    model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts-pretrained",
        torch_dtype=torch_dtype
    ).to(device)
    model.eval()

    try:
        from optimum.bettertransformer import BetterTransformer
        model = BetterTransformer.transform(model)
        print("✅ Enabled BetterTransformer optimization")
    except ImportError:
        print("⚠️ Install optimum for additional speed: pip install optimum")

except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

try:
    tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts-pretrained")
    description_tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder._name_or_path)
except Exception as e:
    print(f"❌ Error loading tokenizers: {e}")
    exit()

# ------------------ 🛠 Number Replacement Preprocessing ------------------
number_map = {
    "0": "ಸೊನ್ನೆ",
    "1": "ಒಂದು",
    "2": "ಎರಡು",
    "3": "ಮೂರು",
    "4": "ನಾಲ್ಕು",
    "5": "ಐದು",
    "6": "ಆರು",
    "7": "ಏಳು",
    "8": "ಎಂಟು",
    "9": "ಒಂಭತ್ತು",
    "10": "ಹತ್ತು",
    "50": "ಐವತ್ತು",
    "100": "ನೂರು",
    "2000": "ಎರಡು ಸಾವಿರ",
    "2024": "ಎರಡು ಸಾವಿರ ಇಪ್ಪತ್ತ್ನಾಲ್ಕು",
    "50000000": "ಐದು ಕೋಟಿ"
}

kannada_digit_map = {
    "೦": "ಸೊನ್ನೆ",
    "೧": "ಒಂದು",
    "೨": "ಎರಡು",
    "೩": "ಮೂರು",
    "೪": "ನಾಲ್ಕು",
    "೫": "ಐದು",
    "೬": "ಆರು",
    "೭": "ಏಳು",
    "೮": "ಎಂಟು",
    "೯": "ಒಂಭತ್ತು"
}

def replace_numbers_with_kannada_words(text):
    # Replace English numerals
    def replace_match(match):
        number = match.group(0)
        return number_map.get(number, number)

    text = re.sub(r'\d+', replace_match, text)

    # Replace Kannada digits
    for digit, word in kannada_digit_map.items():
        text = text.replace(digit, word)

    return text
# ------------------------------------------------------------------------

# Kannada text input
kannada_text = """
ಕನ್ನಡ ಭಾಷೆ ಭಾರತದ ಕರ್ನಾಟಕ ರಾಜ್ಯದ ಅಧಿಕೃತ ಭಾಷೆಯಾಗಿದೆ.
ಇದು ದ್ರಾವಿಡ ಭಾಷಾ ಕುಟುಂಬಕ್ಕೆ ಸೇರಿದೆ ಮತ್ತು ಸುಮಾರು ೫ ಕೋಟಿ ಜನರು ಮಾತನಾಡುವ ಪ್ರಮುಖ ಭಾಷೆಯಾಗಿದೆ.
ಕನ್ನಡವು ಅತ್ಯಂತ ಪ್ರಾಚೀನ ಭಾಷೆಗಳಲ್ಲಿೊಂದಾಗಿದೆ, ಇದರ ಇತಿಹಾಸ ಸುಮಾರು 2000 ವರ್ಷಗಳಷ್ಟು ಹಿಂದಕ್ಕೆ ಹೋಗುತ್ತದೆ.
ಕನ್ನಡ ಸಾಹಿತ್ಯವು ಅಸಂಖ್ಯಾತ ಕವಿಗಳು ಮತ್ತು ಲೇಖಕರಿಂದ ಸಮೃದ್ಧವಾಗಿ ಬೆಳೆದು ಬಂದಿದೆ.
ಕುವೆಂಪು, ಬೇಂದ್ರೆ, ಕಾರಂತರಂತಹ ಮಹಾನ್ ಸಾಹಿತಿಗಳು ಕನ್ನಡಕ್ಕೆ ಅಪಾರ ಕೊಡುಗೆ ನೀಡಿದ್ದಾರೆ.
ಕನ್ನಡ ಲಿಪಿಯು ತನ್ನದೇ ಆದ ಅನನ್ಯ ಸೌಂದರ್ಯ ಮತ್ತು ವೈಜ್ಞಾನಿಕ ರಚನೆಯನ್ನು ಹೊಂದಿದೆ.
ಇಂದು ಕನ್ನಡವು ಕಂಪ್ಯೂಟರ್ ಮತ್ತು ಮೊಬೈಲ್ ತಂತ್ರಜ್ಞಾನದಲ್ಲಿ ಹೆಚ್ಚು ಬಳಕೆಯಾಗುತ್ತಿದೆ.
ಕನ್ನಡ ಚಲನಚಿತ್ರೋದ್ಯಮವು ಸಾಕಷ್ಟು ಪ್ರಗತಿ ಸಾಧಿಸಿದೆ ಮತ್ತು ರಾಷ್ಟ್ರೀಯ ಮಟ್ಟದಲ್ಲಿ ಮನ್ನಣೆ ಪಡೆದಿದೆ.
ಕನ್ನಡದ ಧ್ವನಿಮೂಲಕ ಶುದ್ಧ ಉಚ್ಛಾರಣೆ, ಶೈಲಿ ಮತ್ತು ವ್ಯಾಕರಣ ಶಿಸ್ತಿನಿಂದ ಇದು ನಿರ್ದಿಷ್ಟ ಸ್ಥಾನವನ್ನು ಹೊಂದಿದೆ.
ವಿದ್ಯಾರ್ಥಿಗಳಿಂದ ಹಿಡಿದು ಸಂಶೋಧಕರವರೆಗೆ ಎಲ್ಲರೂ ಕನ್ನಡದ ಮಹತ್ವವನ್ನು ಗುರುತಿಸುತ್ತಿದ್ದಾರೆ.
ಕನ್ನಡ ಭಾಷೆಯ ಬೋಧನೆ ಇತ್ತೀಚಿನ ದಿನಗಳಲ್ಲಿ ಇನ್ಟರ್‌ನೆಟ್ ಮೂಲಕವೂ ಸಹ ಸುಲಭವಾಗಿದೆ.
ಅನೇಕ ಕನ್ನಡ ವೆಬ್‌ಸೈಟ್‌ಗಳು, ಆಪ್‌ಗಳು ಮತ್ತು ಪಾಠಕ್ರಮಗಳು ಈ ಭಾಷೆಯನ್ನು ಮತ್ತಷ್ಟು ಜನಪ್ರಿಯಗೊಳಿಸುತ್ತಿವೆ.
ಕನ್ನಡ ನುಡಿಗಟ್ಟುಗಳು ಮತ್ತು ಗಾದೆಗಳು ಜನಜೀವನದ ಪ್ರತಿ ಅಂಗದಲ್ಲೂ ಶ್ರೇಷ್ಠತೆ ತಾಳಿವೆ.
ನಮ್ಮ ರಾಜ್ಯದ ಆಡಳಿತ, ಶಿಕ್ಷಣ ಮತ್ತು ಮಾಧ್ಯಮಗಳಲ್ಲಿ ಕನ್ನಡದ ಪ್ರಾಬಲ್ಯ ಮುಂದುವರೆದಿದೆ.
ಕನ್ನಡದ ಜೊತೆಗೆ ಬೆಳೆದು ಬಂದ ಜನಪದ ಸಾಹಿತ್ಯ, ಹಾಡುಗಳು, ಮತ್ತು ನೃತ್ಯಗಳು ನಮ್ಮ ಸಂಸ್ಕೃತಿಗೆ ಜೀವಂತತೆ ನೀಡುತ್ತವೆ.
ವಿವಿಧ ಭಾಷೆಗಳ ನಡುವೆ ಇದ್ದರೂ ಕನ್ನಡ ತನ್ನ ವೈಶಿಷ್ಟ್ಯತೆ ಹಾಗೂ ಶ್ರೇಷ್ಠತೆಯಿಂದ ಪೈಪೋಟಿಗೆ ತೊಡಗಿದೆ.
ಪ್ರತಿ ಕನ್ನಡಿಗನು ಕನ್ನಡದ ಬಗ್ಗೆ ಹೆಮ್ಮೆ ಪಡಬೇಕು ಮತ್ತು ನಿತ್ಯ ಬಳಕೆಯಲ್ಲಿ ಪ್ರೋತ್ಸಾಹಿಸಬೇಕು.
ಅಂತರ್ಜಾಲ ಯುಗದಲ್ಲೂ ಕನ್ನಡದ ಸಂಸ್ಕೃತಿಯ ಸಂರಕ್ಷಣೆ ಮತ್ತು ಅಭಿವೃದ್ಧಿಗೆ ನಾವು ಮುಂದಾಗಬೇಕು.
ಬಾಲಕಂದಿಗರಿಂದ ಹಿರಿಯ ನಾಗರಿಕರ ತನಕ ಎಲ್ಲರಿಗೂ ಕನ್ನಡವನ್ನು ಅರಿಯಲು ಹೆಚ್ಚಿನ ಅವಕಾಶಗಳು ಲಭ್ಯವಿವೆ.
ಕನ್ನಡ ನಾಟಕಗಳು, ಸಿನಿಮಾಗಳು, ಕಿರುಚಿತ್ರಗಳು ಸಹ ಈ ಭಾಷೆಯ ಸಾಂಸ್ಕೃತಿಕ ಶಕ್ತಿಯನ್ನು ಬಿಂಬಿಸುತ್ತವೆ.
ಮಟ್ಟಮಟ್ಟದ ಸಾಹಿತ್ಯ ಸಂमेलನಗಳು ಕನ್ನಡ ಸೃಜನಶೀಲತೆಯನ್ನು ಬೆಳೆಸಲು ಸಹಕಾರಿಯಾಗಿವೆ.
ವೈದ್ಯಕೀಯ, ವಿಜ್ಞಾನ, ತಂತ್ರಜ್ಞಾನ ಕ್ಷೇತ್ರಗಳಲ್ಲಿಯೂ ಕನ್ನಡದಲ್ಲಿ ಹೆಚ್ಚಿನ ಸಂಪತ್ತು ಸೃಷ್ಟಿಯಾಗುತ್ತಿದೆ.
ಭದ್ರತೆ, ಸಂಸ್ಕೃತಿ ಮತ್ತು ತಾಯಿ ಭಾಷೆಯ ಗೌರವ ಉಳಿಸಲು ಕನ್ನಡದ ಬಳಕೆ ಅತ್ಯಗತ್ಯವಾಗಿದೆ.
ಕನ್ನಡ ಭಾಷೆಯ ಬಳಕೆಯನ್ನು ಪ್ರಾಚೀನ ಶಾಸನಗಳಿಂದ ಹಿಡಿದು ಇತ್ತೀಚಿನ ಡಿಜಿಟಲ್ ಮಾಧ್ಯಮಗಳವರೆಗೆ ಕಾಣಬಹುದು.
ಇದು ಸಂಸ್ಕೃತದ ಪ್ರಭಾವವನ್ನೂ, ಆದರೆ ತನ್ನದೇ ಆದ ಶಬ್ದಸಂಪತ್ತಿಯನ್ನೂ ಹೊಂದಿರುವ ಭಾಷೆಯಾಗಿದ್ದು, ಪರಿಷ್ಕೃತರೂಪದಲ್ಲಿ ಬೆಳೆಯುತ್ತಿದೆ.
ಕನ್ನಡದಲ್ಲಿ ಇಂದಿಗೂ ಹಲವು ಗ್ರಾಮೀಣ ಶೈಲಿಗಳು, ಪ್ರಾದೇಶಿಕ ಉಪಭಾಷೆಗಳು ಜೀವಂತವಾಗಿವೆ.
ಇವುಗಳಿಂದ ಭಾಷೆಯ ವೈವಿಧ್ಯತೆ ಹಾಗೂ ಸ್ಥಳೀಯ ಸಂಸ್ಕೃತಿಯ ಪೂರಕತೆ ಸ್ಪಷ್ಟವಾಗಿ ಗೋಚರಿಸುತ್ತದೆ.
ಅಕ್ಷರಮಾಲೆಯ ವೈಜ್ಞಾನಿಕ ವಿನ್ಯಾಸದಿಂದ ಕನ್ನಡವನ್ನು ಕಂಪ್ಯೂಟರ್ ಪ್ರೊಗ್ರಾಮಿಂಗ್ ಹಾಗೂ ಕಲಿಕಾ ಸಾಧನಗಳಲ್ಲೂ ಬಳಸಬಹುದು.
ಕನ್ನಡದ ಬಾಲ ಸಾಹಿತ್ಯ ಮಕ್ಕಳಲ್ಲಿ ಭಾಷಾ ಪ್ರೀತಿ ಬೆಳೆಸುವಲ್ಲಿ ಪ್ರಮುಖ ಪಾತ್ರ ವಹಿಸುತ್ತಿದೆ.
ವಿಶ್ವದ ಹಲವೆಡೆ ನೆಲೆಸಿರುವ ಕನ್ನಡಿಗರು ಕನ್ನಡದ ಹರಡುವಿಕೆಗೆ ಬಹುಮೂಲ್ಯ ಕೊಡುಗೆ ನೀಡುತ್ತಿದ್ದಾರೆ.
ಕನ್ನಡ ಭಾಷೆಯ ಪ್ರಸಾರಕ್ಕಾಗಿ ಪ್ರಸ್ತುತ ಹಲವು ಪಾಡ್‌ಕ್ಯಾಸ್ಟ್‌ಗಳು, ಯೂಟ್ಯೂಬ್ ಚಾನೆಲ್‌ಗಳು ಜನಪ್ರಿಯವಾಗಿವೆ.
ಕನ್ನಡದ ವ್ಯಾಕರಣ, ತತ್ವಶಾಸ್ತ್ರ, ಮತ್ತು ಚರಿತ್ರೆ ಕುರಿತ ಅಧ್ಯಯನವೂ ವಿಶ್ವದ ಹಲವು ವಿಶ್ವವಿದ್ಯಾಲಯಗಳಲ್ಲಿ ನಡೆಯುತ್ತಿದೆ.
ಇಂತಹ ಸಂಪೂರ್ಣತೆಯಿಂದ ಕನ್ನಡ ನಿಜಕ್ಕೂ ಜಗತ್ತಿನ ಶ್ರೇಷ್ಠ ಭಾಷೆಗಳ ಪೈಕಿ ಒಂದಾಗಿ ಬೆಳೆಯುತ್ತಿದೆ.




""".strip()

# ✅ Apply number word replacement
kannada_text = replace_numbers_with_kannada_words(kannada_text)

# Voice description
voice_description = "A professional female speaker Anu, speaks with clear pronunciation and moderate pacing"


print(f"🔠 Processing {len(kannada_text)} characters...")

# Split into sentences
sentences = [s.strip() + '.' for s in kannada_text.split('.') if s.strip()]
print(f"📜 Found {len(sentences)} sentences")

# Pre-tokenize description
with torch.no_grad():
    description_inputs = description_tokenizer(
        voice_description,
        return_tensors="pt",
        max_length=512,
        truncation=True
    ).to(device)

# Generation config
generation_config = {
    'do_sample': True,
    'temperature': 0.95,
    'top_k': 50,
    'top_p': 0.95,
    'max_new_tokens': 1024,
    'num_beams': 1,
    'use_cache': True
}

# Sentence processor
def process_sentence(sentence):
    try:
        with torch.no_grad():
            inputs = tokenizer(
                sentence,
                return_tensors="pt",
                max_length=512,
                truncation=True
            ).to(device)

            audio = model.generate(
                input_ids=description_inputs.input_ids,
                attention_mask=description_inputs.attention_mask,
                prompt_input_ids=inputs.input_ids,
                prompt_attention_mask=inputs.attention_mask,
                **generation_config
            )
            return audio.cpu().numpy().squeeze().astype('float32')
    except Exception as e:
        print(f"⚠️ Error processing sentence: {e}")
        return None

# Generate audio
sampling_rate = model.config.sampling_rate
full_audio = np.array([])
batch_size = 4

print("🎵 Generating audio in batches...")
for i in range(0, len(sentences), batch_size):
    batch = sentences[i:i + batch_size]
    with ThreadPoolExecutor() as executor:
        batch_results = list(executor.map(process_sentence, batch))

    for j, audio_arr in enumerate(batch_results):
        if audio_arr is not None:
            full_audio = np.concatenate((full_audio, audio_arr))
            if j < len(batch_results) - 1:
                full_audio = np.concatenate((
                    full_audio,
                    np.zeros(int(0.15 * sampling_rate))
                ))
    print(f"✅ Processed batch {i // batch_size + 1}/{(len(sentences) // batch_size) + 1}")

# Save output
output_file = "kannada_tts_optimized.wav"
sf.write(output_file, full_audio, sampling_rate)
print(f"\n🎉 Successfully saved to {output_file}")
print(f"⏱️ Duration: {len(full_audio)/sampling_rate:.2f} seconds")
