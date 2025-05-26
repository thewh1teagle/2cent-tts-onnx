"""
uv sync
wget https://github.com/taylorchu/2cent-tts/releases/download/v0.2.0/2cent.gguf
wget https://github.com/taylorchu/2cent-tts/releases/download/v0.2.0/tokenizer.json

uv run src/main.py
"""

from phonemize import phonemize
from features import FeatureExtractor
from tokens import Tokenizer
from snac import SNAC
import torch

def main():
    feature_extractor = FeatureExtractor('2cent.gguf')
    tokenizer = Tokenizer('tokenizer.json')
    snac_decoder = SNAC.from_pretrained("hubertsiuzdak/snac_32khz").eval().cuda()
    
    # Phonemize
    text = "Hello world! How are you?"
    phonemes = phonemize(text, preserve_punctuation=True)

    # Tokenize
    token_ids = tokenizer.tokenize_ids(phonemes)
    input_ids = torch.LongTensor([token_ids])

    # Extract features
    features = feature_extractor.extract_features(input_ids)

    # Decode with snac
    with torch.inference_mode():
        codec = ...
        audio_hat = snac_decoder.decode(codec)
    

main()
    
    
