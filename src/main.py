"""
uv sync
wget https://github.com/taylorchu/2cent-tts/releases/download/v0.2.0/2cent.gguf
wget https://github.com/taylorchu/2cent-tts/releases/download/v0.2.0/tokenizer.json
wget https://huggingface.co/onnx-community/snac_24khz-ONNX/resolve/main/onnx/decoder_model.onnx

uv run src/main.py
"""

from phonemize import phonemize
from features import FeatureExtractor
from tokens import Tokenizer
from decoder import SNACDecoder
import torch
import soundfile as sf
import re
import sys
from strategy6 import test_snac_strategy_6, test_snac_strategy_6_detailed


def generate_audio_tokens(text, tokenizer, feature_extractor, max_new_tokens=1024):
    """
    Generate tokens from phonemes using the Qwen3 model
    """
    
    # Step 1: Phonemize text
    phonemes = phonemize(text, language='en-us', backend='espeak', with_stress=True)
    
    print(f"Phonemes: {phonemes}")
    
    # Step 2: Tokenize phonemes
    token_ids = tokenizer.tokenize_ids(phonemes) 
    token_ids = [x for x in token_ids if x != 4136]
    token_ids = token_ids + [2] + [4136]
    
    input_ids = torch.LongTensor([token_ids])
    print(f"Input shape: {input_ids.shape}")
    print(f"Input tokens: {[tokenizer.get_tokenizer().id_to_token(token) for token in token_ids]}")
    
    # Step 3: Generate audio tokens using the causal LM
    with torch.no_grad():
        # Try to get pad_token_id and eos_token_id safely
        pad_token_id = None
        eos_token_id = None
        
        try:
            if hasattr(tokenizer.get_tokenizer(), 'pad_token_id'):
                pad_token_id = tokenizer.get_tokenizer().pad_token_id
        except:
            pad_token_id = 1  # Default 
        try:
            eos_token_id = tokenizer.get_tokenizer().token_to_id("</s>") 
            # if hasattr(tokenizer.get_tokenizer(), 'eos_token_id'):
            #     eos_token_id = tokenizer.get_tokenizer().eos_token_id
        except:
            eos_token_id = None  # Let model decide
        
        # Use the underlying Qwen3ForCausalLM model to generate
        generated_ids = feature_extractor.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
    
    new_tokens = generated_ids[0][len(token_ids):]
    
    return new_tokens

def get_logits_and_sample(text, tokenizer, feature_extractor):
    """
    Alternative: Get logits and manually sample tokens
    """
    # Phonemize and tokenize
    phonemes = phonemize(text, preserve_punctuation=True)
    token_ids = tokenizer.tokenize_ids(phonemes)
    input_ids = torch.LongTensor([token_ids])
    
    with torch.no_grad():
        # Get logits from the model
        outputs = feature_extractor.model(input_ids)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]
        
        # Get the logits for the last position (next token prediction)
        next_token_logits = logits[0, -1, :]  # Shape: [vocab_size]
        
        # Sample or take the most likely token
        next_token_id = torch.argmax(next_token_logits).item()
        
        # Convert to string
        next_token_str = tokenizer.get_tokenizer().id_to_token(next_token_id)
        
        print(f"Next predicted token: {next_token_str} (ID: {next_token_id})")
        
        return next_token_str, next_token_logits

def check_vocabulary_for_audio_tokens(tokenizer):
    """
    Check if the tokenizer vocabulary contains audio tokens
    """
    # Try different ways to get vocab size
    vocab_size = None
    try:
        vocab_size = tokenizer.get_tokenizer().vocab_size
    except AttributeError:
        try:
            vocab_size = len(tokenizer.get_tokenizer().get_vocab())
        except:
            # Use the model's vocab size from config
            vocab_size = 6000  # From your FeatureExtractor config
    
    audio_tokens = []
    print(f"Checking vocabulary of size {vocab_size} for audio tokens...")
    
    # Sample a range of token IDs to see what tokens exist
    sample_ranges = [
        range(0, 100),          # First 100 tokens
        range(vocab_size-100, vocab_size),  # Last 100 tokens
        range(1000, 1100),      # Middle range
        range(2000, 2100),      # Another middle range
        range(4000, 4100),      # Higher range where audio tokens might be
    ]
    
    for token_range in sample_ranges:
        for token_id in token_range:
            try:
                token_str = tokenizer.get_tokenizer().id_to_token(token_id)
                if token_str and ('audio' in token_str.lower() or token_str.startswith('<') or token_str.startswith('▁<')):
                    audio_tokens.append((token_id, token_str))
            except:
                continue
    
    print(f"Found potential audio tokens: {audio_tokens[:20]}")
    return audio_tokens

def debug_model_generation(text, tokenizer, feature_extractor):
    """
    Debug the model's generation capabilities
    """
    phonemes = phonemize(text, language='en-us', backend='espeak', with_stress=True)
  
    token_ids = tokenizer.tokenize_ids(phonemes)
    input_ids = torch.LongTensor([token_ids])
    
    print("=== MODEL DEBUG INFO ===")
    print(f"Model type: {type(feature_extractor.model)}")
    print(f"Config vocab size: {feature_extractor.config.vocab_size}")
    print(f"Hidden size: {feature_extractor.config.hidden_size}")
    
    # Debug tokenizer
    print(f"Tokenizer type: {type(tokenizer)}")
    print(f"Tokenizer methods: {[m for m in dir(tokenizer) if not m.startswith('_')]}")
    
    # Test forward pass
    with torch.no_grad():
        outputs = feature_extractor.model(input_ids)
        print(f"Logits shape: {outputs.logits.shape}")
        print(f"Logits range: {outputs.logits.min().item():.3f} to {outputs.logits.max().item():.3f}")
        
        # Check if model has generate method
        if hasattr(feature_extractor.model, 'generate'):
            print("✓ Model has generate() method")
        else:
            print("✗ Model missing generate() method")
        
        # Show some sample tokens
        print("\nSample token mappings:")
        for i in range(0, min(50, feature_extractor.config.vocab_size), 5):
            try:
                token_str = tokenizer.get_tokenizer().id_to_token(i)
                print(f"  {i}: '{token_str}'")
            except:
                print(f"  {i}: <error>")
                break

# Main usage function

def main(): 

    version = "v2"
    text = "Hello world, this is a test"
    # Parse args: first can be version (e.g., v1 or v2), rest is text
    if len(sys.argv) > 1:
        if sys.argv[1] in ["v1", "v2"]:
            version = sys.argv[1]
            text = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else text
        else:
            text = " ".join(sys.argv[1:])

    # Version-based selection
    if version == "v1":
        feature_extractor = FeatureExtractor('2cent_v1.gguf', 64)
    else:
        feature_extractor = FeatureExtractor('2cent.gguf', 256)

    tokenizer = Tokenizer('tokenizer.json')
    decoder = SNACDecoder("decoder_model.onnx", num_bands=3)
    
    # Phonemize 
    #it doesn't like punctuations ! ? .
    #text = "Hello world, this is a test"
       
    # Debug first 
    # debug_model_generation(text, tokenizer, feature_extractor)
    # check_vocabulary_for_audio_tokens(tokenizer)
    
    # Try to generate audio tokens
    try:
        new_tokens = generate_audio_tokens(text, tokenizer, feature_extractor)
        #print(f"\nFinal audio tokens: {audio_tokens}")


        new_tokens = new_tokens.tolist()
        while new_tokens[0] == 4136:
            new_tokens = new_tokens[1:]

        eos_token = 3
        if new_tokens and new_tokens[-1] == eos_token:
            new_tokens = new_tokens[:-1]
            
        # for i in range(7):
        #     success = test_snac_strategy_6(new_tokens, "tokenizer.json", "decoder_model.onnx")
        #     if success:
        #         print("Strategy 6 PASSED")
        #         break
        #     else:
        #         new_tokens = new_tokens[1:]

        new_tokens = [x - 4 for x in new_tokens]
        #snac_codes = [int(re.search(r'\d+', token).group()) for token in audio_tokens]
        samples, sample_rate = decoder.decode(new_tokens)
        samples = decoder.normalize_samples(samples)
        sf.write('audio.wav', samples, sample_rate)
        print('Created audio.wav')

    
    except Exception as e:
        print(f"Generation failed: {e}")

main()
