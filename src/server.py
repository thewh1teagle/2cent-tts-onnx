from flask import Flask, request, send_file
import io
import re
import requests
import time
import torch
import soundfile as sf
from features import FeatureExtractor
from llama_cpp import Llama


from phonemize import phonemize
from tokens import Tokenizer
from decoder import SNACDecoder
from transformers import PreTrainedTokenizerFast

app = Flask(__name__)

# Global model objects
tokenizer = None
decoder = None
feature_extractor = None
real_tokenizer = None
llm = None

def initialize_loaders():
    global tokenizer, decoder, feature_extractor, real_tokenizer, llm

    real_tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer.json")  

    # Uncomment if you want to use feature_extractor
    #feature_extractor = FeatureExtractor('2cent.gguf', 256)
    tokenizer = Tokenizer('tokenizer.json')
    decoder = SNACDecoder("decoder_model.onnx", num_bands=3)

    # Initialize with deterministic parameters
    llm = Llama(
        model_path="2cent.gguf",
        seed=42,
        temperature=0.0,  # Deterministic
        top_k=1,         # Only top token
        top_p=1.0,       # No nucleus sampling
        repeat_penalty=1.0,
        n_ctx=4096,
        n_gpu_layers=-1  # Use all layers on GPU
    )



def generate_audio_tokens_pytorch(text, tokenizer, max_new_tokens=1024):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    phonemes = phonemize(text, language='en-us', backend='espeak', with_stress=True)
    print(f"Phonemes: {phonemes}")

    token_ids = tokenizer.tokenize_ids(phonemes)
    token_ids = [x for x in token_ids if x != 4136]
    token_ids = token_ids + [2] + [4136]
    print(token_ids)


    build_phonemes = ""
    for token_id in token_ids:
        token = real_tokenizer.convert_ids_to_tokens([token_id])[0]
        build_phonemes = build_phonemes + "||" + token
    
    print( build_phonemes )


    # Create input tensor and move to correct device
    input_ids = torch.LongTensor([token_ids]).to(device)
    feature_extractor.model.to(device)

    with torch.inference_mode():
        # Try to get pad_token_id and eos_token_id safely
        pad_token_id = 1
        eos_token_id = 3

        start = time.time()

        generated_ids = feature_extractor.model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.01,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )

        print(f"Token generation time: {time.time() - start:.4f} seconds")

    print("Model device:", next(feature_extractor.model.parameters()).device)
    print("Input tensor device:", input_ids.device)


    new_tokens = generated_ids[0][len(token_ids):]
    new_tokens = new_tokens.tolist()


    while new_tokens[0] == 4136:
        new_tokens = new_tokens[1:]

    eos_token = 3
    if new_tokens and new_tokens[-1] == eos_token:
        new_tokens = new_tokens[:-1]
    new_tokens = [x - 4 for x in new_tokens]   

    print(new_tokens)

    return new_tokens
    


def generate_audio_tokens_llamacpp(text, tokenizer, max_new_tokens=1024*10):

    start = time.time()

    phonemes = phonemize(text, language='en-us', backend='espeak', with_stress=True)
    token_ids = tokenizer.tokenize_ids(phonemes)

    token_ids = [x for x in token_ids if x != 4136]
    tokens = [4136] + token_ids + [2]


    
    llm.reset()
    llm.eval(tokens)
    
    output_tokens = []
    for _ in range(max_new_tokens):
        token = llm.sample()
        output_tokens.append(token)
        
        if token == llm.token_eos() or token == 151645:
            break
            
        llm.eval([token])

    print(f"llamacpp took {time.time() - start:.2f}s")


    ###########################
    new_tokens = output_tokens
    while new_tokens[0] == 4136:
        new_tokens = new_tokens[1:]

    eos_token = 3
    if new_tokens and new_tokens[-1] == eos_token:
        new_tokens = new_tokens[:-1]
    new_tokens = [x - 4 for x in new_tokens]   


    return new_tokens


def generate_audio_tokens_lmstudio(text, tokenizer, max_new_tokens=1024*10):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    phonemes = phonemize(text, language='en-us', backend='espeak', with_stress=True)
    
    token_ids = tokenizer.tokenize_ids(phonemes)
    build_phonemes = ""
    for token_id in token_ids:
        token = real_tokenizer.convert_ids_to_tokens([token_id])[0]
        build_phonemes = build_phonemes + "||" + token
    
    print( build_phonemes )
    build_phonemes = build_phonemes.replace("||▁", " ")
    build_phonemes = build_phonemes.replace("||", "")
    phonemes = build_phonemes.strip() + "<s>"
    print(f"Phonemes: {phonemes}")
    
    # phonemes = text.rstrip() + "<s>"
    # print(f"Phonemes: {phonemes}")

    url = "http://localhost:1234/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "tts-1",
        "prompt": phonemes,
        "max_tokens": 1024,
        "temperature": 0.01,    # Sometimes 0.0 enables fallback sampling
        "top_k": 1,
        "top_p": 0.1,          # Very restrictive
        "repetition_penalty": 1.0,
        "seed": 42,
        "mirostat": 0,         # Disable mirostat
        "mirostat_tau": 5.0,
        "mirostat_eta": 0.1
    }

    response = requests.post(url, headers=headers, json=data)
    response_data = response.json()
    response_text = response_data["choices"][0]["text"]
    new_tokens = [int(n) for n in re.findall(r"<audio_(\d+)>", response_text)]
    return new_tokens

def generate_audio(text):
    
    #LM STUDIO DOESN'T RETURN THE CORRECT TOKENS
    new_tokens = generate_audio_tokens_llamacpp(text, tokenizer)
    #new_tokens = generate_audio_tokens_lmstudio(text, tokenizer)

    #new_tokens = generate_audio_tokens_pytorch( text, tokenizer )
    while new_tokens and new_tokens[0] == 4136:
        new_tokens = new_tokens[1:]

    eos_token = 3
    if new_tokens and new_tokens[-1] == eos_token:
        new_tokens = new_tokens[:-1]

    start = time.time()
    samples, sample_rate = decoder.decode(new_tokens)
    print(f"Decoder took: {time.time() - start:.2f}s")

    samples = decoder.normalize_samples(samples)

    buffer = io.BytesIO()
    sf.write(buffer, samples, sample_rate, format='WAV')
    buffer.seek(0)
    return buffer

@app.route('/tts')
def tts():
    text = request.args.get('text')
    if not text:
        return "Missing 'text' parameter", 400
    try:
        start = time.time()
        audio_io = generate_audio(text)
        print(f"Generated audio for '{text}' in {time.time() - start:.2f}s")
        return send_file(
            audio_io,
            mimetype='audio/wav',
            as_attachment=False,
            download_name='tts.wav'
        )
    except Exception as e:
        return f"Audio generation failed: {e}", 500

if __name__ == "__main__":
    initialize_loaders()
    app.run(host='0.0.0.0', port=5000)

