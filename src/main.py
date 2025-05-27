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

def main():
    feature_extractor = FeatureExtractor('2cent.gguf')
    tokenizer = Tokenizer('tokenizer.json')
    decoder = SNACDecoder("decoder_model.onnx", num_bands=3)
    
    # Phonemize
    text = "Hello world! How are you?"
    phonemes = phonemize(text, preserve_punctuation=True)

    # Tokenize
    token_ids = tokenizer.tokenize_ids(phonemes)
    input_ids = torch.LongTensor([token_ids])

    # Extract features using the GGUF model
    features = feature_extractor.extract_features(input_ids)
    # ^ [1, 17, 1024]

    token_id_list = input_ids[0].tolist()  # Convert tensor [1, N] → list of ints
    snac_codes = decoder.extract_snac_codes_from_tokens(token_id_list, tokenizer.get_tokenizer())
    snac_codes = [3981, 426, 3909, 3977, 2426, 1568, 3422, 2068, 1736, 3422, 3422, 1736, 1568, 1568, 3718, 1314, 816, 1467, 2681, 869, 477, 1048, 437, 1865, 3287, 957, 1311, 2624, 3902, 1036, 2511, 897, 1473, 3245, 3272, 2363, 2463, 1123, 270, 3165, 2432, 2808, 4091, 703, 3529, 1030, 98, 2291, 3568, 1291, 3251, 3762, 1157, 3736, 1943, 1194, 1648, 1860, 1308, 3159, 2921, 1655, 1034, 2631, 2576, 386, 2958, 743, 806, 4092, 611, 1656, 1313, 613, 2967, 1689, 2485, 286, 3690, 224, 3219, 1627, 3233, 3371, 1794, 1219, 2416, 2715, 2474, 309, 1694, 148, 3208, 261, 1355, 2694, 2445, 3885, 2830, 1872, 1893, 1226, 1879, 1891, 2014, 3081, 2214, 3602, 1130, 2674, 1188, 3825, 819, 3429, 1953, 2923, 2646, 2044, 3888, 3081, 50, 1834, 1479, 1370, 2065, 490, 1441, 1070, 1153, 524, 3318, 739, 1438, 2985, 2755, 3779, 1135, 1295, 1462, 1846, 148, 1442, 2221, 3319, 2557, 212, 231, 3092, 1198, 2035, 3545, 4042, 3476, 986, 2423, 850, 1373, 3392, 348, 2957, 2259, 718, 3590, 3324, 3223, 551, 2183, 3745, 588, 591, 2612, 2424, 542, 1673, 4015, 2335, 825, 2506, 3668, 3155, 2757, 1508, 2335, 768, 2045, 1763, 2330, 1128, 3325, 2901, 125, 508, 1920, 3512, 2805, 2459, 1050, 1336, 2349, 3927, 455, 1547, 1744, 2879, 1233, 2315, 1440, 663, 1721, 793, 4007, 3554, 3567, 1014, 727, 512, 1754, 3885, 4081, 2030, 724, 392, 3431, 3651, 2644, 3550, 3945, 429, 2554, 2202, 2743, 2068, 2609, 1301, 155, 1736, 3422, 1568, 2068, 1736, 3422, 3422, 1736, 3422, 3422, 3981, 2426, 1568, 1532, 2426, 3602, 1855]
    samples, sample_rate = decoder.decode(snac_codes)
    samples = decoder.normalize_samples(samples)
    sf.write('audio.wav', samples, sample_rate)
    print('Created audio.wav')
    

main()