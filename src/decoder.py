import numpy as np
import onnxruntime as ort
import torch
import re

def parse_hierarchical_snac_tokens(tokens, num_bands=3):
    """
    Parse SNAC's depth-first hierarchical tokens into band structure.
    
    SNAC uses depth-first traversal where each node has exactly 2 children.
    For 3 bands: 7 tokens per complete tree (1 + 2 + 4 = 7)
    
    Example: [1,2,3,4,5,6,7] represents:
    Band 0: [1]        (root)
    Band 1: [2, 5]     (children of 1)  
    Band 2: [3,4, 6,7] (children of 2 and 5)
    
    Args:
        tokens (list): Flat list of hierarchical token IDs
        num_bands (int): Number of hierarchical bands
        
    Returns:
        list: List of token arrays per band [band0, band1, band2, ...]
    """
    # Calculate tokens per complete tree
    tokens_per_tree = sum(2**i for i in range(num_bands))
    
    # Pad if necessary
    if len(tokens) % tokens_per_tree != 0:
        padding_needed = tokens_per_tree - (len(tokens) % tokens_per_tree)
        tokens = tokens + [0] * padding_needed
        print(f"Padded {padding_needed} tokens for complete trees")
    
    num_trees = len(tokens) // tokens_per_tree
    print(f"Processing {num_trees} hierarchical trees ({tokens_per_tree} tokens each)")
    
    # Initialize bands
    band_tokens = [[] for _ in range(num_bands)]
    
    # Process each complete tree
    for tree_idx in range(num_trees):
        tree_start = tree_idx * tokens_per_tree
        tree_tokens = tokens[tree_start:tree_start + tokens_per_tree]
        
        # Parse single tree using depth-first structure
        token_idx = 0
        
        def parse_node(band_level):
            nonlocal token_idx
            if band_level >= num_bands or token_idx >= len(tree_tokens):
                return
                
            # Add current token to its band
            current_token = tree_tokens[token_idx]
            band_tokens[band_level].append(current_token)
            token_idx += 1
            
            # Recursively parse children (binary tree: 2 children per node)
            for _ in range(2):  # Each node has exactly 2 children
                parse_node(band_level + 1)
        
        # Start parsing from root (band 0)
        parse_node(0)
    
    # Print band statistics
    for i, band in enumerate(band_tokens):
        print(f"Band {i}: {len(band)} tokens")
    
    return band_tokens


class SNACDecoder:
    def __init__(self, model_path, num_bands=3):
        self.model_path = model_path
        self.session = None
        self.sample_rate = 24000
        self.num_bands = num_bands

        self._initialize()

    def _initialize(self):
        """Load the ONNX model and inspect its structure"""
        self.session = ort.InferenceSession(self.model_path)
        print("SNAC model loaded successfully")
        
        print("\nModel inputs:")
        for inp in self.session.get_inputs():
            print(f"  {inp.name}: shape={inp.shape}, dtype={inp.type}")
        
        print("\nModel outputs:")
        for out in self.session.get_outputs():
            print(f"  {out.name}: shape={out.shape}, dtype={out.type}")

    def decode(self, token_ids):
        """
        Decode hierarchical SNAC tokens into audio
        
        Args:
            token_ids (list): Depth-first hierarchical token sequence
            
        Returns:
            numpy.ndarray: Audio waveform
        """
        if not self.session:
            raise RuntimeError("Model not initialized. Call initialize() first.")

        print(f"Decoding {len(token_ids)} hierarchical tokens...")
        
        # Parse hierarchical structure into bands
        band_tokens = parse_hierarchical_snac_tokens(token_ids, self.num_bands)
        
        # Prepare ONNX inputs
        inputs = {}
        input_names = [inp.name for inp in self.session.get_inputs()]
        
        # Try different common SNAC input formats
        if all(f'audio_codes.{i}' in input_names for i in range(self.num_bands)):
            # Format: Separate inputs per band (audio_codes.0, audio_codes.1, ...)
            print("Using separate band inputs format")
            for band_idx in range(self.num_bands):
                input_name = f'audio_codes.{band_idx}'
                # Shape: [batch_size, sequence_length]
                band_array = np.array([band_tokens[band_idx]], dtype=np.int64)
                inputs[input_name] = band_array
                print(f"  {input_name}: {band_array.shape}")
                
        elif 'codes' in input_names:
            # Format: Combined codes tensor [batch, bands, time]
            print("Using combined codes format")
            max_length = max(len(band) for band in band_tokens)
            
            # Pad all bands to same length
            padded_bands = []
            for band in band_tokens:
                padded_band = band + [0] * (max_length - len(band))
                padded_bands.append(padded_band)
            
            codes_array = np.array([padded_bands], dtype=np.int64)
            inputs['codes'] = codes_array
            print(f"  codes: {codes_array.shape}")
            
        elif 'audio_codes' in input_names:
            # Format: Single audio_codes input
            print("Using single audio_codes format")
            # Stack bands: [batch, bands, time]
            max_length = max(len(band) for band in band_tokens)
            padded_bands = []
            for band in band_tokens:
                padded_band = band + [0] * (max_length - len(band))
                padded_bands.append(padded_band)
            
            codes_array = np.array([padded_bands], dtype=np.int64)
            inputs['audio_codes'] = codes_array
            print(f"  audio_codes: {codes_array.shape}")
            
        else:
            raise ValueError(f"Unrecognized input format. Available inputs: {input_names}")

        # Run inference
        try:
            outputs = self.session.run(None, inputs)
            print("✓ Inference completed successfully")
        except Exception as e:
            print(f"✗ ONNX Runtime Error: {e}")
            print("\nDebugging info:")
            for name, tensor in inputs.items():
                print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}")
            raise

        # Extract and process audio output
        audio = outputs[0]
        print(f"Raw output shape: {audio.shape}")
        
        # Handle different output formats
        if len(audio.shape) == 3:  # [batch, channels, samples]
            audio_out = audio[0]  # Remove batch dimension
            if audio_out.shape[0] == 1:  # Mono
                audio_out = audio_out[0]
            else:  # Multi-channel - take first or mix down
                audio_out = audio_out[0]
        elif len(audio.shape) == 2:  # [batch, samples]
            audio_out = audio[0]
        else:  # [samples]
            audio_out = audio
            
        print(f"Final audio shape: {audio_out.shape}")
        print(f"Duration: {len(audio_out) / self.sample_rate:.2f} seconds")
        
        return audio_out, self.sample_rate

    def normalize_samples(self, samples):
        """Save audio to WAV file"""
        # Normalize and convert to int16
        if samples.dtype in [np.float32, np.float64]:
            # Clip to valid range
            samples_data = np.clip(samples, -1.0, 1.0)
            samples_int16 = (samples_data * 32767).astype(np.int16)
        else:
            samples_int16 = samples_data.astype(np.int16)
        
        return samples_int16

    @staticmethod
    def extract_snac_codes_from_tokens(token_ids, tokenizer):

        audio_codes = []
        current_timestep = []
        
        for token_id in token_ids:
            try:
                token_str = tokenizer.id_to_token(token_id)
                print(f"Token ID {token_id} -> '{token_str}'")
                if not token_str:
                    continue
                
                # SNAC format: <|audio_codes.layer.code|> or similar
                if 'audio_codes' in token_str:
                    # Extract layer and code from token
                    # Format variations: <|audio_codes.0.123|>, audio_codes.0.123, etc.
                    match = re.search(r'audio_codes\.(\d+)\.(\d+)', token_str)
                    if match:
                        layer = int(match.group(1))
                        code = int(match.group(2))
                        
                        # Ensure we have enough layers in current timestep
                        while len(current_timestep) <= layer:
                            current_timestep.append(None)
                        
                        current_timestep[layer] = code
                        print(f"SNAC: Layer {layer}, Code {code} from token '{token_str}'")
                        
                        # If this completes a timestep (layer 0 usually indicates new timestep)
                        if layer == 0 and len(current_timestep) > 1:
                            # Save the previous timestep if it was complete
                            if len(audio_codes) > 0 or any(c is not None for c in current_timestep[1:]):
                                audio_codes.append([c for c in current_timestep if c is not None])
                            current_timestep = [code]  # Start new timestep
                        
            except (IndexError, KeyError, ValueError) as e:
                print(f"Warning: Could not process SNAC token ID {token_id}: {e}")
                continue
        
        if current_timestep and any(c is not None for c in current_timestep):
            audio_codes.append([c for c in current_timestep if c is not None])
        
        return audio_codes


    @staticmethod
    def validate_hierarchical_structure(tokens, num_bands=3):
        """
        Validate that tokens follow SNAC's hierarchical structure
        
        Args:
            tokens (list): Token sequence to validate
            num_bands (int): Expected number of bands
            
        Returns:
            bool: True if structure is valid
        """
        tokens_per_tree = sum(2**i for i in range(num_bands))
        
        print(f"Validation:")
        print(f"  Tokens per tree: {tokens_per_tree}")
        print(f"  Total tokens: {len(tokens)}")
        print(f"  Complete trees: {len(tokens) // tokens_per_tree}")
        print(f"  Remainder: {len(tokens) % tokens_per_tree}")
        
        return len(tokens) % tokens_per_tree == 0


    