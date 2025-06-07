import numpy as np
import onnxruntime as ort
import json
from typing import List, Dict, Any, Optional, Tuple

class SNACAudioDecoder:
    def __init__(self, tokenizer_path: str, model_path: str):
        """Initialize the SNAC Audio Decoder"""
        self.tokenizer_path = tokenizer_path
        self.model_path = model_path
        
        # Load tokenizer
        with open(tokenizer_path, 'r') as f:
            self.tokenizer = json.load(f)
        
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(model_path)
        
        # Get model input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def split_audio_codes_to_bands(self, audio_codes: List[int], num_bands: int = 3) -> Dict[str, np.ndarray]:
        """Split audio codes into SNAC hierarchical bands using depth-first traversal"""
        if num_bands == 3:
            # For 3 bands: 7 tokens per complete tree (1 + 2 + 4 = 7)
            tokens_per_tree = 7
            bands = {0: [], 1: [], 2: []}
            
            # Process tokens in groups of 7
            for i in range(0, len(audio_codes), tokens_per_tree):
                tree_tokens = audio_codes[i:i + tokens_per_tree]
                
                if len(tree_tokens) >= 1:
                    bands[0].append(tree_tokens[0])  # Root (Band 0)
                if len(tree_tokens) >= 3:
                    bands[1].extend([tree_tokens[1], tree_tokens[2]])  # Level 1 (Band 1)
                if len(tree_tokens) >= 7:
                    bands[2].extend(tree_tokens[3:7])  # Level 2 (Band 2)
            
            # Convert to numpy arrays with proper shape
            result = {}
            for band_idx in range(num_bands):
                if bands[band_idx]:
                    result[f'audio_codes.{band_idx}'] = np.array(bands[band_idx], dtype=np.int64).reshape(1, -1)
                else:
                    # If band is empty, create a minimal array
                    result[f'audio_codes.{band_idx}'] = np.array([[0]], dtype=np.int64)
                    
            return result
        else:
            # Fallback: split evenly across bands
            chunk_size = len(audio_codes) // num_bands
            result = {}
            
            for band_idx in range(num_bands):
                start_idx = band_idx * chunk_size
                if band_idx == num_bands - 1:
                    end_idx = len(audio_codes)
                else:
                    end_idx = start_idx + chunk_size
                
                band_codes = audio_codes[start_idx:end_idx]
                if band_codes:
                    result[f'audio_codes.{band_idx}'] = np.array(band_codes, dtype=np.int64).reshape(1, -1)
                else:
                    result[f'audio_codes.{band_idx}'] = np.array([[0]], dtype=np.int64)
            
            return result

    def test_strategy_6(self, tokens: List[int]) -> Dict[str, Any]:
        """
        Test Strategy 6: Original tokens with SNAC split
        Returns a simple pass/fail result with details
        """
        try:
            # Use SNAC split with original tokens
            band_inputs = self.split_audio_codes_to_bands(tokens, num_bands=3)
            
            # Try inference
            audio_output = None
            error_msg = None
            
            # Check if we have the right number of inputs
            if len(band_inputs) != len(self.input_names):
                return {
                    "strategy": "original_snac_3band",
                    "status": "FAIL",
                    "error": f"Input count mismatch: model expects {len(self.input_names)} inputs, got {len(band_inputs)}",
                    "audio": None,
                    "details": {
                        "input_shapes": {key: value.shape for key, value in band_inputs.items()},
                        "expected_inputs": len(self.input_names),
                        "provided_inputs": len(band_inputs)
                    }
                }
            
            try:
                # Try with int64 first
                outputs = self.session.run(self.output_names, band_inputs)
                audio_output = outputs[0]
                
            except Exception as e:
                # Try with float32 conversion
                try:
                    band_inputs_float = {}
                    for key, value in band_inputs.items():
                        band_inputs_float[key] = value.astype(np.float32)
                    
                    outputs = self.session.run(self.output_names, band_inputs_float)
                    audio_output = outputs[0]
                    
                except Exception as e2:
                    return {
                        "strategy": "original_snac_3band",
                        "status": "FAIL",
                        "error": f"Int64 error: {str(e)}, Float32 error: {str(e2)}",
                        "audio": None,
                        "details": {
                            "input_shapes": {key: value.shape for key, value in band_inputs.items()},
                            "attempted_dtypes": ["int64", "float32"]
                        }
                    }
            
            # Check if we got valid audio output
            if audio_output is not None and audio_output.size > 0:
                # Basic quality check
                audio_flat = audio_output.flatten()
                non_zero_ratio = np.count_nonzero(audio_flat) / len(audio_flat)
                mean_amplitude = np.mean(np.abs(audio_flat))
                
                # Simple pass criteria: not all zeros and reasonable amplitude
                if non_zero_ratio > 0.01 and mean_amplitude > 1e-6:
                    return {
                        "strategy": "original_snac_3band",
                        "status": "PASS",
                        "error": None,
                        "audio": audio_output,
                        "details": {
                            "audio_shape": audio_output.shape,
                            "non_zero_ratio": float(non_zero_ratio),
                            "mean_amplitude": float(mean_amplitude),
                            "input_shapes": {key: value.shape for key, value in band_inputs.items()}
                        }
                    }
                else:
                    return {
                        "strategy": "original_snac_3band",
                        "status": "FAIL",
                        "error": "Audio output appears to be empty or near-zero",
                        "audio": audio_output,
                        "details": {
                            "audio_shape": audio_output.shape,
                            "non_zero_ratio": float(non_zero_ratio),
                            "mean_amplitude": float(mean_amplitude)
                        }
                    }
            else:
                return {
                    "strategy": "original_snac_3band",
                    "status": "FAIL",
                    "error": "No audio output received from model",
                    "audio": None,
                    "details": {
                        "output_received": audio_output is not None,
                        "output_shape": audio_output.shape if audio_output is not None else None
                    }
                }
                
        except Exception as e:
            return {
                "strategy": "original_snac_3band",
                "status": "FAIL",
                "error": f"Unexpected error: {str(e)}",
                "audio": None,
                "details": {
                    "exception_type": type(e).__name__
                }
            }

def test_snac_strategy_6(tokens: List[int], tokenizer_path: str, model_path: str) -> bool:
    """
    Simple function to test Strategy 6 and return True/False
    
    Args:
        tokens: List of token integers
        tokenizer_path: Path to tokenizer.json
        model_path: Path to decoder_model.onnx
        
    Returns:
        bool: True if PASS, False if FAIL
    """
    try:
        decoder = SNACAudioDecoder(tokenizer_path, model_path)
        result = decoder.test_strategy_6(tokens)
        
        print(f"Strategy 6 Status: {result['status']}")
        if result['error']:
            print(f"Error: {result['error']}")
        if result.get('details'):
            print(f"Details: {result['details']}")
            
        return result['status'] == "PASS"
        
    except Exception as e:
        print(f"Failed to initialize decoder: {str(e)}")
        return False

def test_snac_strategy_6_detailed(tokens: List[int], tokenizer_path: str, model_path: str) -> Dict[str, Any]:
    """
    Detailed function to test Strategy 6 and return full result
    
    Args:
        tokens: List of token integers
        tokenizer_path: Path to tokenizer.json
        model_path: Path to decoder_model.onnx
        
    Returns:
        Dict: Full result dictionary with status, audio, and details
    """
    try:
        decoder = SNACAudioDecoder(tokenizer_path, model_path)
        return decoder.test_strategy_6(tokens)
        
    except Exception as e:
        return {
            "strategy": "original_snac_3band",
            "status": "FAIL",
            "error": f"Failed to initialize decoder: {str(e)}",
            "audio": None,
            "details": {"initialization_error": True}
        }