from hashlib import sha256
import gguf
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

class FeatureExtractor:
    def __init__(self, gguf_path: str):
        # First, inspect the GGUF file to understand the actual model structure
        reader = gguf.GGUFReader(gguf_path)
        
        # Debug: Print tensor information to understand the model structure
        print("GGUF file tensor information:")
        tensor_info = {}
        for gg_tensor in reader.tensors:
            print(f"Tensor: {gg_tensor.name}, Shape: {gg_tensor.shape}, Type: {gg_tensor.tensor_type}")
            tensor_info[gg_tensor.name] = gg_tensor.shape
        
        # Analyze the structure to determine correct config values
        hidden_size = None
        vocab_size = None
        num_layers = 0
        
        # Extract hidden_size from norm weights or embedding weights
        if 'output_norm.weight' in tensor_info:
            hidden_size = tensor_info['output_norm.weight'][0]
        elif 'token_embd.weight' in tensor_info:
            vocab_size, hidden_size = tensor_info['token_embd.weight']
        
        # Count number of layers
        layer_indices = set()
        for name in tensor_info.keys():
            if name.startswith('blk.'):
                parts = name.split('.')
                if len(parts) >= 2:
                    try:
                        layer_idx = int(parts[1])
                        layer_indices.add(layer_idx)
                    except ValueError:
                        continue
        
        if layer_indices:
            num_layers = max(layer_indices) + 1
        
        print(f"\nInferred model structure:")
        print(f"Hidden size: {hidden_size}")
        print(f"Vocab size: {vocab_size}")
        print(f"Number of layers: {num_layers}")
        
        # Analyze attention tensor shapes to get correct dimensions
        num_attention_heads = None
        num_key_value_heads = None
        head_dim = None
        
        # Look at the first layer's attention weights
        q_shape = tensor_info.get('blk.0.attn_q.weight')
        k_shape = tensor_info.get('blk.0.attn_k.weight')
        v_shape = tensor_info.get('blk.0.attn_v.weight')
        o_shape = tensor_info.get('blk.0.attn_output.weight')
        
        if q_shape is not None and k_shape is not None and v_shape is not None and o_shape is not None:
            print(f"Attention tensor shapes:")
            print(f"  Q: {q_shape}")
            print(f"  K: {k_shape}")
            print(f"  V: {v_shape}")
            print(f"  O: {o_shape}")
            
            # For transformers, weights are stored as [out_features, in_features]
            # Q: [hidden_size, num_attention_heads * head_dim]
            # K, V: [hidden_size, num_key_value_heads * head_dim]
            # O: [num_attention_heads * head_dim, hidden_size]
            
            q_out_dim = q_shape[1]  # Output dimension of Q projection
            k_out_dim = k_shape[1]  # Output dimension of K projection
            v_out_dim = v_shape[1]  # Output dimension of V projection
            o_in_dim = o_shape[0]   # Input dimension of O projection
            
            print(f"  Q output dim: {q_out_dim}")
            print(f"  K output dim: {k_out_dim}")
            print(f"  V output dim: {v_out_dim}")
            print(f"  O input dim: {o_in_dim}")
            
            # Verify consistency
            assert k_out_dim == v_out_dim, f"K and V output dimensions must match: {k_out_dim} vs {v_out_dim}"
            assert q_out_dim == o_in_dim, f"Q output and O input dimensions must match: {q_out_dim} vs {o_in_dim}"
            
            # Calculate head dimensions
            # Find common divisors to determine head_dim
            #change to 64 for v0.1 model
            head_dim = 256
            num_attention_heads = q_out_dim // head_dim
            num_key_value_heads = k_out_dim // head_dim
            
            if head_dim is None:
                # Fallback: assume smallest reasonable head dimension
                head_dim = k_out_dim  # Use K/V dimension as head_dim
                num_key_value_heads = 1
                num_attention_heads = q_out_dim // head_dim
            
            print(f"Calculated attention parameters:")
            print(f"  head_dim: {head_dim}")
            print(f"  num_attention_heads: {num_attention_heads}")
            print(f"  num_key_value_heads: {num_key_value_heads}")
        
        # Fallback if we couldn't infer from tensors
        if num_attention_heads is None:
            num_attention_heads = 8
            num_key_value_heads = 1
            head_dim = hidden_size // num_attention_heads
            print(f"Using fallback attention config: heads={num_attention_heads}, kv_heads={num_key_value_heads}, head_dim={head_dim}")
        
        # Calculate intermediate size from FFN weights
        intermediate_size = None
        ffn_gate_shape = tensor_info.get('blk.0.ffn_gate.weight')
        if ffn_gate_shape is not None:  # Fixed: Check for None instead of truthiness
            intermediate_size = ffn_gate_shape[1]  # Output dimension of gate projection
        else:
            intermediate_size = hidden_size * 2  # Common default
        
        print(f"Intermediate size: {intermediate_size}")
        
        # Create config with correct parameters
        self.config = Qwen3Config(
            vocab_size=vocab_size or 6000,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            max_position_embeddings=4096,
            tie_word_embeddings=True,
        )
        
        print(f"\nFinal config:")
        print(f"vocab_size: {self.config.vocab_size}")
        print(f"hidden_size: {self.config.hidden_size}")
        print(f"intermediate_size: {self.config.intermediate_size}")
        print(f"num_hidden_layers: {self.config.num_hidden_layers}")
        print(f"num_attention_heads: {self.config.num_attention_heads}")
        print(f"num_key_value_heads: {self.config.num_key_value_heads}")
        print(f"head_dim: {self.config.head_dim}")
        
        self.model = Qwen3ForCausalLM(self.config)
        self.model.eval()
        
        actual_param_count = sum(p.numel() for p in self.model.parameters())
        print(f"\nModel parameter count: {actual_param_count}")
        
        # Load weights from gguf with better error handling
        sum_numel = 0
        for gg_tensor in reader.tensors:
            try:
                np_tensor = gguf.dequantize(gg_tensor.data, gg_tensor.tensor_type)
                # Make a writable copy to avoid the NumPy warning
                np_tensor = np_tensor.copy()
                pt_tensor = torch.from_numpy(np_tensor)
                sum_numel += pt_tensor.numel()
                assert pt_tensor.dtype == torch.float32

                if gg_tensor.name == 'output_norm.weight':
                    expected_shape = self.model.model.norm.weight.shape
                    actual_shape = pt_tensor.shape
                    print(f"Loading output_norm.weight: expected {expected_shape}, got {actual_shape}")
                    if expected_shape != actual_shape:
                        raise RuntimeError(f"Shape mismatch for output_norm.weight: expected {expected_shape}, got {actual_shape}")
                    self.model.model.norm.weight.data[:] = pt_tensor
                    
                elif gg_tensor.name == 'token_embd.weight':
                    expected_shape = self.model.model.embed_tokens.weight.shape
                    actual_shape = pt_tensor.shape
                    print(f"Loading token_embd.weight: expected {expected_shape}, got {actual_shape}")
                    if expected_shape != actual_shape:
                        raise RuntimeError(f"Shape mismatch for token_embd.weight: expected {expected_shape}, got {actual_shape}")
                    self.model.model.embed_tokens.weight.data[:] = pt_tensor
                    if self.config.tie_word_embeddings:
                        self.model.lm_head.weight.data[:] = pt_tensor
                    
                else:
                    parts = gg_tensor.name.split('.')
                    if len(parts) != 4:
                        print(f"Warning: Unexpected tensor name format: {gg_tensor.name}")
                        continue
                        
                    blk, i, k, weight = parts
                    if blk != 'blk' or weight != 'weight':
                        print(f"Warning: Unexpected tensor name format: {gg_tensor.name}")
                        continue
                        
                    try:
                        i = int(i)
                    except ValueError:
                        print(f"Warning: Invalid layer index in tensor name: {gg_tensor.name}")
                        continue
                    
                    # Load layer weights with shape checking
                    if k == 'attn_k':
                        target_weight = self.model.model.layers[i].self_attn.k_proj.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'attn_k_norm':
                        target_weight = self.model.model.layers[i].self_attn.k_norm.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'attn_norm':
                        target_weight = self.model.model.layers[i].input_layernorm.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'attn_output':
                        target_weight = self.model.model.layers[i].self_attn.o_proj.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'attn_q':
                        target_weight = self.model.model.layers[i].self_attn.q_proj.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'attn_q_norm':
                        target_weight = self.model.model.layers[i].self_attn.q_norm.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'attn_v':
                        target_weight = self.model.model.layers[i].self_attn.v_proj.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'ffn_down':
                        target_weight = self.model.model.layers[i].mlp.down_proj.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'ffn_gate':
                        target_weight = self.model.model.layers[i].mlp.gate_proj.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'ffn_norm':
                        target_weight = self.model.model.layers[i].post_attention_layernorm.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    elif k == 'ffn_up':
                        target_weight = self.model.model.layers[i].mlp.up_proj.weight
                        self._load_weight_with_check(pt_tensor, target_weight, gg_tensor.name)
                    else:
                        print(f"Warning: Unknown tensor name key: {k} in {gg_tensor.name}")
                        
            except Exception as e:
                print(f"Error loading tensor {gg_tensor.name}: {e}")
                raise

        print(f"\nTotal parameters loaded: {sum_numel}")
        model_param_count = sum(p.numel() for p in self.model.parameters())
        print(f"Model parameter count: {model_param_count}")

    def _load_weight_with_check(self, source_tensor, target_param, tensor_name):
        """Helper method to load weights with shape checking"""
        expected_shape = target_param.shape
        actual_shape = source_tensor.shape
        
        if expected_shape != actual_shape:
            print(f"Shape mismatch for {tensor_name}: expected {expected_shape}, got {actual_shape}")
            raise RuntimeError(f"Shape mismatch for {tensor_name}: expected {expected_shape}, got {actual_shape}")
        
        target_param.data[:] = source_tensor
        print(f"Successfully loaded {tensor_name}: {actual_shape}")

    def extract_features(self, input_ids: torch.LongTensor):
        """
        Run the model forward and extract hidden states as features.
        :param input_ids: Tensor of token ids, shape (batch_size, sequence_length)
        :return: hidden states tensor of shape (batch_size, seq_len, hidden_size)
        """
        with torch.no_grad():
            outputs = self.model.model(
                input_ids=input_ids,
                output_hidden_states=True,
                return_dict=True,
            )
            # Return last hidden state (features)
            return outputs.last_hidden_state
