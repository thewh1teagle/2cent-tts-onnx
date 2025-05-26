"""
Convert the GGUF model 
from https://github.com/taylorchu/2cent-tts to transformers
"""

from hashlib import sha256
import gguf
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

class FeatureExtractor:
    def __init__(self, gguf_path: str):
        # Verify file hash
        with open(gguf_path, 'rb') as rb:
            h = sha256(rb.read()).hexdigest()
            assert h == '41a5ac9ddee430ba858b4c8225ebdaa100af18cf07ddce6e5483adb80b6cb74b', \
                f"Hash mismatch: {h}"

        PARAM_COUNT = 59_637_760
        self.config = Qwen3Config(
            vocab_size=6000,
            hidden_size=1024,
            intermediate_size=2048,
            num_hidden_layers=6,
            num_attention_heads=4,
            num_key_value_heads=1,
            head_dim=256,
            rms_norm_eps=1e-06,
            rope_theta=1000000.0,
            max_position_embeddings=4096,
            tie_word_embeddings=True,
        )
        self.model = Qwen3ForCausalLM(self.config)
        self.model.eval()
        assert sum(p.numel() for p in self.model.parameters()) == PARAM_COUNT

        # Load weights from gguf
        reader = gguf.GGUFReader(gguf_path)
        sum_numel = 0
        for gg_tensor in reader.tensors:
            np_tensor = gguf.dequantize(gg_tensor.data, gg_tensor.tensor_type)
            pt_tensor = torch.from_numpy(np_tensor)
            sum_numel += pt_tensor.numel()
            assert pt_tensor.dtype == torch.float32

            if gg_tensor.name == 'output_norm.weight':
                self.model.model.norm.weight.data[:] = pt_tensor
            elif gg_tensor.name == 'token_embd.weight':
                self.model.model.embed_tokens.weight.data[:] = pt_tensor
                self.model.lm_head.weight.data[:] = pt_tensor
            else:
                blk, i, k, weight = gg_tensor.name.split('.')
                assert blk == 'blk' and weight == 'weight'
                i = int(i)
                if k == 'attn_k':
                    self.model.model.layers[i].self_attn.k_proj.weight.data[:] = pt_tensor
                elif k == 'attn_k_norm':
                    self.model.model.layers[i].self_attn.k_norm.weight.data[:] = pt_tensor
                elif k == 'attn_norm':
                    self.model.model.layers[i].input_layernorm.weight.data[:] = pt_tensor
                elif k == 'attn_output':
                    self.model.model.layers[i].self_attn.o_proj.weight.data[:] = pt_tensor
                elif k == 'attn_q':
                    self.model.model.layers[i].self_attn.q_proj.weight.data[:] = pt_tensor
                elif k == 'attn_q_norm':
                    self.model.model.layers[i].self_attn.q_norm.weight.data[:] = pt_tensor
                elif k == 'attn_v':
                    self.model.model.layers[i].self_attn.v_proj.weight.data[:] = pt_tensor
                elif k == 'ffn_down':
                    self.model.model.layers[i].mlp.down_proj.weight.data[:] = pt_tensor
                elif k == 'ffn_gate':
                    self.model.model.layers[i].mlp.gate_proj.weight.data[:] = pt_tensor
                elif k == 'ffn_norm':
                    self.model.model.layers[i].post_attention_layernorm.weight.data[:] = pt_tensor
                elif k == 'ffn_up':
                    self.model.model.layers[i].mlp.up_proj.weight.data[:] = pt_tensor
                else:
                    raise ValueError(f"Unknown tensor name key: {k}")

        assert sum_numel == sum(p.numel() for p in self.model.parameters()), \
            "Loaded weights count mismatch."

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
