
import torch

from transformer_model import TransformerLanguageModel
from transformer_data import vocab_size

model = TransformerLanguageModel(
    vocab_size=vocab_size,
    block_size=64,
    n_embd=64,
    num_layers=3,
    num_heads=4
)

model.load_state_dict(
    torch.load("model.pth", map_location="cpu", weights_only=True)
)

model.eval()

#generate a sequence of 20 random token IDs
tokens = torch.randint(0, vocab_size, (1, 20))

with torch.inference_mode():

    #brute force: process all 20 tokens
    full_logits, _ = model(tokens)

    #prefill first 19 tokens
    _, cache = model(tokens[:, :19])

    #decode the 20th token using cached K/V
    cached_logits, cache = model(
        tokens[:, 19:20],
        kv_caches=cache
    )

    #compare predictions for the final token
    expected = full_logits[:, -1, :]
    actual = cached_logits[:, -1, :]

    print("Maximum absolute difference:",
          (expected - actual).abs().max().item())

    print("Numerically equivalent:",
          torch.allclose(expected, actual, atol=1e-5, rtol=1e-5))
