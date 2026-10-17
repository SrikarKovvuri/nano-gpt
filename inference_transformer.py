
import torch

from transformer_model import TransformerLanguageModel
from transformer_data import vocab_size

#choose device
device = "cuda" if torch.cuda.is_available() else "cpu"

#initialize model
model = TransformerLanguageModel(
    vocab_size=vocab_size,
    block_size=64,
    n_embd=64,
    num_layers=3,
    num_heads=4
)

#load trained weights
model.load_state_dict(
    torch.load("model.pth", map_location=device, weights_only=True)
)

#move model to device
model.to(device)

#evaluation mode
model.eval()

#ask user for prompt
prompt = input("Enter your prompt: ")

#run inference without tracking gradients
with torch.inference_mode():
    response = model.generate(prompt, max_new_tokens=200)

print("\nGenerated response:")
print(response)
