
import torch
import torch.nn.functional as F

from transformer_model import TransformerLanguageModel
from transformer_data import get_batch, vocab_size

def main():
    #hyperparameters
    learning_rate = 3e-4
    max_iters = 5000
    eval_iters = 100

    #choose device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    #init model
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        block_size=64,
        n_embd=64,
        num_layers=3,
        num_heads=4
    )

    #move model to device
    model.to(device)

    #create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    model.train()

    for step in range(max_iters):
        x,y = get_batch("train")

        #move tensors to device
        x = x.to(device)
        y = y.to(device)

        #forward pass
        logits, _ = model(x)

        #reshape combine batch and sequence number so we get (2048, 65)
        B, T, C = logits.shape

        logits = logits.reshape(B*T, C)

        y = y.reshape(B*T)

        #calculate loss
        loss = F.cross_entropy(logits, y) # scalar avg loss across 2048 tokens

        #reset grads to zero
        optimizer.zero_grad()

        loss.backward() # backpropagate the loss
        optimizer.step() # update the weights

        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss.item():.4f}")

    #validation step
    #telling pytorch not to track gradients so we can save memory
    model.eval()

    with torch.no_grad():
        losses = []
        for i in range(eval_iters):
            x,y = get_batch("val")
            x = x.to(device)
            y = y.to(device)

            logits = model(x)

            B, T, C = logits.shape

            logits = logits.reshape(B*T, C)

            y = y.reshape(B*T)

            loss = F.cross_entropy(logits, y)
            losses.append(loss.item())

        print(f"Validation Loss: {sum(losses)/len(losses):.4f}")

    #save model
    torch.save(model.state_dict(), "model.pth")
    print("Model saved to model.pth")


if __name__ == "__main__":
    main()
