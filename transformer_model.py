import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size = 64, n_embd = 64):
        super().__init__()

        #lookup table for each character
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        #lookup table for each position
        self.position_embedding_table = nn.Embedding(block_size, n_embd)


    def forward(self, input_token_ids):
        B, T = input_token_ids.shape

        #get the embeddings for each token
        token_embeddings = self.token_embedding_table(input_token_ids) # (B, T, n_embd)

        #get the position embeddings for each position
        positions = torch.arange(T, device = input_token_ids.device)
        position_embeddings = self.position_embedding_table(positions) # (T, n_embd)

        #add the token and position embeddings
        x = token_embeddings + position_embeddings # (B, T, n_embd)

        #apply the transformer blocks #(B, T, n_embd)

class TransformerHead(nn.Module):
    #init always defines the learanble layers and parameters
    def __init__(self, n_embd, head_size, block_size):
        super().__init__()

        self.query = nn.Linear(n_embd, head_size, bias = False) # (B, T, head_size)
        self.key = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        #create a mask
        # self.register buffer is used so we can move it to cuda if needed. torch.tril upper diagonal is zeroes. torch.ones is justa  block of ones.
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    #forward mererely uses those layers and parameters to perfrom computations on the input
    def forward(self, x):
        B, T, C = x.shape
        
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        #everything above is (B, T, head_size)

        #compute attention scores
        #S = QK^T
        scores = q @ k.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) = (B, T, T)
        #scale the scores by the square root of the head size
        scores = scores * (k.shape[-1]) ** -0.5
        #mask the scores. We slice the mask to the same size as the ampunt of input tokens
        scores = scores.masked_fill(self.mask[:T, :T] == 0, float('-inf'))

        #softmax the scores
        weights = F.softmax(scores, dim=-1) # (B, T, T)
        #apply the weights to the values
        out = weights @ v # (B, T, T) @ (B, T, head_size) = (B, T, head_size)
        return out



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, block_size):
        super().__init__()

        heads_size = n_embd // num_heads

        for i in range(num_heads):
            self.heads.append(TransformerHead(n_embd, head_size, block_size))
        
        self.heads = torch.stack(self.heads)
    
    def forward(self, x):
        head_outputs = []
        for head in self.heads:
            head_outputs.append(head(x))

        return torch.cat(head_outputs, dim=-1)


class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )
    
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, n_embd, block_size, num_heads):
        super().__init__()

        #multi head attention
        self.attention MultiHeadAttention(num_heads, n_embd, block_size)

        #ffn
        self.feed_forward = FeedForward(n_embd)

        #layer_norm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        #layer norm -> attention -> residual
        #x_new = x_old + attention(layer_norm(x_old))
        x = x + self.attention(self.ln1(x))

        #layer norm -> ffn -> residual
        #x_new = x_old + ffn(layer_norm(x_old))
        x = x + self.feed_forward(self.ln2(x))

        return x





        
