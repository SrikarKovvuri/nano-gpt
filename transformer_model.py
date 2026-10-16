import torch
import torch.nn as nn

Cless TransfromerLanguageModel(nn.Module):
    def __init__(self, vocab_size, block_size = 64, n-embd = 64):
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

        self.query = nn.linear(n_embd, head_size, bias = False) # (B, T, head_size)
        self.key = nn.linear(n_embd, head_size, bias = False)
        self.value = nn.linear(n_embd, head_size, bias = False)

        #create a mask
        # self.register buffer is used so we can move it to cuda if needed. torch.tril upper diagonal is zeroes. torch.ones is justa  block of ones.
        self.register_buffer('mask', torch.tril(torch.ones(block_size, block_size)))

    #forward mererely uses those layers and parameters to perfrom computations on the input
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        #everything above is (B, T, n_embd)

        #compute attention scores
        scores = 
`
        
