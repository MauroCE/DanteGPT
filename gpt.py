import torch
import torch.nn as nn
from torch.nn import functional as F


def get_batch(split, training_data, validation_data, dev, context_size, batch_size):
    """Generates batch of data of inputs `x` and targets `y`."""
    dataset = training_data if split == "train" else validation_data
    # Sample integers from [0, n-block_size], representing off-sets, one for each batch
    ix = torch.randint(len(dataset) - context_size, (batch_size, ))
    # Grab context and target
    _context = torch.stack([dataset[i:i+context_size] for i in ix])  # (batch_size, block_size)
    _targets = torch.stack([dataset[i+1:i+context_size+1] for i in ix])  # (batch_size, block_size)
    _context, _targets = _context.to(dev), _targets.to(dev)
    return _context, _targets


@torch.no_grad()
def estimate_loss(gpt_model, training_data, validation_data, dev, eval_iters, context_size, batch_size):
    out = {}
    gpt_model.eval()
    for split in ['train', 'val']:
        _losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split=split, training_data=training_data, validation_data=validation_data,
                             dev=dev, context_size=context_size, batch_size=batch_size)
            _logits, _loss = gpt_model(idx=X, targets=Y, device=dev)
            _losses[k] = _loss.item()
        out[split] = _losses.mean()
    gpt_model.train()
    return out


class Head(nn.Module):
    """One head of self-attention."""
    def __init__(self, head_latent_dim,  n_emb, context_size, dropout_prop):
        super().__init__()
        self.key = nn.Linear(n_emb, head_latent_dim, bias=False)
        self.query = nn.Linear(n_emb, head_latent_dim, bias=False)
        self.value = nn.Linear(n_emb, head_latent_dim, bias=False)
        # Basically, using register buffer it is not treated as a parameter
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        # randomly prevent some of the nodes from communicating with a dropout
        self.dropout = nn.Dropout(dropout_prop)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)     # (B, T, C)
        q = self.query(x)   # (B, T, C)
        # Compute attention scores ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B, T, C) @ (B, C, T) = (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # Weighted aggregation of the values
        v = self.value(x)  # (B, T, C)
        out = wei @ v      # (B, T, T) @ (B, T, C) = (B, T, C)
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads fo self-attention, in parallel."""
    def __init__(self, num_heads, head_latent_dim, n_emb, context_size, dropout_prop):
        super().__init__()
        self.heads = nn.ModuleList([
            Head(head_latent_dim,  n_emb, context_size, dropout_prop) for _ in range(num_heads)
        ])
        self.proj = nn.Linear(n_emb, n_emb)
        self.dropout = nn.Dropout(dropout_prop)  # add a dropout typically added right before the residual connection

    def forward(self, x):
        # remember the output of each head is (B, T, C) so here we are concatenating the output on the final dimension
        # thus obtaining (B, T, num_heads*C)
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # linear projection of the output
        out = self.proj(out)
        # dropout before residual/skip connection
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """A simple linear layer followed by a non-linearity"""
    def __init__(self, n_emb, dropout_prop):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_emb, 4 * n_emb),  # see AIAUN paper
            nn.ReLU(),
            nn.Linear(4 * n_emb, n_emb),
            # add a dropout typically added right before the residual connection.
            nn.Dropout(dropout_prop)
        )
        # to understand why 4*n_embd see section 3.3 "Position-wise Feed-Forward Networks" in the
        # "Attention is All You Need" paper. There n_embd=512 and dff=2048

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block: communication followed by computation."""
    def __init__(self, n_emb, num_heads, context_size, dropout_prop):
        """Here n_embd is the embedding dimension and n_head is the number of heads."""
        super().__init__()
        head_latent_dim = n_emb // num_heads
        self.sa = MultiHeadAttention(num_heads, head_latent_dim, n_emb, context_size, dropout_prop)
        self.ffwd = FeedForward(n_emb, dropout_prop)
        self.ln1 = nn.LayerNorm(n_emb)  # per-token transformation that normalizes the features
        self.ln2 = nn.LayerNorm(n_emb)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # layer norm + self-attention + residual connection
        x = x + self.ffwd(self.ln2(x))  # layer norm + feed-forward + residual connection
        return x


class Model1(nn.Module):
    """GPT2-like model, almost identical to Andrej Karpathy's MakeMore series."""

    def __init__(self, n_emb, num_heads, context_size, dropout_prop, vocabulary_size, num_layers):
        """GPT model similar to ChatGPT and inspired by Andrej Karpathy's MakeMore series."""
        super().__init__()
        # Tokens read off the logits for the next token from a lookup table
        # Token embedding table has size (vocab_size, vocab_size)
        # The way it works is that the input, say 24 (the first one in xb above) will take the 24th row of this
        # embedding table.
        self.token_embedding_table = nn.Embedding(vocabulary_size, n_emb)
        # We now also encode the position. Each position from 0 to block_size-1 will have a corresponding embedding
        self.position_embedding_table = nn.Embedding(context_size, n_emb)
        # Transformer
        self.blocks = nn.Sequential(*[
            Block(n_emb, num_heads, context_size, dropout_prop) for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(n_emb)  # there should always be a layer norm at the end of the transformer
        self.lm_head = nn.Linear(n_emb, vocabulary_size)

    def forward(self, idx, device, targets=None):
        """Forward pass. Takes `idx` and `targets` which are both `(B, T)` tensors of integers.
        Here `B` is the batch_size and `T` should be the block/context length."""
        B, T = idx.shape
        # PyTorch will grab the row corresponding to the indices provided and return logits in
        # the shape (batch, time, channel). Here batch=4, time=8, channel=65 (vocab size)
        # The logits here are like the scores for the next token in the sequence
        tok_emb = self.token_embedding_table(idx)  # (B, T, C=embedding_dimension), these re token embeddings now.
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T, C)
        x = tok_emb + pos_emb  # (B, T, C)
        x = self.blocks(x)  # (B, T, C)
        _logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            _loss = None
        else:
            B, T, C = _logits.shape
            _logits = _logits.view(B*T, C)
            targets = targets.view(B*T)
            _loss = F.cross_entropy(_logits, targets)
        return _logits, _loss

    def generate(self, idx, max_new_tokens, context_size, device):
        """Here `idx` is the current context of tokens in some batch, so it is `(B, T)`. This function will continue
        the generation one by one, for both the B and T dimensions. It keeps doing this until max_new_tokens."""
        for _ in range(max_new_tokens):
            # We need to make sure that the idx that we feed into the model is the same size as the context
            idx_cond = idx[:, -context_size:]  # (B, T) --> (B, block_size)
            _logits, _loss = self(idx_cond, device=device)   # Get the predictions (calls forward(idx, targets=None))
            _logits = _logits[:, -1, :]  # (B, T, C) --> (B, C) we focus only on the last "time step"
            probs = F.softmax(_logits, dim=-1)  # Use Softmax to get probabilities. (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample using the probabilities (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # append the sampled index to the running sequence (B, T+1)
        return idx


