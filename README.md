# Dante GPT
A transformer-based language model trained on Dante Alighieri's Divina Commedia. The model is strongly inspired by Andrej Karpathy's nanoGPT, which is itself a simplified version of GPT-2.

<p align="center">
<img src="images/dante_robot.png" height="300">
</p>

# Models
### Model 1
Model 1 is pretty much identical to nanoGPT in the MakeMore series by Andrej Karpathy. It uses a simple lookup table as token embedding with `n_emb=384`, implemented with `nn.Embedding(vocab_size, n_emb)`. These `384`-dimensional token embedding vectors are added to `384`-dimensional positional embedding vectors, obtained with `nn.Embedding(context_size, n_emb)`. The resulting embedded vectors contain both positional and identity information. These are fed through `6` Transformer blocks. Each transformer block starts with a `LayerNorm`, then the input is fed through `MultiHeadSelfAttention` using `6` heads with latent dimension (i.e. the dimension of the keys, queries and values) of `d_head=n_emb/n_heads=64`. The results of the `6` heads are aggregated with a linear projection `nn.Linear(n_emb, n_emb)` and a `dropout` with rate `0.2` is applied. In this model we use `self-attention`, i.e. this is a `decoder-only` architecture. After the dropout, we use a `residual/skip connection`, go through another `LayerNorm` and through a one-hidden-layer `Feed-Forward Neural Network` with `ReLU` activation function, `4 n_emb` hidden units and dropout of `0.2`. Then this is fed through another `residual/skip connection`. This is all repeated `6` times. After the sixth block, we use one last `LayerNorm`, a `Linear` layer to project back onto a space of dimension `vocab_size` and this is fed through the `cross-entropy` loss.

<p align="center">
<img src="images/model1.png" height="300">
</p>