import torch
import pickle
from gpt import Model1, estimate_loss, get_batch


if __name__ == "__main__":
    torch.manual_seed(333)  # reproducibility

    # Settings
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200
    n_embd = 384  # each head is 384//6 = 64 dimensional, which is standard
    n_layers = 6
    n_head = 6
    dropout = 0.2  # 20% of neurons are dropped out

    # Device (this works for mac silicons, use cuda for nvidia gpus)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("DEVICE: ", device)

    # Read divina commedia
    with open('data/commedia.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Compute vocabulary size for divina commedia, here we work on a character level
    vocabulary = sorted(list(set(text)))
    vocab_size = len(vocabulary)

    # Mappings from characters to integers and vice versa
    str_to_int = {character: integer for integer, character in enumerate(vocabulary)}
    int_to_str = {integer: character for integer, character in enumerate(vocabulary)}

    # Encoder and Decoder from string to indices and vice versa
    str2int = lambda string: [str_to_int[character] for character in string]  # string --> list(int)
    int2str = lambda int_list: ''.join([int_to_str[integer] for integer in int_list])  # list(int) --> string

    # Encode divina commedia
    data = torch.tensor(str2int(text), dtype=torch.long)

    # (Naive) Train-Test split
    n = int(0.9*len(data))
    train_data = data[:n]  # 90% training
    val_data = data[n:]    # 10% validation

    # Instantiate model and send params to device
    model = Model1(
        n_emb=n_embd,
        num_heads=n_head,
        context_size=block_size,
        dropout_prop=dropout,
        vocabulary_size=vocab_size,
        num_layers=n_layers)
    gpt = model.to(device)

    # Adam optimizer, as usual
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=1e-3)

    # Store losses
    losses = []

    # Training loop
    for iteration in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iteration % eval_interval == 0:
            losses = estimate_loss(
                gpt_model=model,
                training_data=train_data,
                dev=device,
                validation_data=val_data,
                eval_iters=eval_iters, context_size=block_size,
                batch_size=batch_size)
            print(f"step {iteration} train loss: {losses['train']:.4f} val loss: {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = get_batch(split="train",
                           training_data=train_data,
                           validation_data=val_data,
                           dev=device,
                           context_size=block_size, batch_size=batch_size)

        # evaluate the loss
        logits, loss = model(idx=xb, device=device, targets=yb)
        losses.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # Save model
    torch.save(model.state_dict(), "models/model1.pth")
    with open("losses/model1.pkl", "wb") as file:
        pickle.dump(losses, file)


