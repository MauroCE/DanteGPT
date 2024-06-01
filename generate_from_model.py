import torch
from gpt import GPT


if __name__ == "__main__":
    # Settings
    batch_size = 64
    block_size = 256
    max_iters = 1  # 5000
    eval_interval = 500
    learning_rate = 3e-4
    eval_iters = 200
    n_embd = 384  # each head is 384//6 = 64 dimensional, which is standard
    n_layers = 6
    n_head = 6
    dropout = 0.2  # 20% of neurons are dropped out
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print("DEVICE: ", device)

    # Read divina commedia
    with open('data/commedia.txt', 'r', encoding='utf-8') as f:
        text = f.read()

    # Compute vocabulary size for divina commedia, here we work on a character level
    vocabulary = sorted(list(set(text)))
    vocab_size = len(vocabulary)

    # Model
    model = GPT(
        n_emb=n_embd,
        num_heads=n_head,
        context_size=block_size,
        dropout_prop=dropout,
        vocabulary_size=vocab_size,
        num_layers=n_layers)

    PATH = "models/gpt_divina_commedia.pth"

    state_dict = torch.load(PATH)
    model.load_state_dict(state_dict)
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    # Generate text now
    str_to_int = {character: integer for integer, character in enumerate(vocabulary)}
    int_to_str = {integer: character for integer, character in enumerate(vocabulary)}
    str2int = lambda string: [str_to_int[character] for character in string]  # string --> list(int)
    int2str = lambda int_list: ''.join([int_to_str[integer] for integer in int_list])  # list(int) --> string
    with torch.no_grad():
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
        text = model.generate(context, device=device, max_new_tokens=500, context_size=block_size)[0].tolist()
        print(int2str(text))
