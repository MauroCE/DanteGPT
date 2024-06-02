import torch
from dataclasses import dataclass
from gpt import Model1


@dataclass
class GPTConfig:
    batch_size: int = 64
    context_size: int = 256
    n_emb: int = 384  # each head is 384//6 = 64 dimensional, which is standard
    num_layers: int = 6
    num_heads: int = 6
    dropout_prop: float = 0.2  # 20% of neurons are dropped out
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    vocabulary_size: int = 68


class Chatbot:

    def __init__(self, model, config, path="models/model1.pth"):
        # Read data
        with open('data/commedia.txt', 'r', encoding='utf-8') as f:
            self.text = f.read()
        vocabulary = sorted(list(set(self.text)))
        vocab_size = len(vocabulary)
        # Model
        self.model = model(
            n_emb=config.n_emb,
            num_heads=config.n_head,
            context_size=config.block_size,
            dropout_prop=config.dropout,
            vocabulary_size=config.vocab_size,
            num_layers=config.n_layers)
        state_dict = torch.load(path)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(config.device)
        # Functions for converting to and from indices/strings
        str_to_int = {character: integer for integer, character in enumerate(vocabulary)}
        int_to_str = {integer: character for integer, character in enumerate(vocabulary)}
        self.str2int = lambda string: [str_to_int[character] for character in string]  # string --> list(int)
        self.int2str = lambda int_list: ''.join([int_to_str[integer] for integer in int_list])  # list(int) --> string

    def chat(self):
        """Starts a chat with the bot."""
        print("Hey, I'm DanteGPT. Message me, or type 'exit' to end the conversation.")
        while True:
            user_input = input("You: ")
            if user_input.lower() == 'exit':
                print("DanteGPT: Goodbye!")
                break

            context = torch.tensor(self.str2int(user_input), device=self.model.device).view(1, -1)
            indices = self.model.generate(context, device=self.model.device, max_new_tokens=500,
                                          context_size=self.model.context_size)[0].tolist()
            output_text = self.int2str(indices)
            print(f"DanteGPT: {output_text}")



