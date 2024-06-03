import torch
from gpt import Model1, GPTConfig1


class Chatbot:

    def __init__(self, model, config, path="models/model1_tracking.pth"):
        # Read data
        with open('data/commedia.txt', 'r', encoding='utf-8') as f:
            self.text = f.read()
        vocabulary = sorted(list(set(self.text)))
        # Config
        self.config = config
        # Model
        self.model = model(self.config)
        state_dict = torch.load(path, map_location=self.config.device)
        self.model.load_state_dict(state_dict)
        self.model.load_state_dict(state_dict)
        self.model.eval()  # Set the model to evaluation mode
        self.model.to(self.config.device)
        # Functions for converting to and from indices/strings
        str_to_int = {character: integer for integer, character in enumerate(vocabulary)}
        int_to_str = {integer: character for integer, character in enumerate(vocabulary)}
        self.str2int = lambda string: [str_to_int[character] for character in string]  # string --> list(int)
        self.int2str = lambda int_list: ''.join([int_to_str[integer] for integer in int_list])  # list(int) --> string

    def chat(self):
        """Starts a chat with the bot."""
        print("Ciao, sono DanteGPT. Scrivi qualcosa, o digita 'exit' per finire la conversazione.")
        while True:
            user_input = input("Tu: ")
            if user_input.lower() == 'exit':
                print("DanteGPT: Addio!")
                break

            context = torch.tensor(self.str2int(user_input), device=self.config.device).view(1, -1)
            indices = self.model.generate(context, device=self.config.device, max_new_tokens=200,
                                          context_size=self.config.context_size)[0].tolist()
            output_text = self.int2str(indices)
            # Trim to the latest
            print(f"DanteGPT: {output_text[len(user_input):]}")


if __name__ == "__main__":
    cb = Chatbot(Model1, GPTConfig1)
    cb.chat()
