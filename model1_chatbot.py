import torch
import pickle
from gpt import Model1, GPTConfig1, GPTConfig2Small, Model2


class Chatbot:

    def __init__(self, model, config, path="models/model1_tracking.pth"):
        # Read data
        # with open('data/commedia.txt', 'r', encoding='utf-8') as f:
        #     self.text = f.read()
        # vocabulary = sorted(list(set(self.text)))
        # Config
        self.config = config
        # Model
        self.model = model(self.config)
        self.model.load_state_dict(torch.load(path, map_location=self.config.device))
        self.model.eval()  # Set the model to evaluation mode
        # self.model.to(self.config.device)
        # Functions for converting to and from indices/strings
        self.str_to_int = {character: integer for integer, character in enumerate(config.vocabulary)}
        self.int_to_str = {integer: character for integer, character in enumerate(config.vocabulary)}
        self.str2int = lambda string: [self.str_to_int[character] for character in string]  # string --> list(int)
        self.int2str = lambda int_list: ''.join([self.int_to_str[integer] for integer in int_list])  # list(int) to str

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
    config = GPTConfig2Small()
    config.device = 'cpu'
    # make it smaller
    config.n_emb = 100
    cb = Chatbot(Model2, config, path="models_heroku/model2_smaller2.pth")
    cb.chat()
