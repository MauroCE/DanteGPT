import torch


class GPTConfig1:
    """Settings for Model1."""
    def __init__(self):
        self.batch_size: int = 64
        self.context_size: int = 256
        self.n_emb: int = 384  # each head is 384//6 = 64 dimensional, which is standard
        self.num_layers: int = 6
        self.num_heads: int = 6
        self.dropout_prop: float = 0.2  # 20% of neurons are dropped out
        self.device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.vocabulary_size: int = 68
        self.vocabulary: tuple = ('\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D',
                                  'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X',
                                  'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'q',
                                  'r', 's', 't', 'u', 'v', 'x', 'y', 'z', '~', 'à', 'è', 'é', 'ì', 'ï', 'ò', 'ó', 'ù')
        self.str_to_int: dict = {character: integer for integer, character in enumerate(self.vocabulary)}
        self.int_to_str = {integer: character for integer, character in enumerate(self.vocabulary)}
        self.str2int = lambda string: [self.str_to_int[character] for character in string]  # string --> list(int)
        self.int2str = lambda int_list: ''.join([self.int_to_str[integer] for integer in int_list])  # list(int) to str


class GPTConfig2:
    """Settings for Model2"""
    def __init__(self):
        self.batch_size: int = 64
        self.context_size: int = 256
        self.n_emb: int = 100  # each head is 100//4 = 25 dimensional, which is smaller than standard
        self.num_layers: int = 4
        self.num_heads: int = 4
        self.dropout_prop: float = 0.2  # 20% of neurons are dropped out
        self.device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.vocabulary_size: int = 68
        self.vocabulary: tuple = ('\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D',
                                  'E', 'F', 'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X',
                                  'Z', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'q',
                                  'r', 's', 't', 'u', 'v', 'x', 'y', 'z', '~', 'à', 'è', 'é', 'ì', 'ï', 'ò', 'ó', 'ù')
        self.str_to_int: dict = {character: integer for integer, character in enumerate(self.vocabulary)}
        self.int_to_str = {integer: character for integer, character in enumerate(self.vocabulary)}
        self.str2int = lambda string: [self.str_to_int[character] for character in string]  # string --> list(int)
        self.int2str = lambda int_list: ''.join([self.int_to_str[integer] for integer in int_list])  # list(int) to str
