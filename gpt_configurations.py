import torch


class GPTConfig1:
    """Settings for Model1."""
    batch_size: int = 64
    context_size: int = 256
    n_emb: int = 384  # each head is 384//6 = 64 dimensional, which is standard
    num_layers: int = 6
    num_heads: int = 6
    dropout_prop: float = 0.2  # 20% of neurons are dropped out
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    vocabulary_size: int = 68
    vocabulary: tuple = ('\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F',
                         'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', 'a', 'b', 'c',
                         'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x',
                         'y', 'z', '~', 'à', 'è', 'é', 'ì', 'ï', 'ò', 'ó', 'ù')
    str_to_int: dict = {character: integer for integer, character in enumerate(vocabulary)}
    int_to_str = {integer: character for integer, character in enumerate(vocabulary)}


class GPTConfig2Small:
    """Settings for Model2, small version."""
    batch_size: int = 64
    context_size: int = 256
    n_emb: int = 192  # each head is 384//6 = 64 dimensional, which is standard
    num_layers: int = 4
    num_heads: int = 4
    dropout_prop: float = 0.2  # 20% of neurons are dropped out
    device: str = 'mps' if torch.backends.mps.is_available() else 'cpu'
    vocabulary_size: int = 68
    vocabulary: tuple = ('\n', ' ', '!', '"', "'", '(', ')', ',', '-', '.', ':', ';', '?', 'A', 'B', 'C', 'D', 'E', 'F',
                         'G', 'H', 'I', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Z', 'a', 'b', 'c',
                         'd', 'e', 'f', 'g', 'h', 'i', 'j', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'x',
                         'y', 'z', '~', 'à', 'è', 'é', 'ì', 'ï', 'ò', 'ó', 'ù')
    str_to_int: dict = {character: integer for integer, character in enumerate(vocabulary)}
    int_to_str = {integer: character for integer, character in enumerate(vocabulary)}
