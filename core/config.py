class GraphormerConfig:
    def __init__(self):
        self.num_atom_types = 132
        self.embed_dim = 768
        self.ffn_dim = 3072
        self.num_heads = 32
        self.num_layers = 6
        self.num_kernel = 128
        self.dropout = 0.1
        self.attn_dropout = 0.1
        self.act_dropout = 0.1