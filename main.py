import torch
from torch import nn
from torch.nn import functional as F


def scaled_dot_product_attention(q, k, v, masked=False):
    # q, k, v are of shape (b, s, d)
    attention = q.bmm(k.transpose(1, 2))
    attention = attention / q.size(-1) ** 0.5
    if masked:
        pass
    attention = F.softmax(attention, dim=-1)
    return attention.bmm(v)


def positional_embedding(seq_len, dim_model):
    pos = torch.arange(seq_len).reshape(1, -1, 1)
    dim = torch.arange(dim_model).reshape(1, 1, -1)
    phase = torch.div(pos, 1e4 ** (2 * dim // dim_model))
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase))


class AttentionHead(torch.nn.Module):

    def __init__(self, dim_model, dim_k, dim_v, masked=False):
        super(AttentionHead, self).__init__()
        self.linear_q = nn.Linear(dim_model, dim_k)
        self.linear_k = nn.Linear(dim_model, dim_k)
        self.linear_v = nn.Linear(dim_model, dim_v)

    def forward(self, q, k, v):
        return scaled_dot_product_attention(self.linear_q(q), self.linear_k(k), self.linear_v(v), masked=False)


class MultiHeadAttention(torch.nn.Module):

    def __init__(self, num_heads, dim_model, dim_k, masked=False):
        super().__init__()
        assert dim_model % num_heads == 0
        dim_v = dim_model // num_heads
        self.attention_heads = [AttentionHead(dim_model, dim_k, dim_v, masked) for _ in range(num_heads)]
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, q, k, v):
        output = [head(q, k, v) for head in self.attention_heads]
        return self.linear(torch.concat(output, dim=-1))


class Residual(nn.Module):

    def __init__(self, layers, norm=nn.LayerNorm, drop=nn.Dropout(0.1)):
        super().__init__()
        self.layers = layers
        self.norm = norm
        self.drop = drop

    def forward(self, x):
        return self.norm(x[0] + self.drop(self.layers(*x)))


class PositionWiseFFD(nn.Module):
    def __init__(self, dim_in, dim_hidden=2048, dim_out=512):
        super(PositionWiseFFD, self).__init__()
        self.linear1 = nn.Linear(dim_in, dim_hidden)
        self.linear2 = nn.Linear(dim_hidden, dim_out)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, num_heads=8, dim_model=512, dim_k=64, dim_ffd=2048):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = Residual(MultiHeadAttention(num_heads, dim_model, dim_k))
        self.position_wise_ffd = Residual(PositionWiseFFD(dim_in=dim_model, dim_hidden=dim_ffd, dim_out=dim_model))

    def forward(self, x):
        x = self.attention(x, x, x)
        return self.position_wise_ffd(x)


class TransformerEncoder(nn.Module):
    def __init__(self, num_layer=6, num_heads=8, dim_model=512, dim_k=64, dim_ffd=2048):
        super(TransformerEncoder, self).__init__()
        self.num_layer = num_layer
        self.encode_layers = nn.Sequential(
            *[TransformerEncoderLayer(num_heads, dim_model, dim_k, dim_ffd) for _ in range(num_layer)])

    def forward(self, x):
        seq_len, dim_model = x.shape[1:3]
        x += positional_embedding(seq_len, dim_model)
        return self.encode_layers(x)


class TransformerDecoderLayer(nn.Module):
    def __init__(self, num_heads=8, dim_model=512, dim_k=64, dim_ffd=2048):
        super(TransformerDecoderLayer, self).__init__()
        self.masked_attention = Residual(MultiHeadAttention(num_heads, dim_model, dim_k, masked=True))
        self.position_wise_ffd = Residual(PositionWiseFFD(dim_in=dim_model, dim_hidden=dim_ffd, dim_out=dim_model))
        self.attention = Residual(MultiHeadAttention(num_heads, dim_model, dim_k))

    def forward(self, x, encoder_f):
        x = self.masked_attention(x, x, x)
        x = self.attention(x, encoder_f, encoder_f)
        return self.position_wise_ffd(x)


class TransformerDecoder(nn.Module):
    def __init__(self, num_layer=6, num_heads=8, dim_model=512, dim_k=64, dim_ffd=2048):
        super(TransformerDecoder, self).__init__()
        self.num_layer = num_layer
        self.decode_layers = nn.ModuleList(
            [TransformerDecoderLayer(num_heads, dim_model, dim_k, dim_ffd) for _ in range(num_layer)])
        self.linear = nn.Linear(dim_model, dim_model)

    def forward(self, x, encoder_f):
        seq_len, dim_model = x.shape[1:3]
        x += positional_embedding(seq_len, dim_model)
        for layer in self.decode_layers:
            x = layer(x, encoder_f)
        return self.linear(x)


class Transformer(nn.Module):
    def __init__(self, encoder_layer=6, decoder_layer=6, num_heads=8, dim_model=512, dim_k=64, dim_ffd=2048):
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder(encoder_layer, num_heads, dim_model, dim_k, dim_ffd)
        self.decoder = TransformerDecoder(decoder_layer, num_heads, dim_model, dim_k, dim_ffd)

    def forward_train(self, input_embedding, output_embedding):
        encoder_f = self.encoder(input_embedding)
        return self.decoder(output_embedding, encoder_f)
    
    def forward_test(self, input_embedding):
        num_words = len(input_embedding)
        result = []
        output_embedding = '<start>'
        encoder_f = self.encoder(input_embedding)
        for _ in range(num_words):
            output_embedding = self.decoder(output_embedding, encoder_f)
            result.append(output_embedding)
        return result
