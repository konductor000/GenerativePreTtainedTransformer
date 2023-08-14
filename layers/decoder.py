import torch
from torch import nn
from .attention import MaskedMultiHeadedSelfAttention
from .feed_forward import FeedForward

class DecoderLayer(nn.Module):
    def __init__(self, embedding_size, number_of_heads,
                 extention_factor, additional_feed_forward_layers, attention_activation, dropout_rate):
        super(DecoderLayer, self).__init__()

        self.multi_headed_self_attention = MaskedMultiHeadedSelfAttention(embedding_size,
                                                                          number_of_heads,
                                                                          attention_activation)
        self.feed_forward = FeedForward(embedding_size, extention_factor, additional_feed_forward_layers)
        self.dropout = nn.Dropout(dropout_rate)

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embedding_size),
            nn.LayerNorm(embedding_size)
        ])

    def forward(self, x, mask):
        normalized_x = self.layer_norms[0](x)
        attention_output = self.multi_headed_self_attention(normalized_x, mask)
        residual_output = x + attention_output
        normalized_residual_output = self.layer_norms[1](residual_output)
        feed_forward_output = self.feed_forward(normalized_residual_output)

        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)
        else:
            feed_forward_output = feed_forward_output * (1 - self.dropout.p)

        return residual_output + feed_forward_output

class DecoderStack(nn.Module):
    def __init__(self, embedding_size, number_of_layers, number_of_heads,
                 extention_factor, additional_feed_forward_layers, attention_activation, dropout_rate, max_sequence_length):
        super(DecoderStack, self).__init__()

        self.decoder_layers = nn.ModuleList([
            DecoderLayer(embedding_size, number_of_heads, extention_factor, additional_feed_forward_layers, attention_activation, dropout_rate)
            for _ in range(number_of_layers)
        ])

    def forward(self, x, mask):
        for decoder_layer in self.decoder_layers:
            x = decoder_layer(x, mask)
        return x