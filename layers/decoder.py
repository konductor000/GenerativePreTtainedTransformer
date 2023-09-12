import torch
from .attention import MaskedMultiHeadedSelfAttention
from .feed_forward import FeedForward


class DecoderLayer(torch.nn.Module):
    def __init__(self, embedding_size, number_of_heads,
            extention_factor, additional_feed_forward_layers, dropout_rate, use_flash_att=False):
        super().__init__()

        self.embedding_size = embedding_size
        self.number_of_heads = number_of_heads
        self.dropout_rate = dropout_rate
        self.use_flash_att = use_flash_att

        self.multi_headed_self_attention = MaskedMultiHeadedSelfAttention(embedding_size,
                                        number_of_heads, use_flash_att=self.use_flash_att)
        self.feed_forward = FeedForward(embedding_size, extention_factor, additional_feed_forward_layers)
        self.dropout = torch.nn.Dropout(dropout_rate)

        self.layer_normalization_1 = torch.nn.LayerNorm(embedding_size)
        self.layer_normalization_2 = torch.nn.LayerNorm(embedding_size)

    def forward(self, x, mask):
        normalized_x = self.layer_normalization_1(x)
        attention_output = self.multi_headed_self_attention(normalized_x, mask)
        residual_output = x + attention_output
        normalized_residual_output = self.layer_normalization_2(residual_output)
        feed_forward_output = self.feed_forward(normalized_residual_output)

        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)

        return residual_output + feed_forward_output


class DecoderStack(torch.nn.Module):
    def __init__(self, embedding_size, number_of_layers, number_of_heads,
            extention_factor, additional_feed_forward_layers, dropout_rate, use_flash_att=False):
        super().__init__()

        self.encoder_layers = torch.nn.ModuleList(
            [DecoderLayer(embedding_size, number_of_heads, extention_factor, additional_feed_forward_layers,
                           dropout_rate, use_flash_att=use_flash_att)
              for _ in range(number_of_layers)])

    def forward(self, x, mask):
        decoder_outputs = x
        for decoder_layer in self.encoder_layers:
            decoder_outputs = decoder_layer(decoder_outputs, mask)

        return decoder_outputs