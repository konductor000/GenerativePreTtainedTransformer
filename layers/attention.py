import torch
import torch.nn.functional as F

class MaskedSelfAttention(torch.nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.embedding_dimension = embedding_size
        self.head_size = head_size
        self.query_layer = torch.nn.Linear(embedding_size, head_size)
        self.key_layer = torch.nn.Linear(embedding_size, head_size)
        self.value_layer = torch.nn.Linear(embedding_size, head_size)

    def forward(self, x, padding_mask):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Compute the scaled dot-product attention scores
        attention_weights = torch.matmul(query, key.transpose(-2, -1)) / (self.head_size ** 0.5)

        causal_mask = torch.tril(torch.ones_like(attention_weights), diagonal=0)
        attention_weights = attention_weights - (1 - causal_mask) * 1e9  # Set to a large negative value instead of float('-inf')

        # Apply the mask to prevent attending to padded or invalid positions
        attention_weights = attention_weights.masked_fill(padding_mask.unsqueeze(1) == 0, -1e9)  # Set to a large negative value instead of float('-inf')

        # Compute the attention probabilities using softmax
        attention_scores = F.softmax(attention_weights, dim=-1)

        # Apply the attention scores to the value vectors
        output = torch.matmul(attention_scores, value)

        return output


class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    def __init__(self, embedding_size, number_of_heads, attention_activation):
        super().__init__()
        self.embedding_dimension = embedding_size
        self.head_size = embedding_size // number_of_heads
        self.number_of_heads = number_of_heads

        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_size, self.head_size) for _ in range(number_of_heads)])

        self.output_layer = torch.nn.Linear(number_of_heads * self.head_size, embedding_size)

    def forward(self, x, mask):
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        return self.output_layer(concatenated_self_attention_outputs)