import torch


class MaskedSelfAttention(torch.nn.Module):
    def __init__(self, embedding_size, head_size):
        super().__init__()
        self.embedding_dimension = embedding_size
        self.head_size = head_size
        self.query_layer = torch.nn.Linear(embedding_size, self.head_size)
        self.key_layer = torch.nn.Linear(embedding_size, self.head_size)
        self.value_layer = torch.nn.Linear(embedding_size, self.head_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask):
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Compute the scaled dot-product attention scores
        attention_weights = torch.matmul(query, key.transpose(-2, -1))
        attention_weights = attention_weights / torch.sqrt(torch.tensor([self.head_size], dtype=attention_weights.dtype, device=attention_weights.device))

        causal_mask = torch.tril(torch.ones_like(attention_weights), diagonal=-1)
        attention_weights = attention_weights.masked_fill(causal_mask == 0, float('-inf'))

        # Apply the mask to prevent attending to padded or invalid positions
        mask = mask.unsqueeze(1)  # Add a dimension to match attention_weights shape
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))

        # Compute the attention probabilities using softmax
        attention_scores = self.softmax(attention_weights)

        # Apply the attention scores to the value vectors
        output = torch.matmul(attention_scores, value)

        return output
    

class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    def __init__(self, embedding_size, number_of_heads):
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
