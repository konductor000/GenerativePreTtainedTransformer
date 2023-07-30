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

        attention_weights = torch.matmul(query, key.transpose(-2, -1))

        attention_weights = attention_weights / torch.sqrt(torch.tensor([self.head_size])).to(attention_weights.device)

        mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
        attention_weights = attention_weights.masked_fill(mask == 0, float('-inf'))

        attention_scores = self.softmax(attention_weights)

        return torch.bmm(attention_scores, value)
    

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
