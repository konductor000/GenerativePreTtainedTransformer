import torch
from torch import nn
from torch.nn import functional


class SelfAttention(nn.Module):
    def __init__(self, num_heads, embedding_size):
        super().__init__()

        self.embedding_size = embedding_size
        self.num_heads = num_heads

        # Linear layers to create QUERY, KEY, VALUE vectors
        self.toquery = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.tokey = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.tovalue = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        # Linear layer to compute answer after attention
        self.output_fc = nn.Linear(self.embedding_size, self.embedding_size)

    def forward(self, x):
        batch_size, sequence_len, embedding_size = x.size()
        head_size = self.embedding_size // self.num_heads

        assert (self.embedding_size % self.num_heads == 0), 'number of heads must be divisible by embedding size'

        # (batch_size, sequence_len, embedding_size) sized each
        # after divide embedding into num_heads and head_size
        # (batch_size, sequence_len, num_heads, head_size) sized each
        query = self.toquery(x).view(batch_size, sequence_len, self.num_heads, head_size)
        key = self.tokey(x).view(batch_size, sequence_len, self.num_heads, head_size)
        value = self.tovalue(x).view(batch_size, sequence_len, self.num_heads, head_size)

        # reshape vectors to produce multihead attention
        # num_heads are joined to batch size
        # (batch_size * num_heads, sequence_len, head_size) sized each
        query = query.transpose(1, 2).contiguous().view(batch_size * self.num_heads, sequence_len, head_size)
        key = key.transpose(1, 2).contiguous().view(batch_size * self.num_heads, sequence_len, head_size)
        value = value.transpose(1, 2).contiguous().view(batch_size * self.num_heads, sequence_len, head_size)

        # compute attention attention sotfmax((query * key.T) / embedding_size ** 0.5) * value
        dot_product = torch.bmm(query, key.transpose(1, 2)) / self.embedding_size ** 0.5

        # apply mask for text generation
        #indices = torch.triu_indices(sequence_len, sequence_len, offset=1)
        #dot_product[:, indices[0], indices[1]] = float('-inf')

        # take softmax of dot product
        dot_product = functional.softmax(dot_product, dim=2)

        attention = torch.bmm(dot_product, value).view(batch_size, self.num_heads, sequence_len, head_size)
        # restore input sizes (batch_size, sequence_len, embedding_size)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, sequence_len, self.embedding_size)

        # (batch_size, sequence_len, embedding_size)
        answer = self.output_fc(attention)

        return answer
    

class TransformerBlock(nn.Module):
    def __init__(self, num_heads, embedding_size):
        super().__init__()

        self.attention = SelfAttention(num_heads, embedding_size)

        self.normalization1 = nn.LayerNorm(embedding_size)
        self.normalization2 = nn.LayerNorm(embedding_size)

        expansion_factor = 4
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, embedding_size * expansion_factor),
            nn.ReLU(),
            nn.Linear(embedding_size * expansion_factor, embedding_size)
        )

    def forward(self, x):
        batch_size, sequence_len, embedding_size = x.size()

        # all shapes are (batch_size, sequence lenght, embedding size)
        skip_connection = x
        x = self.attention(x)
        x = self.normalization1(x + skip_connection)

        skip_connection = x
        x = self.feed_forward(x)
        x = self.normalization2(x + skip_connection)

        return x


class Transformer(nn.Module):
    def __init__(self, num_heads, embedding_size, num_blocks, max_sequence_len, vocab_size, output_size):
        super().__init__()

        self.embedding_size = embedding_size
        self.max_sequence_len = max_sequence_len
        self.embedding_layer = nn.Embedding(vocab_size, embedding_size)

        # apply for each position in sequence an embedding that is trainable
        self.positional_embeddings = nn.Embedding(max_sequence_len, embedding_size)

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(num_heads, embedding_size) for i in range(num_blocks)]
        )

        self.output_fc = nn.Linear(embedding_size, output_size)

    def forward(self, x):
        # input - tokenized words (batch_size, sequence length)
        batch_size, sequence_len = x.size()

        # create embeddiings and apply positional encoding
        # embeddings are (batch_size, sequence lenght, embedding size) sized
        embedded_sequence = self.embedding_layer(x)
        positions = torch.arange(self.max_sequence_len).expand(batch_size, -1)
        positions = self.positional_embeddings(positions)[:, :sequence_len, :]

        embedded_sequence += positions

        # apply all transormer blocks, same size
        transformer_output = self.transformer_blocks(embedded_sequence)

        # average pool embeddings to reduce sequence length dim

        predictions = self.output_fc(transformer_output.mean(dim=1))
        
        return functional.softmax(predictions, dim=1)

        # index_tensor = torch.arange(sequence_len).unsqueeze(0)
        # weights = torch.arange(1, sequence_len + 1).float().reciprocal()

        # cumulative_sum = torch.cumsum(transformer_output, dim=1)
        # pooled_state = cumulative_sum * weights.view(1, -1, 1)
        # pooled_state = pooled_state.sum(dim=1) / weights.sum()

        # predictions = functional.softmax(self.output_fc(pooled_state), dim=1)
        # predictions = predictions.unsqueeze(1).repeat(1, sequence_len, 1)

        # return predictions
