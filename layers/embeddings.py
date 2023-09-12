import torch

class TokenEmbedding(torch.nn.Module):
    def __init__(self, embedding_size, vocab_size):
        super().__init__()
        self.embedding_layer = torch.nn.Embedding(vocab_size, embedding_size)

    def forward(self, x):
        return self.embedding_layer(x)


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.encoding = self.generate_encoding(d_model, max_len)
        
    def generate_encoding(self, d_model, max_len):
        encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)

        return encoding.unsqueeze(0)
        
    def forward(self, x):
        self.encoding = self.encoding.to(x.device)
        x = x + self.encoding[:, :x.size(1)].detach()

        return x
    
    def to(self, device):
        self.encoding = self.encoding.to(device)
        return super(PositionalEncoding, self).to(device)
