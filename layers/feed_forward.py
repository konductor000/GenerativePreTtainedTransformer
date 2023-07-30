import torch


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_size, extention_factor):
        super().__init__()

        self.embedding_dimension = embedding_size
        self.feed_forward_dimension = extention_factor

        self.linear_1 = torch.nn.Linear(embedding_size, extention_factor)
        self.linear_2 = torch.nn.Linear(extention_factor, embedding_size)

    def forward(self, x):

        return self.linear_2(torch.relu(self.linear_1(x)))
    