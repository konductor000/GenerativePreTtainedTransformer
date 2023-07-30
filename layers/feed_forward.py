import torch


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_size, extention_factor, additional_feed_forward_layers):
        super().__init__()

        self.embedding_dimension = embedding_size
        feed_forward_dimension = embedding_size * extention_factor

        self.linear_1 = torch.nn.Linear(embedding_size, feed_forward_dimension)

        self.additional_feed_forward_layers = []
        for _ in range(additional_feed_forward_layers):
            self.additional_feed_forward_layers.append(torch.nn.Linear(feed_forward_dimension, feed_forward_dimension))

        self.linear_last = torch.nn.Linear(feed_forward_dimension, embedding_size)

    def forward(self, x):
        x = torch.relu(self.linear_1(x))

        for layer in self.additional_feed_forward_layers:
            x = torch.relu(layer(x))

        return self.linear_last(x)
