import torch


class FeedForward(torch.nn.Module):
    def __init__(self, embedding_size, extention_factor, additional_feed_forward_layers):
        super().__init__()

        self.embedding_dimension = embedding_size
        feed_forward_dimension = embedding_size * extention_factor

        self.linear_1 = torch.nn.Linear(embedding_size, feed_forward_dimension)

        layers = []
        for _ in range(additional_feed_forward_layers):
            layers.append(torch.nn.Linear(feed_forward_dimension, feed_forward_dimension))
            layers.append(torch.nn.ReLU())

        self.additional_feed_forward_layers = torch.nn.Sequential(*layers)

        self.linear_last = torch.nn.Linear(feed_forward_dimension, embedding_size)

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = self.additional_feed_forward_layers(x)

        return self.linear_last(x)
