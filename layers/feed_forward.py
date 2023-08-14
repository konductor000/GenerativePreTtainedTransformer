import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, embedding_size, extention_factor, additional_feed_forward_layers):
        super(FeedForward, self).__init__()

        feed_forward_dimension = embedding_size * extention_factor

        self.feed_forward_layers = nn.Sequential(
            nn.Linear(embedding_size, feed_forward_dimension),
            *[nn.Sequential(
                nn.Linear(feed_forward_dimension, feed_forward_dimension),
                nn.ReLU()
            ) for _ in range(additional_feed_forward_layers)],
            nn.Linear(feed_forward_dimension, embedding_size)
        )

    def forward(self, x):
        return self.feed_forward_layers(x)