import torch
import torch.nn.functional as F
from .embeddings import TokenEmbedding, PositionalEncoding
from .decoder import DecoderStack
from random import randint

class Transformer(torch.nn.Module):
    def __init__(
            self,
            number_of_tokens,
            max_sequence_length=512,
            embedding_size=512,
            number_of_layers=6,
            number_of_heads=4,
            extention_factor=4,
            additional_feed_forward_layers=0,
            dropout_rate=0.1,
            use_flash_att=False,
    ):
        super().__init__()

        self.max_sequence_length = max_sequence_length
        self.token_embedding = TokenEmbedding(embedding_size, number_of_tokens)
        self.positional_encoding = PositionalEncoding(embedding_size, max_sequence_length)
        self.layer_normalization = torch.nn.LayerNorm(embedding_size)

        self.decoder = DecoderStack(
            embedding_size=embedding_size,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            extention_factor=extention_factor,
            additional_feed_forward_layers=additional_feed_forward_layers,
            dropout_rate=dropout_rate,
            use_flash_att=use_flash_att,
        )

        self.lm_head = LMHead(embedding_size, number_of_tokens)

    def forward(self, x, mask):
        token_embeddings = self.token_embedding(x)
        positional_encoding = self.positional_encoding(token_embeddings)
        positional_encoding_normalized = self.layer_normalization(positional_encoding)
        decoder_outputs = self.decoder(positional_encoding_normalized, mask)
        lm_head_outputs = self.lm_head(decoder_outputs)

        return lm_head_outputs


class LMHead(torch.nn.Module):
    def __init__(self, embedding_size, number_of_tokens):
        super().__init__()
        self.embedding_size = embedding_size
        self.linear = torch.nn.Linear(embedding_size, number_of_tokens)

    def forward(self, x):
        linear_output = self.linear(x)

        return linear_output


class AutoregressiveWrapper(torch.nn.Module):
    def __init__(self, gpt_model):
        super().__init__()
        self.model = gpt_model
        self.max_sequence_length = self.model.max_sequence_length

    def forward(self, x, mask):
        inp, target = x[:, :-1], x[:, 1:]
        mask = mask[:, :-1]
        output = self.model(inp, mask)

        return output, target

    def next_token_probabilities(self, x, mask, temperature=1.0):
        logits = self.model(x, mask)[:, -1]
        logits = logits / temperature
        probabilities = torch.softmax(logits, dim=-1)

        return probabilities
    
    def predict_next(self, x, mask):
        inp, target = x[:, :-1], x[:, -1]
        mask = mask[:, :-1]
        output = self.next_token_probabilities(inp, mask)

        return output, target

    def count_parameters(self):
        num_parameters = sum(p.numel() for p in self.model.parameters())
        num_trainable_parameters = sum(param.numel() for param in self.model.parameters() if param.requires_grad)
        memory_allocated = num_parameters * 4
        memory_allocated /= (1024 ** 3) # total memory used in GB

        return num_parameters, num_trainable_parameters, memory_allocated

    def predict_next(model, input_text, tokenizer, num_predicted_tokens, beam_width):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_tokens = tokenizer.encode(input_text, return_tensors="pt")
        input_tokens = input_tokens[:, :-1].to(device)
        model.eval()

        # Initial beam consists of just the input sequence
        beams = [(input_tokens, 0.0)]

        for _ in range(num_predicted_tokens):
            new_beams = []

            for tokens, score in beams:
                mask = torch.ones_like(tokens)

                with torch.no_grad():
                    probabilities = model.next_token_probabilities(tokens, mask)
                
                # Get top `beam_width` candidates for each beam
                top_scores, top_indices = probabilities.topk(beam_width, dim=-1)
                
                for i in range(beam_width):
                    new_tokens = torch.cat((tokens, top_indices[:, i].unsqueeze(0)), dim=1)
                    new_score = score + top_scores[0, i].item()
                    new_beams.append((new_tokens, new_score))
                
            # Select the top `beam_width` beams
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:beam_width]
            
        # Choose the best beam
        best_tokens, _ = max(beams, key=lambda x: x[1])
        return tokenizer.decode(best_tokens[0])