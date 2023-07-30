import torch
import torch.nn.functional as F
import nltk
from nltk.translate import rouge_score


class TestModel:
    def __init__(self):
        self.metrics = {
            'perplexity': [], 
            'bleu': [], 
            'rouge': [], 
            'loss': [],
        }
        self.epochs = []

    def calculate_perplexity(predicted_logits, targets):
        # Step 1: Apply Softmax
        predicted_probs = F.softmax(predicted_logits, dim=-1)

        # Step 2: Compute Cross Entropy
        batch_size, sequence_length, vocab_size = predicted_probs.size()
        targets_flat = targets.view(-1)  # Flatten targets
        predicted_probs_flat = predicted_probs.view(batch_size * sequence_length, vocab_size)

        cross_entropy_loss = F.cross_entropy(predicted_probs_flat, targets_flat, reduction='sum')

        # Step 3: Average Cross Entropy
        average_cross_entropy = cross_entropy_loss / (batch_size * sequence_length)

        # Step 4: Compute Perplexity
        perplexity = torch.exp(average_cross_entropy)

        return perplexity.item()
    
    def compute_bleu(predicted_sentences, true_sentences, n=4):
        bleu_scores = []

        for pred_sent, true_sent in zip(predicted_sentences, true_sentences):
            pred_tokens = nltk.word_tokenize(pred_sent.lower())
            true_tokens = [nltk.word_tokenize(sent.lower()) for sent in true_sent]
            bleu_score = nltk.translate.bleu_score.sentence_bleu([true_tokens], pred_tokens, weights=[1/n] * n)
            bleu_scores.append(bleu_score)

        return sum(bleu_scores) / len(bleu_scores)
    
    def compute_rouge(predicted_sentences, true_sentences):
        rouge_scores = []

        for pred_sent, true_sent in zip(predicted_sentences, true_sentences):
            rouge_1_score = rouge_score.rouge_n([pred_sent], [true_sent], 1)
            rouge_2_score = rouge_score.rouge_n([pred_sent], [true_sent], 2)
            rouge_l_score = rouge_score.rouge_l([pred_sent], [true_sent])
            rouge_scores.append((rouge_1_score, rouge_2_score, rouge_l_score))

        rouge_scores = [sum(score) / len(score) for score in rouge_scores]
        
        return rouge_scores

    def compute_metrics(self, predicted, true_labels):
        pass

    def test_model(self, model, test_dataloader, num_classes):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        all_predicted = []
        all_true_labels = []

        model.eval()

        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)

            model_outputs = model(input_ids)

            