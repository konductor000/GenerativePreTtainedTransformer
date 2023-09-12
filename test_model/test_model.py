import torch
import evaluate
import spacy
from time import time


class TestModel:
    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        self.metrics = {
            'loss': 0,
            'bleu': 0,
            'rouge1': 0,
            "rouge2": 0,
            "rougeL": 0
        }

        spacy.cli.download("en_core_web_md")
        self.nlp = spacy.load("en_core_web_md")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.loss_function = torch.nn.CrossEntropyLoss().to(device)
        self.bleu_metric = evaluate.load("bleu")
        self.rouge_metric = evaluate.load("rouge")

    def convert_to_text_sequences(self, token_ids, attention_mask):
        batch_size, sequence_length = token_ids.size()
        text_sequences = []

        for i in range(batch_size):
            tokens = token_ids[i].tolist()
            tokens = [tokens[token_id] for token_id in range(len(tokens)) \
                       if attention_mask[i][token_id]]
            tokens = self.tokenizer.convert_ids_to_tokens(tokens)
            text = self.tokenizer.convert_tokens_to_string(tokens)
            text_sequences.append(text)

        return text_sequences

    def compute_metrics(self, predicted, true_labels, attention_mask):
        batch_size = len(predicted)
        predicted_tokens = torch.argmax(predicted, dim=-1)
        predicted_text = self.convert_to_text_sequences(predicted_tokens, attention_mask)
        true_text = self.convert_to_text_sequences(true_labels, attention_mask)

        rouge = self.rouge_metric.compute(predictions=predicted_text, references=true_text)

        loss = self.loss_function(predicted.transpose(1, 2), true_labels)
        bleu = self.bleu_metric.compute(predictions=predicted_text, references=true_text)['bleu']
        rouge1 = rouge['rouge1']
        rouge2 = rouge['rouge2']
        rougeL = rouge['rougeL']

        self.metrics['loss'] += float(loss)
        self.metrics['bleu'] += bleu
        self.metrics['rouge1'] += rouge1
        self.metrics['rouge2'] += rouge2
        self.metrics['rougeL'] += rougeL


    def test_model(self, model, test_dataloader):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for metric in self.metrics:
            self.metrics[metric] = 0

        for batch in test_dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    model_output, target = model(input_ids, attention_mask)

            self.compute_metrics(model_output, target, attention_mask)

        for metric in self.metrics:
            self.metrics[metric] /= len(test_dataloader)

        return self.metrics
        
    def test_simmilarity(self, model, test_dataloader, tokenizer, num_generate=5):
        simmilarity_accuracy = 0
        simmilarity_score = 0

        for batch in test_dataloader:
            text = batch['sentence']
            answers = batch['answers']

            beam_width = 4

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model.predict_next(text, tokenizer, num_generate, beam_width).split()[-num_generate:]
            output = ' '.join(output)
            
            sentence_doc = self.nlp(output)
            simmilarity_score += max([sentence_doc.similarity(self.nlp(answer)) for answer in answers])

            simmilarity_accuracy += True in [answer in output for answer in answers]

        return simmilarity_accuracy / len(test_dataloader), \
               simmilarity_score / len(test_dataloader)

