import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, random_split

def create_data_loader(batch_size, max_sequence_size, tokenizer=None):
    # Load the dataset from Hugging Face's datasets library
    dataset = load_dataset('financial_phrasebank', 'sentences_50agree')['train']
    dataset = dataset.train_test_split(test_size=0.2)

    # Import the BERT base uncased tokenizer
    if tokenizer is None:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['sentence'], padding=True, truncation=True,
                          max_length=max_sequence_size)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Convert the dataset to PyTorch tensors
    tokenized_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create the data loader
    train_dataloader = DataLoader(tokenized_dataset['train'], batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_dataset['test'], batch_size=batch_size)

    return train_dataloader, test_dataloader, tokenizer