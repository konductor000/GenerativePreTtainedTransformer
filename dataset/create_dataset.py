import torch
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, random_split

def create_data_loader(batch_size, max_sequence_size,
                       train_size=36000, test_size=2000):
    # Load the dataset from Hugging Face's datasets library
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')
    train_dataset, test_dataset = dataset['train']. \
        select(range(train_size)), dataset['test'].select(range(test_size))

    # Import the BERT base uncased tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=True, truncation=True,
                          max_length=max_sequence_size)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Convert the dataset to PyTorch tensors
    tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    tokenized_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])

    # Create the data loader
    train_dataloader = DataLoader(tokenized_train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader, tokenizer
