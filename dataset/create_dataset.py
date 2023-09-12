import torch
import datasets
import re
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import DataLoader, Dataset, random_split
import random
import pandas as pd


"""
________________________________________________________________________
create generator with dataloaders with books
"""


def split_book_into_parts(books, max_sequence_length):
    parts = []
    for book_text in books:
        sentences = re.split(r'(?<=[.!?]) +', book_text)
        current_part = []

        for sentence in sentences:
            words = sentence.split()
            if len(words) >= max_sequence_length:
                parts.append(" ".join(list(words)))
                continue
            if len(current_part) + len(words) <= max_sequence_length:
                current_part += list(words)
            else:
                parts.append(" ".join(current_part))
                current_part = list(words)

        if len(current_part):
            parts.append(" ".join(current_part))
    
    return parts


class BookDataset(Dataset):
    def __init__(self, book_parts, tokenizer, max_length):
        self.encodings = tokenizer(book_parts, padding=True, truncation=True,
                                    max_length=max_length, return_tensors='pt')
    
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings.input_ids[idx],
            'attention_mask': self.encodings.attention_mask[idx]
        }


class BookGenerator:
    def __init__(self, max_sequence_length, batch_size, tokenizer):
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        self.dataset = iter(datasets.load_dataset("togethercomputer/RedPajama-Data-1T",
            "book", split="train", streaming=True))

    def next_book(self, num_books):
        books = []
        for i in range(num_books):
            book = next(self.dataset)['text']
            books.append(book)

        book_parts = split_book_into_parts(books, self.max_sequence_length)
        book_dataset = BookDataset(book_parts, self.tokenizer, self.max_sequence_length)
        dataloader = DataLoader(book_dataset, batch_size=self.batch_size)

        return dataloader
    
    def select_test(self, num_books):
        seed = 42
        random.seed(seed)
        all_samples = []

        for i in range(num_books):
            dataloader = self.next_book(1)
            number = random.randint(0, len(dataloader)-1)
            for idx, batch in enumerate(dataloader):
                if idx == number:
                    all_samples.append(batch)
                    break

        return all_samples
    
    def skip_books(self, num_skip):
        for i in range(num_skip*10):
            next(self.dataset)


"""
________________________________________________________________________
create dataloader wikitext test and train
"""


def wikitext_dataloader(batch_size, max_sequence_length, tokenizer,
                       train_size=36000, test_size=2000):
    # Load the dataset from Hugging Face's datasets library
    dataset = load_dataset('wikitext', 'wikitext-103-raw-v1')

    train_dataset, test_dataset = dataset['train']. \
        select(range(train_size)), dataset['test'].select(range(test_size))

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding=True, truncation=True,
                          max_length=max_sequence_length)

    tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
    tokenized_test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Convert the dataset to PyTorch tensors
    tokenized_train_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])
    tokenized_test_dataset.set_format('torch', columns=['input_ids', 'attention_mask'])

    # Create the data loader
    train_dataloader = DataLoader(tokenized_train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(tokenized_test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader


"""
________________________________________________________________________
create dataloader for testing
"""

def create_test_dataset(path):
    data = pd.read_csv(path)

    sentences = data['Sentence'].tolist()
    possible_answers = data['Possible Answers'].tolist()
    possible_answers = [line.split(', ') for line in possible_answers]
    result = [{'sentence': sentence, 'answers': answer} for sentence, answer in zip(sentences, possible_answers)]

    return result
