import torch
import datasets
import re
from datasets import load_dataset
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
        dataloader = DataLoader(book_dataset, batch_size=self.batch_size, num_workers=16)

        return dataloader
    
    def skip_books(self, num_skip):
        for i in range(num_skip*10):
            next(self.dataset)


def create_test(max_sequence_length, batch_size, tokenizer, dataset_size=1024):
    data = load_dataset('monology/pile-uncopyrighted', split='test', streaming=True)
    
    texts = []
    cnt = dataset_size
    for datapoint in data:
        if len(datapoint['text'].split()) < 128:
            continue
        if cnt == 0:
            break
        cnt -= 1
        texts.append(datapoint['text'])

    test_dataset = BookDataset(texts, tokenizer, max_sequence_length)
    dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16)

    return dataloader


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
