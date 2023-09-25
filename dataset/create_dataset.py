import torch
import datasets
import re
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset, random_split
import random
import pandas as pd
from time import time
from transformers import BertTokenizer, AutoTokenizer


"""
________________________________________________________________________
create generator with dataloaders with books
"""


class SimpleDataset(Dataset):
    def __init__(self, book_parts, tokenizer, max_length):
        self.encodings = tokenizer(book_parts, truncation=True, padding=True,
                                    max_length=max_length, return_tensors='pt')


    def __len__(self):
        return len(self.encodings.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings.input_ids[idx],
            'attention_mask': self.encodings.attention_mask[idx]
        }


class PileGenerator:
    def __init__(self, max_sequence_length, batch_size, tokenizer, 
                 num_workers, test_size=4096, num_skip=0):
        self.num_workers = num_workers
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        self.dataset = iter(datasets.load_dataset("stanford-crfm/DSIR-filtered-pile-50M",
                                                   split="train", streaming=True))
        self.test_dataloader =  self.next_loader(test_size)
        self.skip_texts(num_skip)

    def next_loader(self, num_texts):
        texts = []
        for i in range(num_texts):
            book = next(self.dataset)['contents']
            texts.append(book)

        book_dataset = SimpleDataset(texts, self.tokenizer, self.max_sequence_length)
        dataloader = DataLoader(book_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

        return dataloader

    def skip_texts(self, num_skip):
        for i in range(num_skip):
            next(self.dataset)


class StoriesGenerator:
    def __init__(self, max_sequence_length, batch_size, tokenizer, 
                 num_workers, test_size):
        self.num_workers = num_workers
        self.max_sequence_length = max_sequence_length
        self.batch_size = batch_size
        self.tokenizer = tokenizer

        self.dataset = iter(datasets.load_dataset("roneneldan/TinyStories",
                                                   split="train", streaming=True))
        
        test_dataset =  iter(datasets.load_dataset("roneneldan/TinyStories", 
                                              split="validation", streaming=True))
        
        self.test_dataloader = self.next_loader(test_size, test_dataset)

    def next_loader(self, num_texts, dataset=None):
        dataset = dataset if dataset else self.dataset
        texts = []
        for i in range(num_texts):
            book = next(dataset)['text']
            texts.append(book)

        dataset = SimpleDataset(texts, self.tokenizer, self.max_sequence_length)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, num_workers=self.num_workers)

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
