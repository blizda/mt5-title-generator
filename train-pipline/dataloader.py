from torch.utils.data import Dataset
from transformers import T5TokenizerFast
import csv

class TSVDataset(Dataset):
    def __init__(self, path, mtype='google/mt5-small'):
        self.tokenizer = T5TokenizerFast.from_pretrained(mtype)
        self.dataset = self.read_dataset_from_file(path)

    def __len__(self):
        return len(self.dataset)

    def read_dataset_from_file(self, path):
        dataset = []
        with open(path, 'r') as data:
            reader = csv.reader(data, delimiter='\t')
            for it in reader:
                it[0] = 'напиши заголовок: ' + it[0]
                dataset.append(it)
        return dataset

    def convert_to_features(self, example):
        src = self.tokenizer.batch_encode_plus([example[0]], max_length=1024,
                                          padding='max_length', truncation=True, return_tensors="pt")
        tgt = self.tokenizer.batch_encode_plus([example[1]], max_length=1024,
                                          padding='max_length', truncation=True, return_tensors="pt")
        return src, tgt

    def __getitem__(self, index):
        source, targets = self.convert_to_features(self.dataset[index])
        source_ids = source["input_ids"].squeeze()
        target_ids = targets["input_ids"].squeeze()
        src_mask = source["attention_mask"].squeeze()
        target_mask = targets["attention_mask"].squeeze()

        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "target_mask": target_mask}