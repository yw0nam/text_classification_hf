import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):

    def __init__(self, csv, label_col):
        self.csv = csv
        self.csv['text'] = self.csv['text'].map(lambda x: x[:min(len(x), 512)])
        self.label_col = label_col
    def __len__(self):
        return len(self.csv)
    
    def __getitem__(self, idx):
        input_seq = self.csv['text'].iloc[idx]
        label = self.csv[self.label_col].iloc[idx]

        return {
            'inputs': input_seq,
            'labels': label,
        }
        
class data_collator():

    def __init__(self, tokenizer, max_length=512, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text
        
    def __call__(self, samples):
        
        input_seq = [s['inputs'] for s in samples]
        labels = [s['labels'] for s in samples]
        
        encoding = self.tokenizer(
            input_seq,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )
        
        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.LongTensor(labels)
        }
        
        if self.with_text:
            return_value['input_seq'] = input_seq

        return return_value