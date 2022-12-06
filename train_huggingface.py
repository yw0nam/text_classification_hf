import pandas as pd
import argparse
from dataset import *
from transformers import Trainer
from transformers import TrainingArguments
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.model_selection import train_test_split
import os, re

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--csv_path', required=True)
    p.add_argument('--label_col', default='hgd', type=str)
    p.add_argument('--gradient_accumulation_steps', type=int, default=2)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--batch_size_per_device', type=int, default=16)
    p.add_argument('--n_epochs', type=int, default=5)
    p.add_argument('--warmup_ratio', type=float, default=.2)
    p.add_argument('--max_length', type=int, default=512)
    p.add_argument('--load_weight', default=None)
    p.add_argument('--model_save_path', type=str, default='./models_zoo/hp_output/')

    config = p.parse_args()

    return config

def map_fn(x):
    try:
        return re.sub('[\u3131-\u3163\uac00-\ud7a3?\n▣]', '', x)
    except:
        return x
    
def main(config):
    
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    p = re.compile('[-=#)(:]')
    csv = pd.read_csv(config.csv_path)
    csv['text'] = csv['text'].map(lambda x: map_fn(x))
    csv['text'] = csv['text'].map(lambda x: re.sub(p, '', x[1:]))
    dev, _ = train_test_split(csv, test_size=0.1, random_state=1004, stratify=csv[config.label_col])
    train, val = train_test_split(dev, test_size=0.2, random_state=1004, stratify=dev[config.label_col])
    train_dataset = MyDataset(train ,config.label_col)
    val_dataset = MyDataset(val, config.label_col)
    
    print(
        '|train| =', len(train_dataset),
        '|valid| =', len(val_dataset),
    )

    total_batch_size = config.batch_size_per_device * torch.cuda.device_count()
    n_total_iterations = int(len(train_dataset) / total_batch_size * config.n_epochs)
    n_warmup_steps = int(n_total_iterations * config.warmup_ratio)
    print(
        '#total_iters =', n_total_iterations,
        '#warmup_iters =', n_warmup_steps,
    )
    
    training_args = TrainingArguments(
        output_dir=os.path.join(config.model_save_path, 'checkpoints'),
        num_train_epochs=config.n_epochs,
        per_device_train_batch_size=config.batch_size_per_device,
        per_device_eval_batch_size=config.batch_size_per_device,
        warmup_steps=n_warmup_steps,
        fp16=True,
        evaluation_strategy='epoch',
        logging_steps=n_total_iterations // 100,
        save_strategy ='epoch',
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator(tokenizer,
                                config.max_length,
                                with_text=False),
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    
    trainer.model.save_pretrained(os.path.join(config.model_save_path, 'model_weights'))

if __name__ == '__main__':
    config = define_argparser()
    main(config)