from sklearn import metrics
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from dataset import MyDataset, data_collator
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import re

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--label_col', default='hgd', type=str)
    p.add_argument("-c",'--csv_path', type=str, required=True)
    p.add_argument("-m", '--pretrained_model', type=str, required=True)
    p.add_argument("-o", "--output_csv", type=str, default=None)
    args = p.parse_args()

    return args


def map_fn(x):
    try:
        return re.sub('[\u3131-\u3163\uac00-\ud7a3?\n▣]', '', x)
    except:
        return x
    
def main(args):
    csv = pd.read_csv(args.csv_path)
    p = re.compile('[-=#)(:]')
    csv['text'] = csv['text'].map(lambda x: map_fn(x))
    csv['text'] = csv['text'].map(lambda x: re.sub(p, '', x[1:]))
    
    _, test = train_test_split(csv, test_size=0.1, random_state=1004, stratify=csv[args.label_col])
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)
    
    test_dataset = MyDataset(test, args.label_col)
    test_loader = DataLoader(test_dataset, 64, num_workers=8,
                                collate_fn=data_collator(tokenizer), pin_memory=True,
                                shuffle=False, drop_last=False)
    
    pipe = pipeline("text-classification",
                    model=roberta,
                    tokenizer=tokenizer,
                    device=0)
    
    labels = []
    outs = []
    for inputs in tqdm(test_loader):
        label = inputs.pop('labels')
        input_text = inputs['input_seq']
        out = pipe(input_text)
        out = [1 if value['label'] == "LABEL_1" else 0 for value in out]
        labels += list(label.numpy())
        outs += out
        
    print("accuracy : %.4f"%(metrics.accuracy_score(labels, outs)))
    print("F1-score : %.4f"%(metrics.f1_score(labels, outs)))
    print("Precision: %.4f"%(metrics.precision_score(labels, outs)))
    
    if args.output_csv:
        test[args.label_col+'_out'] = outs
        test.to_csv(args.output_csv, index=False)

if __name__ == '__main__':
    args = define_argparser()
    main(args)