from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import argparse
import re
import sys
from time import time

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("-t",'--text_path', type=str, required=True)
    p.add_argument("-m", '--pretrained_model', type=str, required=True)
    args = p.parse_args()

    return args

def remove_useless_text(x):
    x = re.sub('[\u3131-\u3163\uac00-\ud7a3?\n▣]', '', x)
    x = re.sub('[-=#)(:]', '', x[1:])
    return x

def main(args):

    with open(args.text_path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = remove_useless_text(text)
        
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    roberta = AutoModelForSequenceClassification.from_pretrained(args.pretrained_model, num_labels=2)
    
    pipe = pipeline("text-classification",
                    model=roberta,
                    tokenizer=tokenizer)
    
    out = pipe(text)[0]['label']
    if out == 'LABEL_0':
        sys.exit(0)
    else:
        sys.exit(1)
        
if __name__ == '__main__':
    args = define_argparser()
    main(args)
