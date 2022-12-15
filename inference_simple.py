# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import re
import sys
# %%
model = AutoModelForSequenceClassification.from_pretrained('./models_zoo/adenoma/model_weights/', num_labels=2)
# %%
def remove_useless_text(x):
    """
    
    Remove korean and useless special tokens
    Args:
        x (String): text data

    Returns:
        x (String): Text data with unnecessary strings removed
        
    """
    x = re.sub('[\u3131-\u3163\uac00-\ud7a3?\n▣]', '', x)
    x = re.sub('[-=#)(:]', '', x[1:])
    return x

def select_model(model_type):
    """
    
    Select proper model

    Args:
        model_type (String):  H-> helicobacter, A-> Adenoma

    Returns:
        tokenizer: Roberta tokenizer for classificaiton
        model: Roberta model for classificaiton
        
    """
    tokenizer = AutoTokenizer.from_pretrained("./models_zoo/tokenizer/")
    
    if model_type == 'H':
        model = AutoModelForSequenceClassification.from_pretrained('./models_zoo/helicobacter/model_weights/', num_labels=2)
    elif model_type == 'A':
        model = AutoModelForSequenceClassification.from_pretrained('./models_zoo/adenoma/model_weights/', num_labels=2)
    else:
        sys.exit("Wrong Model Type!")
        
    return tokenizer, model

def main():

    model_type = sys.argv[1]
    text_path = sys.argv[2]
    
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
        text = remove_useless_text(text)
        
    tokenizer, model = select_model(model_type)
    
    
    pipe = pipeline("text-classification",
                    model=model,
                    tokenizer=tokenizer)
    
    out = pipe(text)[0]['label']
    
    print(out)
    if out == 'LABEL_0':
        sys.exit(0)
    else:
        sys.exit(1)
        
if __name__ == '__main__':
    main()
