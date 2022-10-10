# Text Classification using huggingface

This is [huggingface](https://huggingface.co/) based text classification repository.
In this repository, i use pathology text. 

The used data comes from Samsung Medical Center. So, the data can't be public.


# Quickstart

## Dependencies

- Linux
- Python 3.7+
- PyTorch 1.10.1 or higher and CUDA

a. Create a conda virtual environment and activate it.

```shell
conda create -n pathology_predict python=3.7
conda activate pathology_predict
```
b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/)

c. Clone this repository.

```shell
git clone https://github.com/yw0nam/Adaspeech/
cd Adaspeech
```

d. Install requirments.

```shell
pip install -r requirements.txt
```

# Datasets

This repository only support table data.
If you want to use other format, you need to modify dataset.py for your data.

The data format is as follow

text | label | etc |
--- | --- | --- |
pathologoy_text_1 | 0 | ... |
pathologoy_text_2 | 1 | ... |
pathologoy_text_3 | 1| ... | 
... | ... | ... |

The data must include text, label columns for training and inferece.

In this repository, The number of pathology text is 53761 for adenoma and 156452 for helicobacter respectively.

The ratio for dev and test is 9:1.

# Train the model

Training the model using below code.

```
CUDA_VISIBLE_DEVICES=0,1 python train_huggingface.py --csv_path your_csv_path -m --model_save_path ./models_zoo/your_model_out_path
```

Check argument in train_huggingface.py for training detail 

# Inference


Here is [pretrained_model](https://drive.google.com/drive/folders/1HVrk0DlN6PN8wFEA2HpnT1ej4JF6b6-R?usp=sharing) for adenoma and helicobacter.

Try this

```
CUDA_VISIBLE_DEVICES=0 python inference.py -c your_csv_path -m ./models_zoo/your_pretrained_model -d gpu -o ./output.csv
```

If you using only cpu,

```
python inference.py -c your_csv_path -m ./models_zoo/your_pretrained_model -d cpu -o ./output.csv
```

# Result

Here is model performance for test set.

Dataset | Accuracy |F1-score | Precision |
--- |  --- | --- | --- |
Helicobacter | 0.9959 | 0.9937 | 0.9912 |
Adenoma | 0.9996 |  0.9997 | 1.0000 |
