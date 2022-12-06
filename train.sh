CUDA_VISIBLE_DEVICES=0,1 python train_huggingface.py --csv_path ./data/adenoma_data.csv --label_col inflammatory --model_save_path ./models_zoo/inflammatory_output/

CUDA_VISIBLE_DEVICES=0,1 python train_huggingface.py --csv_path ./data/adenoma_data.csv --label_col hyperplastic --model_save_path ./models_zoo/hyperplastic_output/