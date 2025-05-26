from transformers import BertTokenizer, BertForSequenceClassification
import os

# 创建模型目录
model_dir = 'D:\\CSstudy\\NLP\\Project\\models'
os.makedirs(model_dir, exist_ok=True)

# 下载模型和分词器
print("Downloading BERT model and tokenizer...")
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name, cache_dir=model_dir)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2, cache_dir=model_dir)

print("Model and tokenizer downloaded successfully!") 