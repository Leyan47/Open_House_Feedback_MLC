# # Load the package
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os
import pandas as pd
import numpy as np

# Load the trained model
os.environ['TRANSFORMERS_OFFLINE'] = '1'

#os.environ['CURL_CA_BUNDLE'] = ''  # requests==2.27.1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = BertForSequenceClassification.from_pretrained('bert-base-chinese', num_labels=16)
model.load_state_dict(torch.load('model.pth'), strict=False)
model.to(device)
model.eval()  # Set to evaluation mode

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('tokenizer_path') 

# If SSLerror
os.environ['TRANSFORMERS_OFFLINE'] = '1'
model_path = "C:/Users/S_MLTask/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=16,
                                                      problem_type="multi_label_classification",  # Explicitly set this
                                                      hidden_dropout_prob=0.2,  # Increase dropout
                                                      attention_probs_dropout_prob=0.2)
model.load_state_dict(torch.load('model.pth'), strict=False)
model.to(device)
model.eval()  # Set to evaluation mode

# Load the tokenizer
tokenizer = BertTokenizer.from_pretrained('tokenizer_path') 

# Predict data
new_data = pd.read_csv("processed_output.csv")
fewdata = new_data[:3000]
fewdata.head(3)

def predict_multilabel(model, tokenizer, dataframe, device, threshold=0.8, batch_size=8):
    texts = dataframe['stripMemo'].tolist()
    predictions = []  # 儲存預測標籤
    probabilities = []  # 儲存機率

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoding = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=200
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits.sigmoid()  # Apply sigmoid to get probabilities  
            probabilities.append(logits.cpu().numpy())  # 收集機率
        
        # Apply threshold to determine labels
        batch_predictions = (logits > threshold).float()
        predictions.append(batch_predictions.cpu().numpy())

    # Concatenate predictions and probabilities
    return np.vstack(predictions), np.vstack(probabilities)

predictions, probabilities = predict_multilabel(model, tokenizer, fewdata, device, threshold=0.6)

# 顯示標籤和機率
for i, row in enumerate(fewdata['stripMemo']):
    print(f"\nSentence: {row}")
    for j, (pred, prob) in enumerate(zip(predictions[i], probabilities[i])):
        label_name = categories[j]
        print(f"{label_name}: Predicted={pred:.0f}, Probability={prob:.3f}")


# Predict labels for the new data
predictions = predict_multilabel(model, tokenizer, fewdata, device, threshold=0.8)

# Map predictions to label names
fewdata.loc[:, 'predicted_labels'] = map_predictions_to_labels(predictions, categories)

# Display results
print(fewdata[['stripMemo', 'predicted_labels']])
