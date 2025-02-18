import pandas as pd
import numpy as np
import os
import time
# ETL
from typing import List, Union
import ast
# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import torch.utils.data as data 
from transformers import BertTokenizer, BertForSequenceClassification  # tokenizers  0.20.3  pypi
from tqdm import tqdm  # process bar

df = pd.read_excel("ManAdjustData.xlsx")



def convert_to_multi_label(labels: Union[List[str], str], all_possible_labels: List[str] = None) -> pd.Series:
    """
    Convert single or multiple labels into multi-column one-hot encoding format.
    """
    if all_possible_labels is None:
        all_possible_labels = [
        '通風採光佳', '學區房', '景觀好', '敦親睦鄰', '環境寧靜',
        '在意風水', '行動友善', '衛生整潔', '隱私安全性', '養寵物',
        '交通便利', '投資需求', '裝潢考量', '有車位', '屋況佳', '空間規劃'
    ]
    
    # Convert single string to list if necessary
    if isinstance(labels, str):
        # Try to evaluate if it's a string representation of a list
        try:
            labels = ast.literal_eval(labels)
        except:
            labels = [labels]
    
    # Create a series with zeros
    result = pd.Series(0, index=all_possible_labels)
    
    # Set 1 for each label that exists in the input
    for label in labels:
        if label in all_possible_labels:
            result[label] = 1
            
    return result

def expand_labels_column(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    """
    Expand a DataFrame column containing labels into multiple one-hot encoded columns.
    
    Args:
        df: Input DataFrame
        label_column: Name of the column containing labels
        
    Returns:
        pd.DataFrame: Original DataFrame with additional one-hot encoded columns
    """
    # Convert each row's labels to one-hot encoding
    encoded_labels = df[label_column].apply(convert_to_multi_label)
    
    # Combine original DataFrame with encoded labels
    result_df = pd.concat([df, encoded_labels], axis=1)
    
    return result_df

class SentenceDataset(data.Dataset):  # which inherits from the torch.utils.data.Dataset class provided by PyTorch
    """以與 PyTorch 相容的方式建立資料，以利於使用 DataLoader 進行批次、洗牌和載入，作用在於預處理和存取資料的自訂邏輯。."""
    def __init__(self, database):
        self.database = database
        
    def __len__(self):
        return self.database.shape[0]
        
    def __getitem__(self, idx):
        # Fetching data like： text, label = dataset[0]

        text = self.database.iloc[idx]["stripMemo"]# Use "stripMemo" as the text column.
        # Get labels
        label = self.database.iloc[idx][[
        '通風採光佳', '學區房', '景觀好', '敦親睦鄰', '環境寧靜',
        '在意風水', '行動友善', '衛生整潔', '隱私安全性', '養寵物',
        '交通便利', '投資需求', '裝潢考量', '有車位', '屋況佳', '空間規劃'
    ]]
        label = np.array(label, dtype=float)
        return text, label

# Apply the transformation
#df = df.dropna(subset=['labels'])
df_label_onehot = expand_labels_column(df, 'labels')

# Create dataset
dataset = SentenceDataset(df_label_onehot)

# Check
df_label_onehot.head()
print(dataset[0]) 



# Declaring ML Parameters
lr = 1e-4
epoch = 4
batch_size = 8

# 分割數據集與創建加載器
train_len = int(0.7 * len(dataset))
valid_len = len(dataset) - train_len
TrainData1, ValidationData1 = random_split(dataset, [train_len, valid_len])
train_loader = DataLoader(TrainData1, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(ValidationData1, batch_size=batch_size, shuffle=False, num_workers=0)

# 模型與tokenizer
# C:\Users\h3096\Downloads\sentence_transformersparaphrase_multilingual_MiniLM_L12_v2
os.environ['TRANSFORMERS_OFFLINE'] = '1'
#os.environ['CURL_CA_BUNDLE'] = ''
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained('bert-base-chinese',  num_labels=16,
                                                      problem_type="multi_label_classification",  # Explicitly set this
                                                      hidden_dropout_prob=0.2,  # Increase dropout
                                                      attention_probs_dropout_prob=0.2)
model.to(device)
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# If SSLerror
os.environ['TRANSFORMERS_OFFLINE'] = '1'
#os.environ['CURL_CA_BUNDLE'] = ''
model_path = "C:/Users/S_MLTask/.cache/huggingface/hub/models--bert-base-chinese/snapshots/c30a6ed22ab4564dc1e3b2ecbf6e766b0611a33f/"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(model_path, num_labels=16,
                                                      problem_type="multi_label_classification",  # Explicitly set this
                                                      hidden_dropout_prob=0.2,  # Increase dropout
                                                      attention_probs_dropout_prob=0.2)
#model.to(device)
tokenizer = BertTokenizer.from_pretrained(model_path)



# 將 labels 轉為清單格式
df['labels'] = df['labels'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

exploded_df = df.explode('labels')
categories = ['通風採光佳', '學區房', '景觀好', '敦親睦鄰', '環境寧靜', '在意風水', 
              '行動友善', '衛生整潔', '隱私安全性', '養寵物', '交通便利', 
              '投資需求', '裝潢考量', '有車位', '屋況佳', '空間規劃']
category_counts = exploded_df['labels'].value_counts()
result_df = pd.DataFrame(
    [{'Category': category, 'Frequency': category_counts.get(category, 0)} 
     for category in categories]
)

print(result_df)

# Set Weihgt(Inbalance variances)
label_frequencies = [
    5095, 477, 1902, 1785, 2246, 4338, 1510, 2100, 2074, 
    71, 2490, 2522, 7454, 5414, 12927, 22993
]



# 計算標準化的反比權重
max_frequency = max(label_frequencies)
alpha = 0.05
pos_weights = [(max_frequency / freq) ** alpha for freq in label_frequencies]
# 創建 PyTorch 的張量
class_weights = torch.tensor(pos_weights, dtype=torch.float).to(device)
print(class_weights)



# Define optimizer
# using Weight decay combats overfitting by penalizing large weights during optimization.
optimizer = optim.AdamW(model.parameters(), lr=lr)

# Define Loss function
criterion = nn.BCEWithLogitsLoss()# pos_weight=class_weights



# Deal with GPU Memory 
# Add gradient accumulation
ACCUMULATION_STEPS = 4  # 梯度累積步數, 允許我們用更小的batch_size,但累積幾個批次的梯度後再更新模型
batch_size = 8  # Reduce batch size, can...

def train(model, iterator, optimizer, criterion, device):
    model.train()
    train_loss = 0
    optimizer.zero_grad()  # Zero gradients at start
    
    progress_bar = tqdm(iterator, desc="Training", leave=False, dynamic_ncols=True)
    for batch_idx, (sentences, labels) in enumerate(progress_bar):
        # Free up memory
        torch.cuda.empty_cache()
        
        # tokenize with shorter max_length
        encoding = tokenizer(sentences, 
                           return_tensors='pt', 
                           padding=True, 
                           truncation=True,
                           max_length=256)
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        
        # Normalize loss and backward
        loss = loss / ACCUMULATION_STEPS
        loss.backward()
        
        # 每累積N步, 更新模型參數, 清空梯度
        if (batch_idx + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        train_loss += loss.item() * ACCUMULATION_STEPS
        
        progress_bar.set_postfix(loss=loss.item() * ACCUMULATION_STEPS)
        categories = [
        '通風採光佳', '學區房', '景觀好', '敦親睦鄰', '環境寧靜',
        '在意風水', '行動友善', '衛生整潔', '隱私安全性', '養寵物',
        '交通便利', '投資需求', '裝潢考量', '有車位', '屋況佳', '空間規劃'
        ]
        # 每幾步秀一下資料
        if batch_idx % 200 == 0:
            print(f"\nBatch {batch_idx}")
            print(f"Loss: {loss.item():.4f}")
            print("Sample predictions:")
            with torch.no_grad():
                probs = outputs.logits.sigmoid()
                for i in range(min(2, len(sentences))):
                    print(f"\nSentence: {sentences[i]}")
                    for j, p in enumerate(probs[i]):
                        label_name = categories[j]
                        print(f"{label_name}: {p.item():.3f}")
    
    return train_loss


def test(model, iterator, criterion, device):
    model.eval()
    correct = 0
    total = 0
    
    # 添加各種指標的計數器
    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0
    
    progress_bar = tqdm(iterator, desc="Testing", leave=False, dynamic_ncols=True)
    
    with torch.no_grad():
        for batch_idx, (sentences, labels) in enumerate(progress_bar):
            # tokenize the sentences
            encoding = tokenizer(sentences, return_tensors='pt', padding=True, truncation=True, max_length=256)
            input_ids = encoding['input_ids']
            attention_mask = encoding['attention_mask']
            
            # move to GPU
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
            
            # generate prediction
            outputs = model(input_ids, attention_mask=attention_mask)
            prob = outputs.logits.sigmoid()
            
            # Prediction with threshold
            THRESHOLD = 0.75
            prediction = prob.detach().clone()
            prediction[prediction > THRESHOLD] = 1
            prediction[prediction <= THRESHOLD] = 0
            
            # 計算正確預測數
            correct += prediction.eq(labels).sum().item()
            total += (labels.size(0) * labels.size(1))
            
            # 計算各種指標
            true_positives += ((prediction == 1) & (labels == 1)).sum().item()
            false_positives += ((prediction == 1) & (labels == 0)).sum().item()
            true_negatives += ((prediction == 0) & (labels == 0)).sum().item()
            false_negatives += ((prediction == 0) & (labels == 1)).sum().item()
            
            # 每 N batch顯示一些例子
            if batch_idx % 200 == 0:
                for i in range(min(2, len(sentences))):  # 顯示前2個例子
                    print("\n=== Example ===")
                    print("Sentence:", sentences[i])
                    print("\nPredictions vs Actuals:")
                    for j, (pred, true, prob_val) in enumerate(zip(prediction[i], labels[i], prob[i])):
                        if pred.item() == 1 or true.item() == 1:  # 只顯示預測為1或實際為1的標籤
                            print(f"Label {j}: Pred={pred.item():.0f} (Prob={prob_val.item():.3f}) True={true.item():.0f}")
    
    # 計算評估指標
    accuracy = 100.0 * correct / total
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # 打印詳細結果
    print("\n=== Final Results ===")
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Precision (預測為1的準確率): {precision:.4f}')
    print(f'Recall (實際為1的召回率): {recall:.4f}')
    print(f'F1 Score: {precision:.4f}')
    print(f'True Positives (正確預測1): {true_positives}')
    print(f'False Positives (誤判為1): {false_positives}')
    print(f'True Negatives (正確預測0): {true_negatives}')
    print(f'False Negatives (誤判為0): {false_negatives}')
    
    # 計算各標籤的表現
    label_names = [
        '通風採光佳', '學區房', '景觀好', '敦親睦鄰', '環境寧靜',
        '在意風水', '行動友善', '衛生整潔', '隱私安全性', '養寵物',
        '交通便利', '投資需求', '裝潢考量', '有車位', '屋況佳', '空間規劃'
    ]
    
    print("\n=== Performance by Label ===")
    for i in range(labels.size(1)):
        label_pred = prediction[:, i]
        label_true = labels[:, i]
        label_correct = label_pred.eq(label_true).sum().item()
        label_total = len(label_true)
        label_tp = ((label_pred == 1) & (label_true == 1)).sum().item()
        label_fp = ((label_pred == 1) & (label_true == 0)).sum().item()
        label_precision = label_tp / (label_tp + label_fp) if (label_tp + label_fp) > 0 else 0
        
        print(f"\n{label_names[i]}:")
        print(f"Accuracy: {100.0 * label_correct / label_total:.2f}%")
        print(f"Precision for 1's: {label_precision:.4f}")
        print(f"Total predictions of 1: {(label_pred == 1).sum().item()}")
        print(f"Total actual 1's: {(label_true == 1).sum().item()}")
    
    return accuracy, precision, recall, f1_score


best_indicator = 0
patience = 3
no_improve = 0

# 訓練與測試
for e in range(epoch):
    print(f"===== Epoch {e + 1}/{epoch} =====")
    train_loss = train(model, train_loader, optimizer, criterion, device)
    
    # test 函數返回四個指標
    accuracy, precision, recall, f1 = test(model, test_loader, criterion, device)
    
    # 使用 recall 作為 early stopping 的指標
    if recall > best_indicator:
        best_indicator = recall
        no_improve = 0
        # 儲存最佳模型
        torch.save(model.state_dict(), 'model.pth')
        tokenizer.save_pretrained('tokenizer_path')
        print(f"New best accuracy: {accuracy:.2f}%")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered")
            print(f"Best accuracy achieved: {best_acc:.2f}%")
            break


# Save the train model
### Save model state dictionary
torch.save(model.state_dict(), 'model.pth')
tokenizer.save_pretrained('tokenizer_path')

# 清理未使用的 CUDA 記憶體
torch.cuda.empty_cache()
