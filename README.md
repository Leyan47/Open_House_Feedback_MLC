## 🏠 Open House Feedback Multi-Lable Classification

### Background
A single property transaction is rarely achieved without multiple viewings. Through these visits, buyers not only gain a clearer understanding of the property's actual condition, but also begin to visualize their ideal home more concretely.
This project builds a BERT-based multi-label classification model using buyer feedback collected by real estate agents after property viewings. The goal is to help identify and converge on buyers’ flexible preferences, which in turn influences the ranking of property matches based on their fixed, non-negotiable requirements — ultimately improving the precision of property recommendations.

一筆物件的成交少不了多次的帶看，買方在看房後，不僅近一步了解房屋的實際狀況，也具象化自身夢寐以求房屋的想像。
本專案根據仲介輸入的買方帶看回饋，建立BERT多標籤分類模型，協助收斂出買方的彈性需求，進而影響固定剛性需求下的房屋配對排序，提升物件推薦的精準度。

### File Structure
```
.
├── data_process.py       # Script for text preprocessing and rule-based labeling.
├── model.py              # Script for model training and evaluation.
├── Prediction.py         # Script for making predictions with the trained model.
├── # ManAdjustData.xlsx  # Manually adjusted training data (input for model.py), which cannot be uploaded publicly to protect user privacy.
├── model.pth             # Saved trained model weights.
├── tokenizer_path/       # Saved trained model tokenizer.
└── README.md             # This documentation file.
```

### Project Workflow
The project follows a three-step process: data preparation, model training, and prediction.

Step 1: Data Preparation (data_process.py)
This script preprocesses raw text and applies an initial set of labels using a rule-based approach.
Input: A CSV file containing raw text feedback (e.g., takeToseeTextdata.csv).

Process:
- Cleans and normalizes the text.
- Applies labels based on a keyword dictionary.
- Balances the dataset by downsampling unlabeled entries.
Output: A new CSV file Rule_Based_Labeling_T2C.csv with pre-labeled data.

To Run:
`python data_process.py`

Note: For best performance, it's recommended to manually review the labels in Rule_Based_Labeling_T2C.csv and save the final, clean dataset as ManAdjustData.xlsx.

Step 2: Model Training (model.py)
This script fine-tunes the BERT model on your labeled dataset.

Input: The cleaned, labeled data file ManAdjustData.xlsx.
Process:
- Splits the data into training (70%) and validation (30%) sets.
- Trains the bert-base-chinese model for multi-label classification.
- Uses early stopping to save the best model based on validation recall.

Output:
model.pth: The trained model weights.
tokenizer_path/: The corresponding tokenizer files.

To Run:
`python model.py`

Step 3: Prediction (Prediction.py)
Use the trained model to predict labels for new, unseen text data.

Input:
The saved model model.pth and tokenizer in tokenizer_path/.
A CSV file with new text data (e.g., processed_output.csv) containing a column named stripMemo.

Process:
- Loads the model and tokenizer.
- Generates predictions and probabilities for each label.

Output: Prints the predictions and probabilities to the console.

To Run:
`python Prediction.py`


### Result Example

| text | labels |
|--------|-------|
| 考慮帶家人二看 樓上中繼水箱有告知 買方有問到是否能晚上來聽聲音 另外有配歐洲村 給買方他不熟要研究  | 環境寧靜  |
| 不考慮裝潢風格 要拆得太多 而且覺得廁所不好搞  | 裝潢考量  |	
| 空間不錯\~客廳大面落地窗看出去很舒服\~朝向也可以\~如果有確定要下斡旋會再拿羅盤良方位|景觀好, 在意風水, 空間規劃|	
|	﻿玻璃窗哪一間房間太小了，裝潢也不合意，很多地方要重新裝潢| 裝潢考量, 空間規劃|
|	﻿主要是幫朋友看自己本身很喜歡採光也很棒唯一小缺點是擔心戶數比較多一點| 通風採光佳, 敦親睦鄰|
