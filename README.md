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
├── # ManAdjustData.xlsx  # Manually adjusted training data (input for model.py).
├── model.pth             # Saved trained model weights.
├── tokenizer_path/       # Saved trained model tokenizer.
└── README.md             # This documentation file.
```


### Result Example

| text | labels |
|--------|-------|
| 考慮帶家人二看 樓上中繼水箱有告知 買方有問到是否能晚上來聽聲音 另外有配歐洲村 給買方他不熟要研究  | 環境寧靜  |
| 不考慮裝潢風格 要拆得太多 而且覺得廁所不好搞  | 裝潢考量  |	
| 空間不錯\~客廳大面落地窗看出去很舒服\~朝向也可以\~如果有確定要下斡旋會再拿羅盤良方位|景觀好, 在意風水, 空間規劃|	
|	﻿玻璃窗哪一間房間太小了，裝潢也不合意，很多地方要重新裝潢| 裝潢考量, 空間規劃|
|	﻿主要是幫朋友看自己本身很喜歡採光也很棒唯一小缺點是擔心戶數比較多一點| 通風採光佳, 敦親睦鄰|
