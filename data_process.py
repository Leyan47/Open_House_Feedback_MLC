#%% Load Data
import pandas as pd
import re
import numpy as np
import pyodbc
import os
os.getcwd()

# import urllib3; urllib3.disable_warnings()
# import os; os.environ['CURL_CA_BUNDLE'] = ''
# import jieba  # 分詞用
# import jieba.posseg as pseg


# connection = pyodbc.connect(
#     "Driver={ODBC Driver 13 for SQL Server};"
#     "Server=srvdbspss;"
#     "Database=DataMining;"
#     "Trusted_Connection=yes;"
# )

# query = """
# WITH uniTable AS (
#     SELECT 
#         service_no,
#         likememo,
#         case_name2 AS case_name,
#         create_date,
#         touch_code,
#         ROW_NUMBER() OVER (PARTITION BY likememo, create_date ORDER BY service_no) AS row_num
#     FROM DataMining..byt13
# )
# SELECT 
#     create_date,
# 	likememo,
#     case_name
# FROM uniTable
# WHERE row_num = 1 -- 僅保留每組 (likememo, create_date) 的第一條記錄
# AND create_date between '2024-08-01' and '2024-11-30'
# AND touch_code <> 99
# AND likememo <> ''
# AND not (likememo = '不考慮' or likememo = '不喜歡'　or likememo = '在討論'　or likememo = '再討論' or likememo like '%配案%' or
# 		 likememo = '評估中' or  likememo = '再評估' or likememo = '在評估' or likememo like '%教育行情%'
# )
# order by service_no
# """
# df = pd.read_sql_query(query, connection)
# connection.close()

# df.to_csv("takeToseeTextdada.csv", index=False, encoding="utf-8-sig")
df = pd.read_csv("C:\\Users\\h3096\\t2c_text\\train data\\takeToseeTextdata(20240801_20241131).csv")

#%% 文本預處理
### Stopwords Cleaning
stopwords = set(["想想", "覺得", "也", "是", "又" ,"都", "而且","因為","因此","什麼",\
                 "還", "中","討論","評估","真","的", "了", "一下","遺下", "再", "在",\
                 "太太", "先生","夫妻","女兒","兒子","爸爸","媽媽", "但是","目前",
                 "和","跟","回去","回家","看看","這間","阿","啊","ㄚ","家人",\
                 "超過","預算","價格","總價"])


def preprocess_text(text):
    """替換英文、移除標點符號"""
    replacement_dict = {"OK": "好", "ok": "好", "push": "推進", "1": "一", "2": "二",
                        "3": "三", "4":"四", "5": "五", "6": "六", "7": "七", "8": "八",
                        "9": "九","0": "零","view": "景觀","VIEW": "景觀", "奸":"間","是內":"室內"}
    for key, value in replacement_dict.items():
        text = re.sub(key, value, text)  # 用正則表達式替換
    text = re.sub(r"[^\u4e00-\u9fa5]", " ", text)
    # 分詞 text = jieba.lcut(text)
    # 移除停用詞
    for word in stopwords:
        text = text.replace(word, "")
    # 移除多餘空格
    text = " ".join(text.split())
    return text

### 處理文本
df["stripMemo"] = df["likememo"].apply(preprocess_text)
df = df[df["stripMemo"] != '']

### Rule-Based Labeling(採光佳、學區房、景觀好、交通便利、房屋結構、在意風水、行動友善、隱私安全感、、)
### rc、sc、src、建材、耐震等級、結構、隔音效果、建商、國泰、
### 空間規劃:動線、進出、收納、規劃
### 社區氛圍(物)社區品質、管理員、代收垃圾、代收包裹、社區環境、生活機能、地點
### 運動休閒: 散步、運動、游泳
### 衛生整潔: ["髒","味道","臭","下水道","貓味","狗味","發霉","霉味","溝邊","水溝","市場",'夜市',"凌亂","雜亂","灰塵","髒污","怪味","異味"]
categories = ['通風採光佳', '學區房', '景觀好', '敦親睦鄰', '環境寧靜', '在意風水', '行動友善','衛生整潔',\
              '隱私安全性', '養寵物', '交通便利', '投資需求', '裝潢考量', '有車位', '屋況佳' , '空間規劃']  # Remove "價格相關"

keywords = {  # 4個月期間個數  # 有潛力標籤：戶外空間 / 代收垃圾 

    "通風採光佳": ["採光", "光線","陽光","通風","透風","日照","悶","對流","不流通"],

    "學區房": ["學校", "學區","國小","國中"],

    "景觀好": ["景觀", "風景","河景","面河","山景","海景","視野","遠景","綠景"],

    "敦親睦鄰":["住戶", "鄰居", "其他戶", "戶數", "社區人口","社區的人", "進出人口", "進出的人","年齡層","租戶","複雜",\
               "酒吧","按摩會館","按摩院","色情"],

    "環境寧靜": ["聲","吵","安靜","寧靜","噪音","清幽"], 

    "在意風水": ["宮","廟","風水","方正","方位","路沖","路衝","墳","墓","壁刀","西曬","擲杯",\
                "朝南","朝北","朝東","朝西","向東","向西","向南","向北","東北","西北",\
                "東南","西南","座相","座向","坐向","坐相","老師","神明","非自然","凶宅","福地",\
                "攔腰","穿堂","壓樑","殯儀","氣場","不正","斜角","漏財","不正","尖角","缺角"],   

    "行動友善": ["樓梯","無障礙","老人","輪椅","殘障","殘疾"],
    
    "衛生整潔": ["髒","味道","臭","下水道","貓味","狗味","發霉","霉味","溝邊","水溝","市場",'夜市',\
                "凌亂","雜亂","灰塵","髒污","怪味","異味","油煙","菸味","煙味","糞","蟑螂","衛生","整潔","乾淨"],

    "隱私安全性":["隱私","私密性","逃生","安全","傾斜","結構","棟距","隔間","間隔","基地不正"], 

    "養寵物":["養寵物","養貓","隻貓","貓咪","貓砂","餵貓","養狗","隻狗","狗狗","養兔","養一","養二","養兩","養三","養四","養五","養六","養七","養八","養九"],#50
    # 需要陽台、大空間、中庭、陽露台、社區規範可養、公園
    "交通便利": ["捷運", "公車","交通","巴士"],
    "投資需求":["投資","出租","收租","租金","報酬","投報","置產"],
    "裝潢考量": ["裝潢","室內設計","裝修","翻新","大改"],

    "有車位": ["車庫","車位","停車"], 

    "屋況佳":["氯離子","梨子","海砂","滲","屋況","壁癌","整理","老舊","漏水","沒窗"],

    "空間規劃": ["坪數","格局","空間","室內空間","室內空家","室內太小","小間","間太小","壓樑","動線","進出",\
                "收納","規劃"]
}

def label_text(text, keywords):
    labels = []
    for category, words in keywords.items():
        pattern = '|'.join(map(re.escape, words))
        if re.search(pattern, text):
            labels.append(category)
    return labels

#%% Save strip data
df["labels"] = df["stripMemo"].apply(lambda x: label_text(x, keywords))
df_useless = df[(df["stripMemo"].str.len() <= 3) & (df['labels'].str.len() < 1)]
df = df[(df["stripMemo"].str.len() >= 4) | (df['labels'].str.len() >= 1)]  # 去除字數短 且 沒標籤的資料
df = df[["likememo", "stripMemo","labels"]]

empty_labels_df = df[df["labels"].apply(lambda x: x == [])]
non_empty_labels_df = df[df["labels"].apply(lambda x: x != [])]
np.random.seed(42)  # 設定隨機種子以確保結果可重現
stay_empty_labels_df = empty_labels_df.sample(frac=0.3)  # 從空標籤的資料中隨機留下

# 合併保留的空標籤資料和非空標籤資料
new_df = pd.concat([non_empty_labels_df, stay_empty_labels_df])
new_df.to_csv("Rule_Based_Labeling_T2C.csv", index=False, encoding="utf-8-sig")

#%%
filtered_df = df[df["labels"].str.contains("價格相關")]
filtered_df = df[df["labels"].apply(lambda x: "價格相關" in x)]
print(filtered_df["labels"])


category_counts = {}

# 如果 labels 是列表形式，可以這樣檢查檢查每個類別
for category in categories:
    if isinstance(df["labels"].iloc[0], list):
        # 計算包含該類別的資料列數量
        count = df[df["labels"].apply(lambda x: category in x)].shape[0]
        category_counts[category] = count

pd.Series(category_counts).sort_values(ascending=False)

