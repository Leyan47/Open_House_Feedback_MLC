#%%
import pyodbc
import pandas as pd

connection = pyodbc.connect(
    "Driver={ODBC Driver 13 for SQL Server};"
    "Server=srvdbspss;"
    "Database=DataMining;"
    "Trusted_Connection=yes;"
)
#%%
# query_bq_lm='''
# SELECT
#     buyquestion,likememo
# FROM byt13
# WHERE (buyquestion <> '' or likememo <> '') and create_date>='20250415' and create_date<'20250515'
# '''

query_touch='''
SELECT 
    cus_no,
    touch_content
FROM  byt14
WHERE byt14.touch_code <> 97
    AND touch_content IS NOT NULL
    AND touch_content <> ''
    and create_date>='20250415' and create_date<'20250515'
    AND NOT (
    touch_content LIKE '%掛%' OR
    touch_content LIKE '%未接%' OR
    touch_content LIKE '%沒接%' OR
    touch_content LIKE '%語音%' OR
    touch_content LIKE '%沒有接%' OR
    touch_content LIKE '%無法接聽%' OR
    touch_content LIKE '%通話失敗%'
    )
'''

# df_buyquestion_likememo = pd.read_sql_query(query_bq_lm, connection)
df_touch = pd.read_sql_query(query_touch, connection)

connection.close()

#%%
import re

# --- 階段一規則定義 ---

# 1. 標記性噪音清除
# 僅包含一個英文字母 (這裡假設是大寫，如果也需小寫可調整)
# 使用正則表達式來精確匹配單個大寫字母
marker_noise_substrings = ["(沒資訊)"]
single_char_noise_pattern = r"^[A-Za-z]$"
a_uppercase_digits_pattern = r"^A[A-Z]+[0-9]+$"
# 2. 純內部流程/系統訊息
# 這些詞語如果單獨出現，或者沒有其他資訊性內容時，視為噪音
# 注意："介紹"是個重要的例外指示詞
internal_phrases_exact_match = [
    "已達上限",
    "任務推薦案件跟客戶介紹中", # 如果沒有"(沒資訊)"，此句本身資訊量也低
    "未接"
]
# 這些詞語如果出現，但沒有伴隨"介紹"或其他核心房地產詞彙，可能為噪音
# 特別處理 "特助" 相關，如果包含 "介紹" 或其他房地產核心詞，則不視為噪音
internal_substrings_conditional = {
    "特助": ["傳送特助", "已傳特助"] # "特助介紹" 另行處理
}
# 資訊指示詞 (如果與 internal_substrings_conditional 中的詞同時出現，則不視為噪音)
info_indicator_keywords = ["介紹"]


# 3. 極短且無意義文本
short_text_threshold = 4 # 小於4個字元 (即長度 0, 1, 2, 3)
core_keywords = ["房", "廳", "衛", "透天","廈", "萬", "千", "元", "捷運", "區", "樓", "路", "車", "坪", "租",
                 "買", "售", "看", "約", "地點", "預算", "格局","店面",
                 # 區域
                 '松山', '信義', '大安', '中山', '中正', '大同', '萬華', '文山',
                 '南港', '內湖', '士林', '北投', '板橋', '三重', '中和', '永和', '新莊', '新店',
                 '樹林', '鶯歌', '三峽', '淡水', '汐止', '瑞芳', '土城', '蘆洲', '五股', '泰山',
                 '林口', '深坑', '石碇', '坪林', '三芝', '石門', '八里', '平溪', '雙溪', '貢寮',
                 '金山', '萬里', '烏來']

# 4. 重複性罐頭回應
canned_responses_exact_match = [
    "看一下資料",
    "i特助介紹","傳I特助介紹","上傳I特助介紹",
    "i特助推薦","用i特助推薦"
]

# --- 執行篩選 ---
excluded_texts_with_reason = []
kept_texts = []

for text_input in df_touch['touch_content']: # Use df_touch['touch_content']
    if not isinstance(text_input, str): # Handle potential non-string data if any
        original_text = str(text_input)
    else:
        original_text = text_input

    text = original_text.strip() # Work with stripped text for most checks
    is_noise = False
    reason = ""

    # Rule Priority:
    # 1. Strongest, most direct noise indicators first.
    # 2. More specific contextual rules.
    # 3. General heuristics like length.

    # Rule: Contains marker "(沒資訊)"
    if not is_noise:
        for marker in marker_noise_substrings:
            if marker in text: # Check in 'text' not 'original_text' if marker can have spaces
                is_noise = True
                reason = f"包含標記 '{marker}'"
                break
    if is_noise:
        excluded_texts_with_reason.append((original_text, reason))
        continue

    # Rule: Only digits
    if not is_noise:
        if text.isdigit() and text: # Ensure text is not empty before isdigit
            is_noise = True
            reason = f"僅包含數字"
    if is_noise:
        excluded_texts_with_reason.append((original_text, reason))
        continue

    # Rule: Single alphabetic character
    if not is_noise:
        if re.fullmatch(single_char_noise_pattern, text):
            is_noise = True
            reason = f"僅為單一英文字母"
    if is_noise:
        excluded_texts_with_reason.append((original_text, reason))
        continue

    if not is_noise:
        if re.fullmatch(a_uppercase_digits_pattern, text):
            is_noise = True
            reason = f"符合A[A-Z]+[0-9]+模式" # (Pattern: A + Uppercase Letters + Digits)
    if is_noise:
        excluded_texts_with_reason.append((original_text, reason))
        continue

    # Rule: Contains "特助" and length <= 10, AND does not contain other core keywords
    if not is_noise:
        if "特助" in text and len(text) <= 10:
            # Check if it ALSO contains core keywords. If so, it might be info.
            # Example: "特助，3房" (length 5) -> keep | "特助幫約" (length 4) -> noise
            contains_core_info = any(core_kw in text for core_kw in core_keywords if core_kw != "特助") # Avoid "特助" being its own core kw here
            if not contains_core_info:
                is_noise = True
                reason = f"包含'特助'且長度<=10且無其他核心資訊詞"
    if is_noise:
        excluded_texts_with_reason.append((original_text, reason))
        continue

    # Rule: Exact match internal/system phrases
    if not is_noise:
        if text in internal_phrases_exact_match:
            is_noise = True
            reason = f"完全匹配內部流程語"
    if is_noise:
        excluded_texts_with_reason.append((original_text, reason))
        continue

    # Rule: Very short text without digits or core keywords
    if not is_noise:
        if len(text) < short_text_threshold:
            contains_digit_in_short = any(char.isdigit() for char in text)
            contains_core_keyword_in_short = any(kw in text for kw in core_keywords)
            if not contains_digit_in_short and not contains_core_keyword_in_short:
                is_noise = True
                reason = f"極短(長度{len(text)})且無數字/核心詞"
    if is_noise:
        excluded_texts_with_reason.append((original_text, reason))
        continue

    # If no noise rule matched, keep the text
    kept_texts.append(original_text)

# --- Output Results ---
print("--- 前10筆被排除的文本及其原因 ---")
count = 0
for text, reason in excluded_texts_with_reason:
    print(f"{count+1}. 文本: \"{text}\" \n   原因: {reason}")
    count += 1
    if count >= 10:
        break

print(f"\n總共排除 {len(excluded_texts_with_reason)} 筆文本")
print(f"總共保留 {len(kept_texts)} 筆文本")


kept_texts = pd.DataFrame(kept_texts)
kept_texts[:700].to_excel("touch_content_row.xlsx", index=False)

#%% 
import pandas as pd
import jieba # 用於中文斷詞
from sklearn.model_selection import (train_test_split, StratifiedKFold,
                                     cross_val_score, RandomizedSearchCV) # 新增 RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score # 新增 f1_score for scoring
import numpy as np # 用於處理可能的NaN
from imblearn.over_sampling import SMOTE # 用於處理類別不平衡
from scipy.stats import randint, uniform # 用於 RandomizedSearchCV 的參數分佈

# 引入要測試的模型
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 保存模型
import joblib

# --- 1. 資料載入與預處理模組 ---
def load_and_preprocess_data(file_path="touch_content.xlsx"):

    try:
        df = pd.read_excel(file_path)
        df.columns = ['text', 'y'] # 假設固定欄位名
        print(f"成功載入資料，資料筆數: {len(df)}")
    except FileNotFoundError:
        print(f"錯誤：找不到 '{file_path}' 檔案。請檢查檔案路徑和名稱。")
        return None, None
    df['text'] = df['text'].fillna('').astype(str)
    df = df.dropna(subset=['y'])
    df['y'] = df['y'].astype(int)
    print("\n開始進行中文斷詞...")
    df['text_segmented'] = df['text'].apply(lambda text: " ".join(jieba.cut(text)))
    print("斷詞完成。")
    return df, df['text_segmented']

def get_tfidf_features(segmented_texts, max_features=1000, min_df=1, ngram_range=(1,1)):
    """使用TF-IDF提取文本特徵"""
    vectorizer = TfidfVectorizer(max_features=max_features, min_df=min_df, ngram_range=ngram_range)
    X = vectorizer.fit_transform(segmented_texts)
    print(f"\nTF-IDF 特徵提取完成，特徵矩陣維度: {X.shape}")
    return X, vectorizer


# --- 2. 模型訓練與評估模組  ---
def train_evaluate_model(model_instance_creator, X_train, y_train, X_test, y_test,
                         param_distributions=None, use_smote=False, model_name="模型",
                         original_df_for_error_analysis=None,
                         test_indices_for_error_analysis=None,
                         n_iter_search=100, cv_folds_search=3): # 新增超參數調優相關參數
    """
    訓練、評估指定的模型，可選進行超參數調優，並執行錯誤分析。
    """
    print(f"\n--- 開始處理: {model_name} ---")

    X_train_processed = X_train.copy()
    y_train_processed = y_train.copy()
    best_model = None

    if use_smote:
        print("對訓練數據使用 SMOTE 進行過採樣...")
        smote = SMOTE(random_state=42)
        X_train_processed, y_train_processed = smote.fit_resample(X_train_processed, y_train_processed)
        print(f"SMOTE 後訓練集大小: {X_train_processed.shape[0]}, 標籤分佈:\n{pd.Series(y_train_processed).value_counts(normalize=True)}")

    # 超參數調優 (如果提供了參數分佈)
    if param_distributions:
        print(f"開始對 {model_name} 進行 RandomizedSearchCV 超參數調優...")
        # 創建模型實例 (因為 RandomizedSearchCV 會 clone 模型)
        base_model = model_instance_creator()

        # RandomizedSearchCV
        # scoring='f1_weighted' or 'f1_macro' or f1 (for positive class)
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter_search, # 嘗試的參數組合數量
            cv=StratifiedKFold(n_splits=cv_folds_search, shuffle=True, random_state=42), # 交叉驗證折數
            scoring='f1_weighted', # 或 'roc_auc', 'accuracy', 'recall_weighted' 等
            random_state=42,
            n_jobs=-1, # 使用所有可用的 CPU 核心
            verbose=1 # 顯示進度
        )
        try:
            random_search.fit(X_train_processed, y_train_processed)
            print(f"最佳參數: {random_search.best_params_}")
            print(f"最佳 F1 (weighted) 分數 (交叉驗證): {random_search.best_score_:.4f}")
            best_model = random_search.best_estimator_ # 使用找到的最佳模型
        except Exception as e:
            print(f"超參數調優過程中發生錯誤 for {model_name}: {e}")
            print("將使用默認參數的模型進行訓練。")
            best_model = model_instance_creator() # 出錯則回退到默認模型
            best_model.fit(X_train_processed, y_train_processed)

    else: # 不進行超參數調優，直接訓練
        print("不進行超參數調優，使用默認/指定參數的模型。")
        best_model = model_instance_creator() # 創建模型實例
        best_model.fit(X_train_processed, y_train_processed)

    print(f"{model_name} 模型訓練完成。")

    print(f"\n開始評估 {model_name} (使用最佳/默認模型)...")
    y_pred = best_model.predict(X_test)
    y_pred_proba_all_classes = np.zeros((X_test.shape[0], 2))

    if hasattr(best_model, "predict_proba"):
        y_pred_proba_all_classes = best_model.predict_proba(X_test)
    # ... (處理 predict_proba 不可用的情況，與之前類似) ...

    # 提取標籤1的機率
    y_pred_proba_class1 = y_pred_proba_all_classes[:, 1] if y_pred_proba_all_classes.shape[1] == 2 else np.zeros(len(y_pred))


    print(f"\n{model_name} - 測試集整體準確率 (Accuracy):", accuracy_score(y_test, y_pred))
    print(f"{model_name} - 測試集混淆矩陣 (Confusion Matrix):\n", confusion_matrix(y_test, y_pred))
    print(f"{model_name} - 測試集分類報告 (Classification Report):\n",
          classification_report(y_test, y_pred, target_names=['噪音 (0)', '有效資訊 (1)'], zero_division=0))

    # 交叉驗證 (如果沒有進行超參數調優，或者想再次確認)
    # 注意：如果在超參數調優中已經做了交叉驗證，這裡的交叉驗證主要是對默認模型或最終選定模型的一個獨立評估
    if not param_distributions: # 只有在沒有進行超參數搜索時才執行這裡的CV
        print(f"\n{model_name} - 交叉驗證 (在原始訓練集上，使用默認/指定參數):")
        try:
            if X_train.shape[0] > 0 and y_train.shape[0] > 0:
                # 創建一個新的實例，以防之前的模型狀態影響CV
                cv_model_instance = model_instance_creator()
                cv_folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(cv_model_instance, X_train, y_train, cv=cv_folds, scoring='f1_weighted')
                print(f"  交叉驗證 F1 (weighted avg) 分數: {scores}")
                print(f"  平均 F1 (weighted avg) 分數: {scores.mean():.4f}")
            else:
                print("  訓練數據不足，無法進行交叉驗證。")
        except Exception as cv_e:
            print(f"  交叉驗證失敗: {cv_e}")

    if original_df_for_error_analysis is not None and test_indices_for_error_analysis is not None:
        print(f"\n{model_name} - 錯誤分析:")
        perform_error_analysis(y_test, y_pred, y_pred_proba_class1,
                               original_df_for_error_analysis, test_indices_for_error_analysis)
    print(f"--- {model_name} 處理完畢 ---")
    return best_model

def perform_error_analysis(y_true, y_pred, y_pred_probas_class1, df_original, test_indices):
    """執行並打印錯誤分析結果"""
    # y_true 應該是 Pandas Series 以便使用 .index 和布林索引
    # y_pred 和 y_pred_probas_class1 應該是 NumPy arrays

    # 確保 y_true 是 Pandas Series，如果不是，嘗試轉換 (儘管在您的主流程中它應該是)
    if not isinstance(y_true, pd.Series):
        # 如果 test_indices 是有效的索引並且長度匹配，可以用它來創建 Series
        if test_indices is not None and len(test_indices) == len(y_true):
            y_true_series = pd.Series(y_true, index=test_indices, name="y_true")
        else:
            # 如果無法恢復索引，錯誤分析的準確性會受影響，但至少可以嘗試按順序匹配
            print("警告: y_true 不是 Pandas Series 且無法安全恢復索引，錯誤分析可能不按原始索引顯示。")
            y_true_series = pd.Series(y_true, name="y_true") # 創建一個無索引或默認索引的Series
    else:
        y_true_series = y_true


    # 創建布林遮罩 (masks)
    # 確保比較時數據類型一致，例如將 Series 轉為 numpy array
    fn_mask = (y_true_series.to_numpy() == 1) & (y_pred == 0)
    fp_mask = (y_true_series.to_numpy() == 0) & (y_pred == 1)

    # 使用 test_indices (這些是 y_test 在原始 df 中的索引) 和布林遮罩來獲取實際的原始索引
    # fn_mask 和 fp_mask 的長度應該與 y_test/test_indices 相同
    fn_actual_original_indices = test_indices[fn_mask]
    fp_actual_original_indices = test_indices[fp_mask]

    print(f"\n--- 錯誤類型1: 假陰性 (FN) - 真實為'有效資訊(1)', 模型預測為'噪音(0)' (共 {len(fn_actual_original_indices)} 筆) ---")
    if not fn_actual_original_indices.empty: # 檢查索引 Series 是否為空
        for original_idx in fn_actual_original_indices:
            try:
                text = df_original.loc[original_idx, 'text']
                true_label_from_df = df_original.loc[original_idx, 'y'] # 從原始df獲取真實標籤以雙重確認

                # 找到這個 original_idx 在 y_pred 和 y_pred_probas_class1 中的對應位置
                # 這需要 y_true (即 y_test) 的索引與 original_idx 的關係
                # 我們假設 test_indices 與 y_true.index 是一致的
                position_in_test_set_tuple = np.where(test_indices == original_idx)

                if len(position_in_test_set_tuple[0]) > 0:
                    pos = position_in_test_set_tuple[0][0] # np.where 返回元組，取第一個陣列的第一個元素
                    predicted_label = y_pred[pos]
                    predicted_proba = y_pred_probas_class1[pos]

                    print(f"  原始索引: {original_idx}, 文本: \"{text}\"")
                    print(f"  真實標籤(df): {true_label_from_df}, 模型預測: {predicted_label}, 預測為有效資訊的機率: {predicted_proba:.3f}")
                    print("-" * 30)
                else:
                    print(f"  警告: 無法在測試集預測中找到原始索引 {original_idx} 的對應項。文本: \"{text}\"")
            except KeyError:
                print(f"  錯誤: 原始索引 {original_idx} 在原始 DataFrame 中未找到。")
            except Exception as e:
                print(f"  處理原始索引 {original_idx} 時發生未知錯誤: {e}")
    else:
        print("  沒有此類型的錯誤。")

    print(f"\n--- 錯誤類型2: 假陽性 (FP) - 真實為'噪音(0)', 模型預測為'有效資訊(1)' (共 {len(fp_actual_original_indices)} 筆) ---")
    if not fp_actual_original_indices.empty:
        for original_idx in fp_actual_original_indices:
            try:
                text = df_original.loc[original_idx, 'text']
                true_label_from_df = df_original.loc[original_idx, 'y']

                position_in_test_set_tuple = np.where(test_indices == original_idx)
                if len(position_in_test_set_tuple[0]) > 0:
                    pos = position_in_test_set_tuple[0][0]
                    predicted_label = y_pred[pos]
                    predicted_proba = y_pred_probas_class1[pos]

                    print(f"  原始索引: {original_idx}, 文本: \"{text}\"")
                    print(f"  真實標籤(df): {true_label_from_df}, 模型預測: {predicted_label}, 預測為有效資訊的機率: {predicted_proba:.3f}")
                    print("-" * 30)
                else:
                    print(f"  警告: 無法在測試集預測中找到原始索引 {original_idx} 的對應項。文本: \"{text}\"")
            except KeyError:
                print(f"  錯誤: 原始索引 {original_idx} 在原始 DataFrame 中未找到。")
            except Exception as e:
                print(f"  處理原始索引 {original_idx} 時發生未知錯誤: {e}")
    else:
        print("  沒有此類型的錯誤。")

# --- 3. 主執行流程 ---
if __name__ == "__main__":
    df_original, segmented_texts = load_and_preprocess_data("touch_content.xlsx")

    final_model = None  # 用來存要用的模型(到時可以考慮混和?)

    if df_original is not None:
        # 調整 TF-IDF 參數，例如增加 ngram_range 或調整 min_df
        X_features, vectorizer = get_tfidf_features(segmented_texts, ngram_range=(1, 2), min_df=2, max_features=1500)
        y_labels = df_original['y']

        if not y_labels.empty and len(y_labels.unique()) > 1:
            X_train, X_test, y_train, y_test = train_test_split(
                X_features, y_labels, test_size=0.2, random_state=47, stratify=y_labels
            )
            test_original_indices = y_test.index
            print(f"\n資料集分割完成。訓練集大小: {X_train.shape[0]}, 測試集大小: {X_test.shape[0]}")
        else:
            # ... (處理數據不足的情況，與之前相同) ...
            X_train, X_test, y_train, y_test = (None,)*4
            test_original_indices = None


        if X_train is not None and X_train.shape[0] > 0:
            # 為每個模型定義參數分佈 (用於 RandomizedSearchCV)
            # 注意：這些範圍僅為示例，您需要根據對模型的理解和實驗來調整
            # param_dist_lgbm = {
            #     'n_estimators': randint(50, 200),
            #     'learning_rate': uniform(0.01, 0.1), # 注意 uniform(loc, loc+scale)
            #     'num_leaves': randint(10, 40), # 調整num_leaves
            #     'min_child_samples': randint(5, 25), # 調整min_child_samples
            #     'reg_alpha': uniform(0, 1),
            #     'reg_lambda': uniform(0, 1),
            #     'colsample_bytree': uniform(0.6, 0.4), # scale is 0.4, so range is 0.6 to 1.0
            #     'subsample': uniform(0.6, 0.4),
            #     # 'class_weight': ['balanced', None], # 可以在這裡試驗，或依賴SMOTE
            #     # 'scale_pos_weight': [1, sum(y_train==0)/sum(y_train==1) if sum(y_train==1)>0 else 1] # 另一種處理不平衡
            # }

            # param_dist_xgb = {  # 表現最爛
            #     'n_estimators': randint(50, 200),
            #     'learning_rate': uniform(0.01, 0.1),
            #     'max_depth': randint(3, 10),
            #     'min_child_weight': randint(1, 10),
            #     'gamma': uniform(0, 0.5),
            #     'subsample': uniform(0.6, 0.4),
            #     'colsample_bytree': uniform(0.6, 0.4),
            #     'reg_alpha': uniform(0,1),
            #     'reg_lambda': uniform(0,1),
            #     # 'scale_pos_weight': [sum(y_train==0)/sum(y_train==1) if sum(y_train==1)>0 else 1]
            # }

            param_dist_svc = {
                'C': uniform(0.1, 10),
                'gamma': ['scale', 'auto'] + list(uniform(0.001, 1).rvs(5)), # 嘗試一些具體值
                'kernel': ['rbf', 'linear'] # 可以嘗試不同的核函數
                 # 'class_weight': ['balanced'] # 固定或也放入搜索
            }
            # param_dist_catboost = {
            #     'iterations': randint(50, 200),
            #     'learning_rate': uniform(0.01, 0.1),
            #     'depth': randint(3, 8),
            #     'l2_leaf_reg': uniform(1, 10),
            #     # 'auto_class_weights': ['Balanced', 'SqrtBalanced', None] # 嘗試不同的不平衡處理
            # }


            # 定義模型創建者 lambda 函數 和對應的參數分佈
            models_to_tune = {
                #"LightGBM": (lambda: LGBMClassifier(random_state=42, class_weight='balanced'), param_dist_lgbm),
                # "XGBoost": (lambda: XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss',
                #                                  scale_pos_weight= (sum(y_train==0)/sum(y_train==1) if sum(y_train==1)>0 else 1)), param_dist_xgb),
                "SVM": (lambda: SVC(random_state=42, probability=True, class_weight='balanced'), param_dist_svc),
                #"CatBoost": (lambda: CatBoostClassifier(random_state=42, verbose=0, auto_class_weights='Balanced'), param_dist_catboost),
                 # 您也可以加入 LogisticRegression 和 MultinomialNB，但它們的超參數較少
                "Logistic Regression": (lambda: LogisticRegression(solver='liblinear', random_state=42, class_weight='balanced'),
                                        {'C': uniform(0.1, 10), 'penalty': ['l1', 'l2']}),
                "Multinomial NB": (lambda: MultinomialNB(), {'alpha': uniform(0.01, 1.0)})
            }

            apply_smote_for_all = True # True 或 False
            # n_iter_search: 每次 RandomizedSearchCV 嘗試的參數組合數量。對於小數據集，可以設小一點。
            # cv_folds_search: 超參數搜索時交叉驗證的折數。
            num_rand_search_iterations = 50
            num_cv_folds_for_search = 3   # 3折交叉驗證

        
        for model_name, (model_creator, params_dist) in models_to_tune.items():
            print(f"\n>>>>>> 正在處理模型: {model_name} <<<<<<")
            model = train_evaluate_model(
                model_creator, X_train, y_train, X_test, y_test,
                param_distributions=params_dist, # 傳入參數分佈
                use_smote=apply_smote_for_all,
                model_name=model_name,
                original_df_for_error_analysis=df_original,
                test_indices_for_error_analysis=test_original_indices,
                n_iter_search=num_rand_search_iterations,
                cv_folds_search=num_cv_folds_for_search
            )
            if model_name == "SVM":
                final_model = model  # 保存訓練好的SVM模型
        else:
            print("由於資料分割失敗或訓練數據不足，跳過模型訓練與評估。")



# 保存最終模型
joblib.dump(model, 'svm_noise_classifier.joblib')

# 保存TF-IDF轉換器到
joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
