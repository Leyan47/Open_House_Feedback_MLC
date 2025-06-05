import argparse
import pandas as pd
import jieba
import joblib
import os

# --- 預處理函數 (需要與訓練時的預處理保持一致) ---
def preprocess_text_for_prediction(text_series):
    """對輸入的文本序列進行斷詞處理"""
    print("對輸入文本進行中文斷詞...")
    # 確保輸入是字串
    text_series = text_series.astype(str).fillna('')
    segmented_texts = text_series.apply(lambda text: " ".join(jieba.cut(text)))
    print("斷詞完成。")
    return segmented_texts

# --- 核心處理函數 ---
def filter_noise_from_file(input_file, output_file, model_path, vectorizer_path,
                           input_format='txt', text_column=None, output_format='txt',
                           positive_label=1):
    """
    從輸入檔案中讀取文本，使用預訓練模型過濾噪音，並將有效資訊輸出到檔案。

    參數:
    input_file (str): 輸入檔案路徑。
    output_file (str): 輸出檔案路徑。
    model_path (str): 預訓練模型的檔案路徑。
    vectorizer_path (str): 預訓練TF-IDF轉換器的檔案路徑。
    input_format (str): 輸入檔案格式 ('txt', 'csv', 'excel')。
    text_column (str/int): 如果輸入是csv/excel，指定包含文本的欄位名或索引。
    output_format (str): 輸出檔案格式 ('txt', 'csv', 'excel')。
    positive_label (int): 模型中代表「有效資訊」的標籤 (通常是1)。
    """
    print(f"正在載入模型從: {model_path}")
    if not os.path.exists(model_path):
        print(f"錯誤: 模型檔案 '{model_path}' 不存在。")
        return
    model = joblib.load(model_path)
    print("模型載入成功。")

    print(f"正在載入TF-IDF轉換器從: {vectorizer_path}")
    if not os.path.exists(vectorizer_path):
        print(f"錯誤: TF-IDF轉換器檔案 '{vectorizer_path}' 不存在。")
        return
    vectorizer = joblib.load(vectorizer_path)
    print("TF-IDF轉換器載入成功。")

    # 讀取輸入檔案
    texts_to_predict = []
    original_data_df = None # 用於CSV/Excel輸出時保留其他欄位

    print(f"正在讀取輸入檔案: {input_file} (格式: {input_format})")
    try:
        if input_format == 'txt':
            with open(input_file, 'r', encoding='utf-8') as f:
                texts_to_predict = [line.strip() for line in f if line.strip()]
            if not texts_to_predict:
                print("警告: 輸入文字檔案為空或沒有有效內容。")
                return
            texts_series = pd.Series(texts_to_predict)
        elif input_format == 'csv':
            original_data_df = pd.read_csv(input_file)
            if text_column is None:
                print("錯誤: CSV格式輸入必須指定文本欄位 (--text_column)。")
                return
            if isinstance(text_column, str) and text_column not in original_data_df.columns:
                print(f"錯誤: CSV檔案中找不到欄位 '{text_column}'。")
                return
            elif isinstance(text_column, int) and text_column >= len(original_data_df.columns):
                print(f"錯誤: CSV檔案中欄位索引 {text_column} 超出範圍。")
                return
            texts_series = original_data_df.iloc[:, text_column] if isinstance(text_column, int) else original_data_df[text_column]
        elif input_format == 'excel':
            original_data_df = pd.read_excel(input_file)
            if text_column is None:
                print("錯誤: Excel格式輸入必須指定文本欄位 (--text_column)。")
                return
            if isinstance(text_column, str) and text_column not in original_data_df.columns:
                print(f"錯誤: Excel檔案中找不到欄位 '{text_column}'。")
                return
            elif isinstance(text_column, int) and text_column >= len(original_data_df.columns):
                print(f"錯誤: Excel檔案中欄位索引 {text_column} 超出範圍。")
                return
            texts_series = original_data_df.iloc[:, text_column] if isinstance(text_column, int) else original_data_df[text_column]
        else:
            print(f"錯誤: 不支援的輸入檔案格式 '{input_format}'。")
            return
    except FileNotFoundError:
        print(f"錯誤: 輸入檔案 '{input_file}' 不存在。")
        return
    except Exception as e:
        print(f"讀取輸入檔案時發生錯誤: {e}")
        return

    if texts_series.empty:
        print("警告: 從輸入檔案中未讀取到任何文本數據。")
        return

    # 預處理文本
    segmented_texts = preprocess_text_for_prediction(texts_series)

    # 轉換為TF-IDF特徵
    print("正在將文本轉換為TF-IDF特徵...")
    X_predict = vectorizer.transform(segmented_texts)
    print("特徵轉換完成。")

    # 進行預測
    print("正在使用模型進行預測...")
    predictions = model.predict(X_predict)
    print("預測完成。")

    # 篩選有效資訊
    if input_format == 'txt':
        # 確保 texts_to_predict 和 predictions 長度一致
        if len(texts_to_predict) == len(predictions):
            effective_texts = [text for text, pred in zip(texts_to_predict, predictions) if pred == positive_label]
        else:
            print(f"錯誤: 原始文本數量 ({len(texts_to_predict)}) 與預測數量 ({len(predictions)}) 不匹配。")
            # Fallback or error handling
            effective_texts = [segmented_texts.iloc[i] for i, pred in enumerate(predictions) if pred == positive_label]
    else: # CSV/Excel
        # 創建布林遮罩
        mask = (predictions == positive_label)
        # 使用遮罩篩選原始 DataFrame
        effective_df = original_data_df[mask].copy() # 使用 .copy() 避免 SettingWithCopyWarning
        if effective_df.empty:
             print("模型判斷所有文本均為噪音，或沒有文本被判斷為有效資訊。")


    # 輸出結果
    print(f"正在將有效資訊寫入到輸出檔案: {output_file} (格式: {output_format})")
    try:
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"已創建輸出目錄: {output_dir}")

        if output_format == 'txt':
            with open(output_file, 'w', encoding='utf-8') as f:
                for text in effective_texts:
                    f.write(text + '\n')
            print(f"成功寫入 {len(effective_texts)} 條有效資訊到 {output_file}")
        elif output_format == 'csv':
            if not effective_df.empty:
                effective_df.to_csv(output_file, index=False, encoding='utf-8-sig') # utf-8-sig for Excel compatibility
                print(f"成功寫入 {len(effective_df)} 條有效資訊到 {output_file}")
            else:
                # 創建一個空的CSV檔案或者包含表頭的空CSV
                if original_data_df is not None:
                     pd.DataFrame(columns=original_data_df.columns).to_csv(output_file, index=False, encoding='utf-8-sig')
                print(f"沒有有效資訊可寫入，已創建空的CSV檔案 {output_file} (或僅含表頭)。")
        elif output_format == 'excel':
            if not effective_df.empty:
                effective_df.to_excel(output_file, index=False)
                print(f"成功寫入 {len(effective_df)} 條有效資訊到 {output_file}")
            else:
                if original_data_df is not None:
                    pd.DataFrame(columns=original_data_df.columns).to_excel(output_file, index=False)
                print(f"沒有有效資訊可寫入，已創建空的Excel檔案 {output_file} (或僅含表頭)。")
        else:
            print(f"錯誤: 不支援的輸出檔案格式 '{output_format}'。")

    except Exception as e:
        print(f"寫入輸出檔案時發生錯誤: {e}")


# --- 主函數，處理命令列參數 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用預訓練模型過濾文本檔案中的噪音。")
    parser.add_argument("input_file", help="包含原始文本的輸入檔案路徑。")
    parser.add_argument("output_file", help="過濾後只包含有效資訊的輸出檔案路徑。")
    parser.add_argument("--model_path", default="svm_noise_classifier.joblib",
                        help="預訓練模型的檔案路徑 (預設: svm_noise_classifier.joblib)。")
    parser.add_argument("--vectorizer_path", default="tfidf_vectorizer.joblib",
                        help="預訓練TF-IDF轉換器的檔案路徑 (預設: tfidf_vectorizer.joblib)。")
    parser.add_argument("--input_format", choices=['txt', 'csv', 'excel'], default='txt',
                        help="輸入檔案的格式 (預設: txt)。")
    parser.add_argument("--text_column",
                        help="如果輸入格式為 csv 或 excel，指定包含文本的欄位名稱或索引 (從0開始)。")
    parser.add_argument("--output_format", choices=['txt', 'csv', 'excel'], default=None,
                        help="輸出檔案的格式 (預設: 與輸入格式相同，若輸入為txt則輸出為txt)。")
    parser.add_argument("--positive_label", type=int, default=1,
                        help="模型中代表「有效資訊」的標籤 (預設: 1)。")

    args = parser.parse_args()

    # 如果未指定輸出格式，則預設與輸入格式相同
    if args.output_format is None:
        args.output_format = args.input_format


    # 如果輸入是CSV/Excel，text_column 可能是字串（欄位名）或數字（索引）
    text_col_processed = args.text_column
    if args.text_column is not None and args.text_column.isdigit():
        text_col_processed = int(args.text_column)


    filter_noise_from_file(args.input_file, args.output_file,
                           args.model_path, args.vectorizer_path,
                           input_format=args.input_format,
                           text_column=text_col_processed,
                           output_format=args.output_format,
                           positive_label=args.positive_label)

    print("\n處理完畢。")
