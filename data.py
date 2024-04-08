import pandas as pd
from datetime import datetime
from datasets import Dataset, Features, Value, Sequence, DatasetDict

PREDICT_LENGTH=365

def load_all_data():
    df = pd.read_excel('all_data.xlsx')
    first_date_str = df['Date'].iloc[0]
    # first_date = pd.to_datetime(first_date_str)
    first_date = datetime.strptime(str(first_date_str), "%Y-%m-%d %H:%M:%S")
    
    
    dataset = {}
    number_of_dates = df['Date'].notnull().sum()
    print(f"number of dates: {number_of_dates}")
    datas = []
    for col_name in df.columns:
        if col_name == "Date":
            continue
        datas.append(df[col_name].to_list())

    num_features = len(datas)
    print(f"The dataset has {num_features} features, each feature has {len(datas[0])} data points")
    train_data = test_data = valid_data =  {}
    num_train_row = number_of_dates - 2*PREDICT_LENGTH
    train_data = {
        'start': [first_date for _ in range(num_features)],
        'target':[data[:num_train_row] for data in datas],
        'feat_static_cat': [[i] for i in range(num_features)],
        'feat_dynamic_real': [None for _ in range(num_features)],
        'item_id': [f"T{i}" for i in range(num_features)]
    }

    num_test_row = number_of_dates - PREDICT_LENGTH
    test_data = {
        'start': [first_date for _ in range(num_features)],
        'target':[data[:num_test_row] for data in datas],
        'feat_static_cat': [[i] for i in range(num_features)],
        'feat_dynamic_real': [None for _ in range(num_features)],
        'item_id': [f"T{i}" for i in range(num_features)]
    }

    num_valid_row = number_of_dates
    valid_data = {
        'start': [first_date for _ in range(num_features)],
        'target':[data[:num_valid_row] for data in datas],
        'feat_static_cat': [[i] for i in range(num_features)],
        'feat_dynamic_real': [None for _ in range(num_features)],
        'item_id': [f"T{i}" for i in range(num_features)]
    }

    train_dataset = Dataset.from_dict(train_data)
    
    test_dataset = Dataset.from_dict(test_data)
    valid_dataset = Dataset.from_dict(valid_data)
    
    dataset = DatasetDict({
        'train': train_dataset,
        'test': test_dataset,
        'validation': valid_dataset
    })

    return dataset

if __name__ == '__main__':
    dataset = load_all_data()
    print(dataset)
