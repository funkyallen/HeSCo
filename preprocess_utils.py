import os
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
def preprocess_data(X_lbl, X_unl, X_test, cat_threshold=10):
    """
    Preprocesses data for NN and Tree-based models without data leakage.
    Uses ColumnTransformer for robust and scalable feature engineering.
    """
    # 1. 统一转换为 DataFrame 以便基于列名操作
    if not isinstance(X_lbl, pd.DataFrame):
        X_lbl = pd.DataFrame(X_lbl)
    X_unl = pd.DataFrame(X_unl, columns=X_lbl.columns) if not isinstance(X_unl, pd.DataFrame) else X_unl.copy()
    X_test = pd.DataFrame(X_test, columns=X_lbl.columns) if not isinstance(X_test, pd.DataFrame) else X_test.copy()
    
    # 确保类别列在拟合前统一转换为字符串，避免 SimpleImputer(fill_value='Missing') 在数值列上报错
    # 我们先在备份上操作
    X_lbl = X_lbl.copy()
    X_unl = X_unl.copy()
    X_test = X_test.copy()
    
    # 拼接训练集用于拟合统计量 (防止特征泄露)
    X_train = pd.concat([X_lbl, X_unl], axis=0, ignore_index=True)

    # 2. 剔除常数列 (Constant Features)
    constant_cols = [col for col in X_train.columns if X_train[col].nunique(dropna=True) <= 1]
    if constant_cols:
        X_train.drop(columns=constant_cols, inplace=True)
        X_lbl.drop(columns=constant_cols, inplace=True)
        X_unl.drop(columns=constant_cols, inplace=True)
        X_test.drop(columns=constant_cols, inplace=True)

    # 3. 特征分类推断逻辑 (Feature Typing)
    obj_cols = set(X_train.select_dtypes(include=['object', 'category', 'bool']).columns)
    int_cols = X_train.select_dtypes(include=['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8']).columns
    
    # 将 unique 值较少的整数列也视为类别特征
    for col in int_cols:
        if X_train[col].nunique(dropna=True) <= cat_threshold:
            obj_cols.add(col)

    cat_cols = list(obj_cols)
    num_cols = [c for c in X_train.columns if c not in cat_cols]

    # 重要：将所有推断出的类别列转换为字符串类型，否则 SimpleImputer 的 'Missing' 填充会报错
    for col in cat_cols:
        X_lbl[col] = X_lbl[col].astype(str)
        X_unl[col] = X_unl[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    
    # 更新 X_train 以便拟合
    X_train = pd.concat([X_lbl, X_unl], axis=0, ignore_index=True)

    # 进一步区分低基数(Low Cardinality)和高基数(High Cardinality)类别特征
    low_card_cats = [col for col in cat_cols if X_train[col].nunique(dropna=True) <= cat_threshold]
    high_card_cats = [col for col in cat_cols if col not in low_card_cats]

    # 4. 构建 sklearn Pipelines (核心逻辑)
    
    # 数值型：中位数填充 -> 标准化
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 低基数类别：常量填充 -> One-Hot 编码
    # sparse_output=False 确保输出密集矩阵，适配 NN
    cat_low_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # 高基数类别：常量填充 -> Ordinal 编码
    # 注意：这里编码后输出的是整数索引，不再经过 StandardScaler！
    cat_high_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    # 5. 组合 ColumnTransformer
    transformers = []
    if num_cols:
        transformers.append(('num', num_pipeline, num_cols))
    if low_card_cats:
        transformers.append(('cat_low', cat_low_pipeline, low_card_cats))
    if high_card_cats:
        transformers.append(('cat_high', cat_high_pipeline, high_card_cats))

    preprocessor = ColumnTransformer(transformers=transformers, remainder='drop', n_jobs=None)

    # 6. 拟合与转换 (Fit on Train, Transform on All)
    preprocessor.fit(X_train)
    
    X_lbl_out = preprocessor.transform(X_lbl)
    X_unl_out = preprocessor.transform(X_unl)
    X_test_out = preprocessor.transform(X_test)

    return X_lbl_out, X_unl_out, X_test_out


def load_data(path):
    """
    Loads data from a CSV file.
    
    Args:
        path (str): Path to the CSV file.
        
    Returns:
        X (pd.DataFrame): Features.
        y (np.array): Target.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found at {path}")
    
    try:
        df = pd.read_csv(path)
    except:
        df = pd.read_csv(path, header=None)

    target_col = df.columns[-1]
    X = df.drop(columns=[target_col])
    y = df[target_col].values

    # Ensure y is numeric
    if not np.issubdtype(y.dtype, np.number):
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = y.astype(float)
        
    return X, y