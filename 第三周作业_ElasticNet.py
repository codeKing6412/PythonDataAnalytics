import numpy as np
import pandas as pd

from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    train_test_split
)
from sklearn.preprocessing import StandardScaler


# Kaggle 原始数据：
# https://www.kaggle.com/datasets/arnavsmayan/netflix-userbase-dataset
# 这里使用同一份数据的 GitHub 镜像，方便直接运行
DATA_URL = (
    "https://raw.githubusercontent.com/"
    "TomasBoda/statistical-project/main/netflix_userbase.csv"
)

df = pd.read_csv(DATA_URL)

print("data shape:", df.shape)
print(df.head())
print("\nmissing values:")
print(df.isnull().sum())


# 日期转换
df["Join Date"] = pd.to_datetime(
    df["Join Date"],
    dayfirst=True
)

df["Last Payment Date"] = pd.to_datetime(
    df["Last Payment Date"],
    dayfirst=True
)

# 用两个日期得到一个比较容易使用的数值特征
df["Tenure Days"] = (
    df["Last Payment Date"] - df["Join Date"]
).dt.days


# Monthly Revenue 作为预测目标
features = [
    "Subscription Type",
    "Country",
    "Age",
    "Gender",
    "Device",
    "Tenure Days"
]

X = df[features].copy()
y = df["Monthly Revenue"].copy()

# 分类变量转成 dummy
X = pd.get_dummies(
    X,
    drop_first=True,
    dtype=float
)

print("\nfeature number:", X.shape[1])
print(X.head())


# 训练集 / 测试集
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Elastic Net 对变量尺度比较敏感，所以先标准化
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 使用网格搜索选择 alpha 和 l1_ratio
enet = ElasticNet(
    max_iter=20000
)

param_grid = {
    "alpha": [
        0.001,
        0.003,
        0.01,
        0.03,
        0.1,
        0.3,
        1.0
    ],
    "l1_ratio": [
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
        1.0
    ]
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

grid = GridSearchCV(
    enet,
    param_grid,
    scoring="neg_mean_squared_error",
    cv=cv,
    n_jobs=-1
)

grid.fit(
    X_train_scaled,
    y_train
)

best_model = grid.best_estimator_

print("\nbest params:")
print(grid.best_params_)


# 测试集预测
y_pred = best_model.predict(
    X_test_scaled
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\nmodel result:")
print("MAE :", round(mae, 4))
print("RMSE:", round(rmse, 4))
print("R2  :", round(r2, 4))


# 看一下 Monthly Revenue 的分组情况
print("\nmean revenue by subscription:")
print(
    df.groupby("Subscription Type")
      ["Monthly Revenue"]
      .mean()
      .round(3)
)


# 系数
coef_table = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": best_model.coef_
})

coef_table["AbsCoefficient"] = (
    coef_table["Coefficient"].abs()
)

coef_table = coef_table.sort_values(
    "AbsCoefficient",
    ascending=False
)

print("\ncoefficients:")
print(
    coef_table.to_string(index=False)
)

non_zero = (
    coef_table["Coefficient"].abs()
    > 1e-8
).sum()

print(
    "\nnon-zero coefficient number:",
    non_zero
)
