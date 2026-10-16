"""第四周作业：SVM 对 UCI Spambase 数据集进行垃圾邮件分类。
运行：pip install pandas numpy scikit-learn
      python 第四周作业_SVM.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path(__file__).resolve().parent / "第四周_spambase.data"


def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    print(f"\n===== {name} =====")
    print("Accuracy:", round(accuracy_score(y_test, pred), 4))
    print("Precision(垃圾):", round(precision_score(y_test, pred, zero_division=0), 4))
    print("Recall(垃圾):", round(recall_score(y_test, pred, zero_division=0), 4))
    print("F1(垃圾):", round(f1_score(y_test, pred, zero_division=0), 4))
    print("混淆矩阵 [正常, 垃圾]:")
    print(confusion_matrix(y_test, pred, labels=[0, 1]))
    print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(classification_report(y_test, pred, target_names=["正常邮件", "垃圾邮件"], digits=4))
    return {"accuracy": accuracy_score(y_test, pred), "f1": f1_score(y_test, pred)}


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"缺少数据文件：{DATA_PATH}；请把仓库中的第四周_spambase.data 和本脚本放在同一目录。")

    data = pd.read_csv(DATA_PATH, header=None)
    if data.shape[1] != 58:
        raise ValueError(f"预期为 57 个特征+1 个标签，实际读取到 {data.shape[1]} 列。")
    data.columns = [f"feature_{i:02d}" for i in range(1, 58)] + ["spam"]
    print("数据形状:", data.shape)
    print("缺失值总数:", data.isna().sum().sum())
    print("类别计数（0正常，1垃圾）:")
    print(data["spam"].value_counts().sort_index())

    X = data.drop(columns=["spam"])
    y = data["spam"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("训练集大小:", len(X_train), "测试集大小:", len(X_test))
    print("测试集标签数量:", y_test.value_counts().sort_index().to_dict())

    # 标准化要放在 Pipeline 内部，这样交叉验证中不会提前看到验证折的数据。
    linear_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="linear", C=1.0)),
    ])
    linear_svm.fit(X_train, y_train)
    linear_result = evaluate("线性 SVM (C=1)", linear_svm, X_test, y_test)
    print("线性 SVM 支持向量数量:", linear_svm.named_steps["svc"].n_support_.tolist())

    # 课件提到了 RBF 和 C、gamma；只在训练集做交叉验证，不用测试集调参。
    rbf_svm = Pipeline([
        ("scaler", StandardScaler()),
        ("svc", SVC(kernel="rbf")),
    ])
    params = {
        "svc__C": [0.1, 1, 10, 100],
        "svc__gamma": [0.001, 0.01, 0.1],
    }
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    search = GridSearchCV(
        rbf_svm, params, scoring="f1", cv=folds, n_jobs=-1
    )
    search.fit(X_train, y_train)
    print("\nRBF 最优参数:", search.best_params_)
    print("训练集 5 折 CV 最佳 F1:", round(search.best_score_, 4))
    rbf_result = evaluate("RBF SVM（网格搜索后）", search.best_estimator_, X_test, y_test)
    print("RBF SVM 支持向量数量:", search.best_estimator_.named_steps["svc"].n_support_.tolist())

    print("\n===== 小结 =====")
    print(f"线性 SVM: Accuracy={linear_result['accuracy']:.4f} F1={linear_result['f1']:.4f}")
    print(f"RBF  SVM: Accuracy={rbf_result['accuracy']:.4f} F1={rbf_result['f1']:.4f}")
    print("F1 更关注垃圾邮件识别的精确率和召回率，两者需要结合混淆矩阵理解。")


if __name__ == "__main__":
    main()
