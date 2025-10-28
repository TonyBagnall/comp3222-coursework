from provided_code.data_loaders import load_tabular_xy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute accuracy given true and predicted labels.
    """
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return correct / len(y_true)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Compute confusion matrix given true and predicted labels.
    """
    unique_labels = np.unique(np.concatenate((y_true, y_pred)))
    n_labels = unique_labels.size
    cm = np.zeros((n_labels, n_labels), dtype=int)
    label_to_index = {label: index for index, label in enumerate(unique_labels)}

    for true, pred in zip(y_true, y_pred):
        i = label_to_index[true]
        j = label_to_index[pred]
        cm[i, j] += 1

    return cm

if __name__ == "__main__":    # Example usage
    name = "balloons"
    X, y = load_tabular_xy("../data/" + name + "/" + name + ".data")
    print(X,y)
    enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    X_ohe = enc.fit_transform(X)
    trainX,  testX, trainy, testy = train_test_split(X_ohe, y, test_size=0.5625,
                                                     random_state=42)
    print("TRAIN")
    print(trainX,trainy)
    print("TEST")
    print(testX,testy)
    ada = AdaBoostClassifier()
    ada.fit(trainX, trainy)
    rf = RandomForestClassifier()
    rf.fit(trainX, trainy)
    preds_ada = ada.predict(testX)
    probs_ada = ada.predict_proba(testX)
    preds_rf = rf.predict(testX)
    probs_rf = rf.predict_proba(testX)

    print("AdaBoost Predictions:", preds_ada)
    print("AdaBoost Probabilities:", probs_ada)
    print("Random Forest Predictions:", preds_rf)
    print("Random Forest Probabilities:", probs_rf)
    print(testy)
    s1 = ada.score(testX, testy)
    s2 = rf.score(testX, testy)
    print(f"AdaBoost score: {s1:.4f}")
    print(f"Random Forest score: {s2:.4f}")
    print(f"AdaBoost train set accuracy: {accuracy(trainy, ada.predict(trainX)):.4f}")
    print(f"Random Forest score: {accuracy(trainy,rf.predict(trainX)):.4f}")

    # diff = 0
    # for name in dirs:
    #     print(name+":", type(X), X.shape, y.shape, "Unique y:", set(y))
    #     ada = DecisionTreeClassifier()
    #     ada.fit(trainX, trainy)
    #     acc = ada.score(testX, testy)
    #     randf=RandomForestClassifier()
    #     randf.fit(trainX, trainy)
    #     accf=randf.score(testX, testy)
    #     print(f"  Decision tree accuracy: {acc:.4f} Random Foresrt accuracy: {accf:.4f}")
    #     diff = diff + accf - acc
    # print("Average difference in acc:", diff/len(dirs))