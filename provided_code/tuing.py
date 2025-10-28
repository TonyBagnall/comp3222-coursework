import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, classification_report
from aeon.datasets import load_classification, load_gunpoint

trainX, trainy = load_gunpoint(split="train", return_type="numpy2D")
testX, testy = load_gunpoint(split="test", return_type="numpy2D")


# X, y assumed ready (numpy arrays). If you have a test set, split first:
# X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


def tune_svm(trainX, trainy):
    pipe = Pipeline([
        ("scaler", StandardScaler()),         # essential for SVMs
        ("clf", SVC())
    ])
    # Two grids so we don't pass 'gamma' to linear kernel
    param_grid = [
        {"clf__kernel": ["rbf"],
         "clf__C": [0.01, 0.1, 1, 10, 100],
         "clf__gamma": ["scale", "auto", 1e-3, 1e-2, 1e-1]}
    ]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1
    )

    search.fit(trainX, trainy)  # or fit(X_tr, y_tr) if you held out a test set
    print("Best params:", search.best_params_)
    print("Best CV accuracy:", search.best_score_)
    best_model = search.best_estimator_
    return best_model


def gunpoint_example():
    trainX, trainy = load_gunpoint( split="train", return_type="numpy2D")
    testX, testy = load_gunpoint(split="test", return_type="numpy2D")
    best_svm = tune_svm(trainX, trainy)
    preds = best_svm.predict(testX)
    acc = accuracy_score(testy, preds)
    print("Test accuracy of best SVM:", acc)
    print("Classification report:\n", classification_report(testy, preds))
    pipe = Pipeline([
        ("scaler", StandardScaler()),         # essential for SVMs
        ("clf", SVC())
    ])
    pipe.fit(trainX, trainy)
    preds_no_tune = pipe.predict(testX)
    acc_no_tune = accuracy_score(testy, preds_no_tune)
    print("Test accuracy of default SVM:", acc_no_tune)
    print("Classification report:\n", classification_report(testy, preds))


if __name__ == "__main__":
    gunpoint_example()

