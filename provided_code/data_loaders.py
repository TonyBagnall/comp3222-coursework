from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

import re
import numpy as np
from typing import Tuple, Optional, List
import pandas as pd

dirs = [
    "balance-scale",
    "chess-krvk",
    "chess-krvkp",
    "connect-4",
    "contraceptive-method",
    "fertility",
    "habermans-survival",
    "hayes-roth",
    "led-display",
    "lymphography",
    "molecular-promoters",
    "molecular-splice",
    "monks-1",
    "monks-2",
    "monks-3",
    "nursery",
    "optdigits",
    "pendigits",
    "semeion",
    "spect-heart",
    "tic-tac-toe",
    "zoo",
]

def load_tabular_xy(
    path: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a simple delimited file into (X, y) with *all* values as strings.
    - Rows: cases
    - Columns: features + target (last column)

    Parameters
    ----------
    path : str
        File path.
    """
    X_rows = []
    y_vals = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            sep = ","
            parts = line.split(sep)

            # Drop trailing empty from lines ending with a delimiter
            if parts and parts[-1] == "":
                parts = parts[:-1]
            # Ensure every cell is a string
            cells = [p.strip() for p in parts]
            idx = len(cells) - 1
            y_vals.append(cells[idx])
            X_rows.append(cells[:idx] + cells[idx+1:])

    X = np.asarray(X_rows, dtype=str)
    y = np.asarray(y_vals, dtype=str)
    return X, y


if __name__ == "__main__":    # Example usage
    print(len(dirs))
    diff = 0
    for name in dirs:
        X, y= load_tabular_xy("../data/"+name+"/"+name+".data")
        print(name+":", type(X), X.shape, y.shape, "Unique y:", set(y))
        dt = DecisionTreeClassifier()
        enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_ohe = enc.fit_transform(X)
        trainX,  testX,trainy, testy = train_test_split(X_ohe, y, test_size=0.3,
                                                         random_state=42)
        dt.fit(trainX, trainy)
        acc = dt.score(testX, testy)
        randf=RandomForestClassifier()
        randf.fit(trainX, trainy)
        accf=randf.score(testX, testy)
        print(f"  Decision tree accuracy: {acc:.4f} Random Foresrt accuracy: {accf:.4f}")
        diff = diff + accf - acc
    print("Average difference in acc:", diff/len(dirs))