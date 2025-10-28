# pip install ucimlrepo pandas tqdm
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, balanced_accuracy_score, matthews_corrcoef
)




from __future__ import annotations
import os, re, json
from pathlib import Path

from ucimlrepo import fetch_ucirepo
import pandas as pd
from tqdm.auto import tqdm

# -------- CONFIG --------
OUT_DIR = Path(r"C:\Temp\UCI")
OUT_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_CSV = OUT_DIR / "uci_classif_categorical_list.csv"
# ------------------------

CAT_TOKENS = {
    "categorical", "nominal", "ordinal", "binary", "boolean", "string", "symbolic", "discrete"
}

def _norm(x):
    return str(x).strip().lower() if x is not None else ""

def _is_classification(meta) -> bool:
    try:
        tasks = getattr(meta, "task", None)
        if tasks is None:
            return False
        if isinstance(tasks, (list, tuple, set)):
            return any(_norm(t) == "classification" for t in tasks)
        return _norm(tasks) == "classification"
    except Exception:
        return False

def _count_categorical_from_variables(variables_df: pd.DataFrame | None) -> int:
    if variables_df is None or variables_df.empty:
        return 0
    cols = {c: _norm(c) for c in variables_df.columns}
    type_col = next(
        (c for c, n in cols.items() if n in {"type", "data_type", "data type", "datatype", "measurement"}),
        None
    )
    if type_col is None:
        return 0
    return int(variables_df[type_col].map(_norm).isin(CAT_TOKENS).sum())

def _has_any_categorical(meta, variables_df) -> tuple[bool, int]:
    # 1) Best: per-variable table
    cat_n = _count_categorical_from_variables(variables_df)
    if cat_n > 0:
        return True, cat_n
    # 2) Fallback: dataset-level feature_types
    try:
        ftypes = getattr(meta, "feature_types", None) or []
        ftypes = [_norm(t) for t in ftypes]
        if any(("categorical" in t) or (t in CAT_TOKENS) for t in ftypes):
            return True, 0
    except Exception:
        pass
    return False, 0

def _iter_dataset_ids():
    """
    Use ucimlrepo.list_available_datasets() if present (some versions print to stdout),
    else probe a safe ID range 1..1200.
    """
    ids: list[int] = []
    try:
        from ucimlrepo import list_available_datasets
        import io, contextlib, re as _re
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            list_available_datasets()
        text = buf.getvalue()
        ids = sorted({int(m.group(1)) for m in _re.finditer(r"^\s*(\d+)\b", text, flags=_re.MULTILINE)})
    except Exception:
        ids = []
    return ids if ids else range(1, 1201)

def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "-", s)
    return re.sub(r"-{2,}", "-", s).strip("-") or "dataset"

def _meta_to_jsonable(meta) -> dict:
    # Best-effort conversion
    try:
        from dataclasses import asdict, is_dataclass
        if is_dataclass(meta):
            return asdict(meta)
    except Exception:
        pass
    try:
        d = dict(meta.__dict__)
        # Make sure all values are JSON-serialisable
        json.dumps(d, default=str)
        return d
    except Exception:
        return {"repr": repr(meta)}

def _to_csv(df_or_array, path: Path, header_prefix="col"):
    try:
        if df_or_array is None:
            return
        if isinstance(df_or_array, pd.DataFrame):
            df_or_array.to_csv(path, index=False)
        else:
            # Convert numpy or other array-likes to DataFrame
            arr = pd.DataFrame(df_or_array)
            # Give simple headers if none
            if not isinstance(df_or_array, pd.DataFrame):
                arr.columns = [f"{header_prefix}{i}" for i in range(arr.shape[1])]
            arr.to_csv(path, index=False)
    except Exception:
        # Last resort: write via numpy.savetxt (only works for numeric/strings cleanly)
        try:
            import numpy as np
            np.savetxt(path, df_or_array, delimiter=",", fmt="%s")
        except Exception:
            pass

def _save_dataset(ds, ds_id: int, base_dir: Path):
    # Folder name: "0001-name"
    try:
        name = getattr(ds.metadata, "name", f"id-{ds_id}")
        folder = base_dir / f"{int(ds_id):04d}-{_slug(str(name))}"
    except Exception:
        folder = base_dir / f"{int(ds_id):04d}"
    folder.mkdir(parents=True, exist_ok=True)
    # Save X, y
    try:
        X = getattr(ds.data, "features", None)
        y = getattr(ds.data, "targets", None)
        if X is not None:
            _to_csv(X, folder / "X.csv", header_prefix="x")
        if y is not None:
            # y might be Series, DataFrame, or array; force single/ multi-column CSV
            if isinstance(y, pd.Series):
                y.to_frame("y").to_csv(folder / "y.csv", index=False)
            else:
                _to_csv(y, folder / "y.csv", header_prefix="y")
    except Exception:
        pass

    # Save variables schema if present
    try:
        vars_df = getattr(ds, "variables", None)
        if isinstance(vars_df, pd.DataFrame) and not vars_df.empty:
            vars_df.to_csv(folder / "variables.csv", index=False)
    except Exception:
        pass

    # Save metadata
    try:
        meta_dict = _meta_to_jsonable(getattr(ds, "metadata", {}))
        (folder / "metadata.json").write_text(json.dumps(meta_dict, indent=2, default=str), encoding="utf-8")
    except Exception:
        pass

def collect_and_save_classif_with_categorical() -> pd.DataFrame:
    rows = []
    for ds_id in tqdm(list(_iter_dataset_ids()), desc="Scanning UCI datasets"):
        try:
            ds = fetch_ucirepo(id=int(ds_id))
        except Exception:
            print(" No dataset ID", ds_id)
            continue  # invalid/missing id for this package

        meta = getattr(ds, "metadata", None)
        if not meta or not _is_classification(meta):
            print(" Not classification or no meta data for dataset ID", ds_id)
            continue

        try:
            has_cat, cat_count = _has_any_categorical(meta, getattr(ds, "variables", None))
        except Exception:
            print(" Exception checking categorical for dataset ID", ds_id)
            continue
        if not has_cat:
            print(" No categorical for dataset ID", ds_id)
            continue

        # Save to disk
        print("Saving dataset ID", ds_id, "to", OUT_DIR)

        _save_dataset(ds, ds_id=int(ds_id), base_dir=OUT_DIR)

        # Gather summary row
        try: name = meta.name
        except Exception: name = None
        try: tasks = meta.task
        except Exception: tasks = None
        try: feature_types = meta.feature_types
        except Exception: feature_types = None
        try:
            n_features = getattr(ds.data, "features", None)
            n_features = None if n_features is None else (n_features.shape[1] if hasattr(n_features, "shape") else None)
        except Exception:
            n_features = None

        rows.append({
            "id": int(ds_id),
            "name": name,
            "tasks": tasks,
            "feature_types": feature_types,
            "n_features": n_features,
            "categorical_feature_count_est": int(cat_count),  # 0 if only detected via feature_types
        })

    out = pd.DataFrame(rows).sort_values(["name", "id"]).reset_index(drop=True)
    out.to_csv(SUMMARY_CSV, index=False)
    return out

if __name__ == "__main__":
    df = collect_and_save_classif_with_categorical()
    print(f"Saved {len(df)} classification datasets with categorical attributes to: {OUT_DIR}")
    print(f"List written to: {SUMMARY_CSV}")
    with pd.option_context("display.max_colwidth", None):
        print(df.head(20))
