# data_cleaner_final.py
"""
DataCleaningEngine — Final (Cleaning-only) v1.0
Safe-by-default cleaning library with audit/reporting.
Dependencies: pandas, numpy, logging, typing
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

import pandas as pd
import numpy as np

# -----------------------------
# Utilities
# -----------------------------
def _ensure_dir(path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

def _safe_number_parse_series(s: pd.Series) -> pd.Series:
    """
    Best-effort locale-aware numeric parsing:
    - If values contain both '.' and ',', assume European format like '1.234,56'
    - Else assume US style and remove thousands commas.
    """
    try:
        sample = s.dropna().astype(str).head(1000)
        if sample.empty:
            return s
        both = sample.apply(lambda x: ('.' in x) and (',' in x)).sum()
        comma_only = sample.apply(lambda x: (',' in x) and ('.' not in x)).sum()
        dot_only = sample.apply(lambda x: ('.' in x) and (',' not in x)).sum()
        if both > 0 or (comma_only > dot_only and comma_only > len(sample) * 0.1):
            # European style: remove thousand dots, replace decimal comma with dot
            def parse_eu(x):
                try:
                    if pd.isna(x): return np.nan
                    t = str(x).strip()
                    t = t.replace('.', '').replace(',', '.')
                    return float(t)
                except Exception:
                    return np.nan
            return s.apply(parse_eu)
        else:
            # US style: remove commas
            def parse_us(x):
                try:
                    if pd.isna(x): return np.nan
                    t = str(x).strip().replace(',', '')
                    return float(t) if t not in ['', 'nan', 'None'] else np.nan
                except Exception:
                    return np.nan
            return s.apply(parse_us)
    except Exception:
        return s

# -----------------------------
# Main Cleaner
# -----------------------------
class DataCleaningEngine:
    """
    Enterprise-grade data cleaner (cleaning only).
    Usage:
        cleaner = DataCleaningEngine(...)
        clean_df, audit = cleaner.clean(df)
        cleaner.export_audit("reports/audit.json", fmt="json")
    """

    DEFAULT_COLUMN_CANDIDATES = {
        "sales": ["sales","revenue","amount","total","price","value","sales_amount"],
        "profit": ["profit","net_profit","profit_val","margin"],
        "price": ["price","unit_price","unitprice","unit_price"],
        "qty": ["qty","quantity","units","quantity_sold"],
        "order_id": ["order_id","orderid","id","transaction_id"],
        "customer_id": ["customer_id","customer","client_id","user_id","email"],
        "product": ["product","product_name","sku","item","product_id"],
        "category": ["category","cat","type","group"],
        "order_date": ["order_date","date","created_at","timestamp","orderdate"],
        "cogs": ["cogs","cost","cost_of_goods_sold","unit_cost"]
    }

    def __init__(
        self,
        missing_strategy: str = "auto",      # auto | drop | fill
        fill_numeric: str = "median",        # mean | median | zero
        fill_string: str = "mode",           # mode | empty
        fill_map: Optional[Dict[str, Any]] = None,
        drop_missing_threshold: float = 0.5,
        remove_duplicates: bool = True,
        detect_ids: bool = True,
        force_id_columns: Optional[List[str]] = None,
        id_unique_ratio: float = 0.98,
        detect_outliers: bool = True,
        outlier_method: str = "iqr",         # iqr | zscore
        zscore_threshold: float = 3.0,
        convert_to_category_threshold: int = 50,
        downcast_numeric: bool = True,
        convert_locale_numbers: bool = True,
        verbose: bool = True
    ):
        # Config
        self.missing_strategy = missing_strategy
        self.fill_numeric = fill_numeric
        self.fill_string = fill_string
        self.fill_map = fill_map or {}
        self.drop_missing_threshold = float(drop_missing_threshold)
        self.remove_duplicates = remove_duplicates
        self.detect_ids = detect_ids
        self.force_id_columns = set(force_id_columns or [])
        self.id_unique_ratio = float(id_unique_ratio)
        self.detect_outliers = detect_outliers
        self.outlier_method = outlier_method
        self.zscore_threshold = float(zscore_threshold)
        self.convert_to_category_threshold = convert_to_category_threshold
        self.downcast_numeric = downcast_numeric
        self.convert_locale_numbers = convert_locale_numbers

        # Logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            handler.setFormatter(logging.Formatter(fmt))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

        # Audit template
        self.audit: Dict[str, Any] = {
            "initial_shape": None,
            "final_shape": None,
            "rows_removed": 0,
            "columns_removed": 0,
            "duplicates_removed": 0,
            "column_types": {},        # col: type
            "numeric_converted": 0,
            "string_normalized": 0,
            "dates_parsed": 0,
            "missing_filled": 0,
            "dropped_columns": [],
            "outliers": {},            # col: count
            "warnings": [],
            "inferences": []
        }

    # -------------------------
    # Smart column detection
    # -------------------------
    def smart_find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}
        # exact
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        # partial
        for c in cols:
            cl = c.lower()
            for cand in candidates:
                if cand.lower() in cl:
                    return c
        return None

    def detect_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping = {}
        for key, candidates in self.DEFAULT_COLUMN_CANDIDATES.items():
            col = self.smart_find_column(df, candidates)
            if col:
                mapping[key] = col
        # honor forced id columns
        for forced in self.force_id_columns:
            if forced in df.columns:
                mapping["order_id" if "order" in forced.lower() else "customer_id"] = forced
                self.audit["inferences"].append(f"Forced id column '{forced}' added to mapping")
        return mapping

    # -------------------------
    # Column type detection
    # -------------------------
    def detect_column_type(self, s: pd.Series) -> str:
        non_null = s.dropna()
        if non_null.empty:
            return "string"
        # ID detection
        if self.detect_ids:
            try:
                unique_ratio = non_null.nunique() / len(non_null)
            except Exception:
                unique_ratio = 0.0
            if unique_ratio >= self.id_unique_ratio:
                try:
                    avg_len = non_null.astype(str).str.len().mean()
                except Exception:
                    avg_len = 0
                if avg_len >= 6:
                    return "id"
        # numeric dtype
        if pd.api.types.is_numeric_dtype(s):
            return "numeric"
        # date detection
        try:
            parsed = pd.to_datetime(non_null, errors="coerce")
            if parsed.notna().mean() >= 0.9:
                return "date"
        except Exception:
            pass
        # numeric-like strings
        try:
            cleaned = non_null.astype(str).str.replace(r"[,\s]", "", regex=True)
            if pd.to_numeric(cleaned, errors="coerce").notna().mean() >= 0.9:
                return "numeric"
        except Exception:
            pass
        return "string"

    # -------------------------
    # Missing handling
    # -------------------------
    def _apply_fill(self, df: pd.DataFrame, col: str, col_type: str) -> pd.DataFrame:
        # per-column override
        if col in self.fill_map:
            val = self.fill_map[col]
            before = df[col].isna().sum()
            df[col] = df[col].fillna(val)
            after = df[col].isna().sum()
            self.audit["missing_filled"] += (before - after)
            return df

        ratio = df[col].isna().mean()
        if self.missing_strategy == "auto" and ratio >= self.drop_missing_threshold:
            self.audit["dropped_columns"].append(col)
            self.audit["inferences"].append(f"Dropped column '{col}' (missing ratio {ratio:.2f})")
            return df.drop(columns=[col])

        if self.missing_strategy == "drop":
            before_rows = len(df)
            df = df.dropna(subset=[col])
            removed = before_rows - len(df)
            self.audit["rows_removed"] += removed
            return df

        # fill strategies
        before = df[col].isna().sum()
        if col_type == "numeric":
            if self.fill_numeric == "mean":
                df[col] = df[col].fillna(df[col].mean())
            elif self.fill_numeric == "median":
                df[col] = df[col].fillna(df[col].median())
            else:
                df[col] = df[col].fillna(0)
        elif col_type == "string":
            if self.fill_string == "mode":
                mode = df[col].mode(dropna=True)
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "")
            else:
                df[col] = df[col].fillna("")
        else:
            df[col] = df[col].fillna(np.nan)
        after = df[col].isna().sum()
        self.audit["missing_filled"] += (before - after)
        return df

    # -------------------------
    # Outliers detection (report only)
    # -------------------------
    def detect_outliers_iqr(self, series: pd.Series) -> int:
        try:
            s = pd.to_numeric(series.dropna(), errors="coerce")
            if s.empty:
                return 0
            Q1 = s.quantile(0.25); Q3 = s.quantile(0.75)
            IQR = Q3 - Q1
            if IQR == 0 or pd.isna(IQR):
                return 0
            lower = Q1 - 1.5 * IQR; upper = Q3 + 1.5 * IQR
            return int(((s < lower) | (s > upper)).sum())
        except Exception:
            return 0

    def detect_outliers_zscore(self, series: pd.Series) -> int:
        try:
            s = pd.to_numeric(series.dropna(), errors="coerce")
            if s.empty:
                return 0
            mean = s.mean(); std = s.std(ddof=0)
            if std == 0 or pd.isna(std):
                return 0
            z = ((s - mean).abs() / std)
            return int((z > self.zscore_threshold).sum())
        except Exception:
            return 0

    # -------------------------
    # Clean a single column
    # -------------------------
    def clean_column(self, df: pd.DataFrame, col: str, col_type: str) -> pd.DataFrame:
        if col not in df.columns:
            return df
        # dates
        if col_type == "date":
            before = df[col].notna().sum()
            try:
                df[col] = pd.to_datetime(df[col], errors="coerce")
            except Exception:
                df[col] = df[col].apply(lambda x: pd.to_datetime(x, errors="coerce") if pd.notna(x) else x)
            after = df[col].notna().sum()
            self.audit["dates_parsed"] += max(0, after - before)
        # numeric
        elif col_type == "numeric":
            # locale parsing if object
            if df[col].dtype == object and self.convert_locale_numbers:
                df[col] = _safe_number_parse_series(df[col])
            # numeric coercion
            before_non_na = df[col].notna().sum()
            df[col] = pd.to_numeric(df[col], errors="coerce")
            after_non_na = df[col].notna().sum()
            self.audit["numeric_converted"] += max(0, after_non_na - before_non_na)
            # outliers
            if self.detect_outliers:
                try:
                    cnt = self.detect_outliers_iqr(df[col]) if self.outlier_method == "iqr" else self.detect_outliers_zscore(df[col])
                    self.audit["outliers"][col] = int(cnt)
                except Exception:
                    self.audit["outliers"][col] = 0
        # string normalization
        elif col_type == "string":
            before_count = df[col].notna().sum()
            try:
                df[col] = df[col].astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
            except Exception:
                df[col] = df[col].apply(lambda x: " ".join(str(x).split()) if pd.notna(x) else x)
            after_count = df[col].notna().sum()
            self.audit["string_normalized"] += int(after_count)
        # apply missing handling
        return self._apply_fill(df, col, col_type)

    # -------------------------
    # Post-processing: categories, downcast
    # -------------------------
    def _post_process(self, df: pd.DataFrame) -> pd.DataFrame:
        # convert low-cardinality strings to category
        for col, t in self.audit["column_types"].items():
            if t == "string" and col in df.columns:
                try:
                    nunique = df[col].nunique(dropna=True)
                    if nunique <= self.convert_to_category_threshold:
                        df[col] = df[col].astype("category")
                        self.audit["inferences"].append(f"Converted '{col}' to category (nunique={nunique})")
                except Exception:
                    pass
        # downcast numeric
        if self.downcast_numeric:
            for c in df.select_dtypes(include=["number"]).columns:
                try:
                    if pd.api.types.is_float_dtype(df[c]):
                        df[c] = pd.to_numeric(df[c], downcast="float")
                    elif pd.api.types.is_integer_dtype(df[c]):
                        df[c] = pd.to_numeric(df[c], downcast="integer")
                except Exception:
                    pass
        return df

    # -------------------------
    # Export audit
    # -------------------------
    def export_audit(self, path: str, fmt: str = "json"):
        fmt = fmt.lower()
        _ensure_dir(path)
        if fmt == "json":
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.audit, f, ensure_ascii=False, indent=2, default=str)
        elif fmt == "csv":
            # Flatten column_types/outliers into rows for CSV
            rows = []
            cols = set(list(self.audit.get("column_types", {}).keys()) + list(self.audit.get("outliers", {}).keys()))
            for c in cols:
                rows.append({
                    "column": c,
                    "type": self.audit.get("column_types", {}).get(c, ""),
                    "outliers": self.audit.get("outliers", {}).get(c, ""),
                    "dropped": c in self.audit.get("dropped_columns", [])
                })
            pd.DataFrame(rows).to_csv(path, index=False)
        elif fmt == "txt":
            with open(path, "w", encoding="utf-8") as f:
                f.write(self.format_audit_report())
        else:
            raise ValueError("Unsupported format for export_audit. Use 'json', 'csv' or 'txt'.")

    # -------------------------
    # Human-readable audit text
    # -------------------------
    def format_audit_report(self) -> str:
        lines = []
        lines.append("===== CLEANING AUDIT REPORT =====")
        lines.append(f"Initial shape: {self.audit.get('initial_shape')}")
        lines.append(f"Final shape: {self.audit.get('final_shape')}")
        lines.append(f"Rows removed: {self.audit.get('rows_removed')}")
        lines.append(f"Columns removed: {self.audit.get('columns_removed')}")
        lines.append(f"Duplicates removed: {self.audit.get('duplicates_removed')}")
        lines.append("")
        lines.append("Column types detected:")
        for col, t in self.audit.get("column_types", {}).items():
            lines.append(f" - {col}: {t}")
        if self.audit.get("dropped_columns"):
            lines.append("")
            lines.append("Dropped columns (high missing ratio):")
            for c in self.audit["dropped_columns"]:
                lines.append(f" - {c}")
        if self.audit.get("outliers"):
            lines.append("")
            lines.append("Outliers (IQR/Z-score counts):")
            for col, cnt in self.audit["outliers"].items():
                lines.append(f" - {col}: {cnt}")
        if self.audit.get("warnings"):
            lines.append("")
            lines.append("Warnings:")
            for w in self.audit["warnings"]:
                lines.append(f" - {w}")
        if self.audit.get("inferences"):
            lines.append("")
            lines.append("Inferences (automations):")
            for i in self.audit["inferences"]:
                lines.append(f" - {i}")
        lines.append("===== END OF REPORT =====")
        return "\n".join(lines)

    # -------------------------
    # Execution summary print
    # -------------------------
    def print_execution_summary(self):
        print("\n===== CLEANING EXECUTION SUMMARY =====")
        print(f"Initial shape: {self.audit.get('initial_shape')}")
        print(f"Final shape:   {self.audit.get('final_shape')}")
        print(f"Rows removed:  {self.audit.get('rows_removed')}")
        print(f"Columns removed: {self.audit.get('columns_removed')}")
        print(f"Duplicates removed: {self.audit.get('duplicates_removed')}")
        print(f"Numeric values converted: {self.audit.get('numeric_converted')}")
        print(f"String values normalized: {self.audit.get('string_normalized')}")
        print(f"Dates parsed: {self.audit.get('dates_parsed')}")
        print(f"Missing values filled: {self.audit.get('missing_filled')}")
        print(f"Outliers detected (per column): {self.audit.get('outliers')}")
        if self.audit.get("warnings"):
            print("Warnings (sample):")
            for w in self.audit["warnings"][:10]:
                print(" -", w)
        print("===== END SUMMARY =====\n")

    # -------------------------
    # Main cleaning pipeline
    # -------------------------
    def clean(self, df: pd.DataFrame, inplace: bool = False) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        # work on copy by default
        target = df if inplace else df.copy()
        self.audit["initial_shape"] = target.shape

        # detect columns automatically
        mapping = self.detect_columns(target)
        # record detected column types
        for col in list(target.columns):
            try:
                t = self.detect_column_type(target[col])
            except Exception:
                t = "string"
                self.audit["warnings"].append(f"Type detection failed for '{col}' - defaulting to string")
            self.audit["column_types"][col] = t

        # cleaning loop over stable list of columns (avoid runtime change)
        for col in list(self.audit["column_types"].keys()):
            col_type = self.audit["column_types"].get(col, "string")
            try:
                target = self.clean_column(target, col, col_type)
            except Exception as e:
                self.audit["warnings"].append(f"Cleaning failed for '{col}': {e}")

        # remove duplicates if requested
        if self.remove_duplicates:
            try:
                dup_count = int(target.duplicated().sum())
                target = target.drop_duplicates()
                self.audit["duplicates_removed"] = dup_count
            except Exception:
                self.audit["warnings"].append("Duplicate removal failed")
                self.audit["duplicates_removed"] = 0
        else:
            self.audit["duplicates_removed"] = 0

        # drop fully empty rows
        try:
            before = len(target)
            target = target.dropna(how="all")
            removed = before - len(target)
            self.audit["rows_removed"] += removed
        except Exception:
            self.audit["warnings"].append("Dropping fully empty rows failed")

        # final post-processing
        try:
            target = self._post_process(target)
        except Exception:
            self.audit["warnings"].append("Post-processing failed")

        # final shapes and column counts
        self.audit["final_shape"] = target.shape
        self.audit["columns_removed"] = int(self.audit["initial_shape"][1] - self.audit["final_shape"][1])
        # ensure integers
        self.audit["rows_removed"] = int(self.audit["rows_removed"])
        self.audit["columns_removed"] = int(self.audit["columns_removed"])

        # print summary to stdout
        self.print_execution_summary()

        return target, self.audit

# End of file

    