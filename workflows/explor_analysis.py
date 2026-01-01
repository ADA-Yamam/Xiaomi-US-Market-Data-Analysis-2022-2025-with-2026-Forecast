# exploratory_eda.py
"""
Exploratory EDA Module (Final, English Version)
- Fully independent from cleaning.
- Intelligent in detecting columns (sales, profit, date, product, customer, city, category,...).
- Outputs: text/JSON/HTML reports + PNG figures for each main analysis.
- Uses matplotlib for plotting only.
- Safe: does not modify the original DataFrame.
- Warnings and notes are stored in audit.
"""

import os
import json
import math
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------
# Helpers
# ---------------------------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _fmt_num(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

def _safe_to_numeric(s):
    try:
        return pd.to_numeric(s, errors="coerce")
    except Exception:
        return s

# ---------------------------
# ExploratoryAnalyzer class
# ---------------------------
class ExploratoryAnalyzer:
    DEFAULT_MAPPING = {
        "sales": ["sales","revenue","amount","total","price","value","sales_amount"],
        "profit": ["profit","profit_val","net_profit","profit_amount"],
        "order_date": ["order_date","date","created_at","timestamp","orderdate","shipdate","ship_date"],
        "customer_id": ["customer_id","customer","client_id","user_id","email","buyer"],
        "order_id": ["order_id","orderid","id","transaction_id"],
        "product": ["product","product_name","sku","item","product_id"],
        "category": ["category","cat","type","group"],
        "city": ["city","town","location","city_name","region"]
    }

    def __init__(self, output_dir: str = "eda_reports", plot: bool = True, max_categories: int = 20, verbose: bool = True):
        self.output_dir = output_dir
        _ensure_dir(self.output_dir)
        self.fig_dir = os.path.join(self.output_dir, "figures")
        _ensure_dir(self.fig_dir)
        self.plot = plot
        self.max_categories = max_categories
        self.verbose = verbose
        self.audit: Dict[str, Any] = {
            "mapping": {},
            "warnings": [],
            "inferences": [],
            "plots": [],
            "stats": {}
        }

    # ---------------------------
    # Smart column finder
    # ---------------------------
    def _smart_find(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}
        for cand in candidates:  # exact match
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        for c in cols:  # partial match
            cl = c.lower()
            for cand in candidates:
                if cand.lower() in cl:
                    return c
        return None

    def auto_map(self, df: pd.DataFrame) -> Dict[str,str]:
        mapping = {}
        for key, candidates in self.DEFAULT_MAPPING.items():
            col = self._smart_find(df, candidates)
            if col:
                mapping[key] = col
        self.audit["mapping"] = mapping
        return mapping

    # ---------------------------
    # Save plot helper
    # ---------------------------
    def _save_plot(self, fig, name: str) -> Optional[str]:
        try:
            path = os.path.join(self.fig_dir, f"{name}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            self.audit["plots"].append(path)
            return path
        except Exception as e:
            self.audit["warnings"].append(f"save_plot_failed({name}): {e}")
            try: plt.close(fig)
            except: pass
            return None

    # ---------------------------
    # Main EDA runner
    # ---------------------------
    def run(self, df: pd.DataFrame, show_plots: bool = True, mapping: Optional[Dict[str,str]] = None) -> Dict[str,Any]:
        """
        df: input DataFrame (not modified)
        show_plots: whether to display plots in notebooks
        mapping: optional manual column mapping
        Returns a dict with all analysis results, file paths, and audit
        """
        results: Dict[str, Any] = {
            "basic": {}, "missing": {}, "descriptive": {}, "categorical": {}, 
            "numeric_distributions": {}, "bivariate": {}, "correlation": {},
            "time": {}, "outliers": {}, "executive": {}, "files": [], "audit": self.audit
        }

        try: df0 = df.copy()
        except: df0 = df

        # Mapping
        if mapping:
            self.audit["mapping"] = mapping
        else:
            mapping = self.auto_map(df0)
        results["mapping"] = mapping

        # 1. Basic info
        try:
            results["basic"]["shape"] = df0.shape
            results["basic"]["columns"] = list(df0.columns)
            results["basic"]["dtypes"] = df0.dtypes.apply(lambda x: str(x)).to_dict()
            if self.verbose:
                print("\n--- BASIC INFO ---")
                df0.info()
                print(f"Shape: {df0.shape}")
        except Exception as e:
            self.audit["warnings"].append(f"basic_info_failed: {e}")

        # 2. Missing values
        try:
            miss = df0.isnull().sum()
            miss_pct = (miss / len(df0)) * 100
            miss_tbl = pd.DataFrame({"missing_count": miss, "missing_pct": miss_pct})
            miss_tbl = miss_tbl[miss_tbl["missing_count"]>0].sort_values("missing_pct", ascending=False)
            results["missing"] = miss_tbl.to_dict(orient="index")
            if self.verbose:
                print("\n--- MISSING VALUES ---")
                print(miss_tbl if not miss_tbl.empty else "No missing values.")
        except Exception as e:
            self.audit["warnings"].append(f"missing_calc_failed: {e}")

        # 3. Descriptive statistics
        try:
            desc = df0.describe(include="all", percentiles=[0.01,0.05,0.95,0.99]).T
            results["descriptive"] = desc.fillna("").to_dict()
            if self.verbose:
                print("\n--- DESCRIPTIVE STATISTICS ---")
                print(desc)
        except Exception as e:
            self.audit["warnings"].append(f"desc_stats_failed: {e}")

        # 4. Categorical summary
        try:
            cat_cols = df0.select_dtypes(include=["object","category"]).columns.tolist()
            cat_summary = {}
            for c in cat_cols:
                try:
                    vc = df0[c].value_counts(dropna=False)
                    cat_summary[c] = {
                        "n_unique": int(df0[c].nunique(dropna=True)),
                        "top_values": vc.head(self.max_categories).to_dict()
                    }
                    if self.plot:
                        fig = plt.figure(figsize=(8,4))
                        vc.dropna().head(self.max_categories).plot(kind="bar", title=f"Value counts: {c}")
                        self._save_plot(fig, f"cat_{c}")
                except Exception as e:
                    self.audit["warnings"].append(f"cat_summary_failed_{c}: {e}")
            results["categorical"] = cat_summary
        except Exception as e:
            self.audit["warnings"].append(f"categorical_failed: {e}")

        # 5. Numeric distributions + histograms + boxplots
        try:
            num_cols = df0.select_dtypes(include=["number"]).columns.tolist()
            num_summary = {}
            for c in num_cols:
                try:
                    series = df0[c].dropna()
                    num_summary[c] = {
                        "count": int(series.count()),
                        "mean": float(series.mean()) if not series.empty else None,
                        "std": float(series.std()) if not series.empty else None,
                        "min": float(series.min()) if not series.empty else None,
                        "25%": float(series.quantile(0.25)) if not series.empty else None,
                        "50%": float(series.median()) if not series.empty else None,
                        "75%": float(series.quantile(0.75)) if not series.empty else None,
                        "max": float(series.max()) if not series.empty else None,
                        "skew": float(series.skew()) if not series.empty else None,
                        "kurt": float(series.kurt()) if not series.empty else None
                    }
                    if self.plot:
                        fig = plt.figure(figsize=(6,3))
                        plt.hist(series, bins=30); plt.title(f"Distribution of {c}")
                        self._save_plot(fig, f"hist_{c}")
                        fig2 = plt.figure(figsize=(4,3))
                        plt.boxplot(series.dropna(), vert=False); plt.title(f"Boxplot of {c}")
                        self._save_plot(fig2, f"box_{c}")
                except Exception as e:
                    self.audit["warnings"].append(f"numeric_summary_failed_{c}: {e}")
            results["numeric_distributions"] = num_summary
        except Exception as e:
            self.audit["warnings"].append(f"numeric_failed: {e}")

        # 6. Bivariate analysis (scatter + box by category)
        try:
            bivar = {}
            if len(num_cols) >= 2:
                x = num_cols[0]; y = num_cols[1]
                bivar["scatter"] = {"x": x, "y": y}
                if self.plot:
                    fig = plt.figure(figsize=(6,4))
                    plt.scatter(df0[x], df0[y], alpha=0.5, s=10)
                    plt.xlabel(x); plt.ylabel(y); plt.title(f"{x} vs {y}")
                    self._save_plot(fig, f"scatter_{x}_{y}")
            if cat_cols and num_cols:
                grp_col = cat_cols[0]; num_col = num_cols[0]
                bivar["box_by_cat"] = {"cat": grp_col, "num": num_col}
                if self.plot:
                    fig = plt.figure(figsize=(8,4))
                    df0.boxplot(column=num_col, by=grp_col); plt.title(f"{num_col} by {grp_col}"); plt.suptitle("")
                    self._save_plot(fig, f"box_by_{grp_col}_{num_col}")
            results["bivariate"] = bivar
        except Exception as e:
            self.audit["warnings"].append(f"bivariate_failed: {e}")

        # 7. Correlation
        try:
            if len(num_cols) > 1:
                corr = df0[num_cols].corr()
                results["correlation"] = corr.fillna(0).to_dict()
                if self.plot:
                    fig = plt.figure(figsize=(8,6))
                    plt.matshow(corr, cmap="coolwarm", fignum=1)
                    plt.xticks(range(len(num_cols)), num_cols, rotation=90)
                    plt.yticks(range(len(num_cols)), num_cols)
                    plt.colorbar(); plt.title("Correlation Heatmap", pad=20)
                    self._save_plot(fig, "correlation_heatmap")
        except Exception as e:
            self.audit["warnings"].append(f"correlation_failed: {e}")

        # 8. Time series analysis
        try:
            date_col = mapping.get("order_date")
            sales_col = mapping.get("sales")
            if date_col and date_col in df0.columns:
                df0[date_col] = pd.to_datetime(df0[date_col], errors="coerce")
            if not date_col:
                for c in df0.columns:
                    if "date" in c.lower() or "time" in c.lower() or "ship" in c.lower():
                        try:
                            if pd.to_datetime(df0[c].dropna().head(50), errors="coerce").notna().sum() > 0:
                                date_col = c
                                mapping["order_date"] = c
                                self.audit["inferences"].append(f"inferred order_date from column '{c}'")
                                break
                        except: continue
            if date_col and date_col in df0.columns:
                df_ts = df0.dropna(subset=[date_col]).copy()
                if sales_col and sales_col in df_ts.columns:
                    df_ts[sales_col] = _safe_to_numeric(df_ts[sales_col])
                    df_ts["__month"] = df_ts[date_col].dt.to_period("M").astype(str)
                    monthly = df_ts.groupby("__month")[sales_col].sum().sort_index()
                    results["time"]["monthly"] = monthly.to_dict()
                    if self.plot:
                        fig = plt.figure(figsize=(9,4))
                        monthly.plot(marker="o", title="Monthly Sales Trend")
                        self._save_plot(fig, "monthly_sales_trend")
                    yearly = df_ts.groupby(df_ts[date_col].dt.year)[sales_col].sum().sort_index()
                    results["time"]["yearly"] = yearly.to_dict()
        except Exception as e:
            self.audit["warnings"].append(f"time_series_failed: {e}")

        # 9. Outliers (IQR)
        try:
            outliers = {}
            for c in num_cols:
                s = df0[c].dropna()
                Q1 = s.quantile(0.25); Q3 = s.quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5*IQR; upper = Q3 + 1.5*IQR
                outliers[c] = int(((df0[c] < lower) | (df0[c] > upper)).sum()) if not s.empty else 0
            results["outliers"] = outliers
        except Exception as e:
            self.audit["warnings"].append(f"outliers_failed: {e}")

        # 10. Executive summary
        try:
            exec_summary = {}
            if sales_col and sales_col in df0.columns:
                total_sales = _safe_to_numeric(df0[sales_col]).sum()
                exec_summary["total_sales"] = float(total_sales)
                exec_summary["avg_sales"] = float(_safe_to_numeric(df0[sales_col]).mean())
            if "profit" in mapping and mapping["profit"] in df0.columns:
                total_profit = _safe_to_numeric(df0[mapping["profit"]]).sum()
                exec_summary["total_profit"] = float(total_profit)
                exec_summary["avg_profit"] = float(_safe_to_numeric(df0[mapping["profit"]]).mean())
            results["executive"] = exec_summary
        except Exception as e:
            self.audit["warnings"].append(f"executive_failed: {e}")

        # 11. Save reports
        try:
            # text
            text_path = os.path.join(self.output_dir, "eda_report.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(results, ensure_ascii=False, indent=2))
            results["text_path"] = text_path

            # json
            json_path = os.path.join(self.output_dir, "eda_report.json")
            with open(json_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(results, default=str, ensure_ascii=False, indent=2))
            results["json_path"] = json_path

            # simple html
            html_path = os.path.join(self.output_dir, "eda_report.html")
            with open(html_path, "w", encoding="utf-8") as f:
                f.write("<html><head><meta charset='utf-8'><title>EDA Report</title></head><body>")
                f.write("<h1>EDA Report</h1>")
                f.write("<pre>{}</pre>".format(json.dumps(results, ensure_ascii=False, indent=2)))
                f.write("</body></html>")
            results["html_path"] = html_path
        except Exception as e:
            self.audit["warnings"].append(f"report_write_failed: {e}")

        return results

# ---------------------------
# Demo usage
# ---------------------------
if __name__ == "__main__":
    print("EDA demo. Reports output in ./eda_reports_demo")
    df_demo = pd.DataFrame({
        "OrderID": [1,1,2,3,4,5,5],
        "Ship Date": ["2023-01-10","2023-01-10","2023-02-15","2023-02-20","2023-03-05","2023-03-05","2023-03-05"],
        "City": ["Cairo","Cairo","Alex","Cairo","Giza","Cairo","Cairo"],
        "Product": ["A","B","C","A","B","A","D"],
        "Sales Amount": ["1,200.50","800","1000","250","","1.300,00","200"],
        "Profit": [200,100,150,50, None, 260, 30],
        "Customer": ["a@x.com","a@x.com","b@x.com","c@x.com","d@x.com","e@x.com","e@x.com"]
    })
    eda = ExploratoryAnalyzer(output_dir="eda_reports_demo", plot=True, verbose=True)
    results = eda.run(df_demo)
    print("Generated reports:", results.get("text_path"), results.get("html_path"))
