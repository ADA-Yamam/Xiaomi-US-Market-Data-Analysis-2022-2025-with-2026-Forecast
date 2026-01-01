# descriptive_analysis_full.py
"""
DescriptiveAnalyzer — Final Unified Descriptive Analysis Library (v1.1)
- Single-file, production-oriented
- Smart mapping & derivation of missing columns
- Comprehensive analyses (time, geo, category, customer, association, pareto, contribution, RFM, cohort, CLV, ROI)
- Each major analysis produces at least one plot (PNG)
- Safe: no uncaught exceptions; warnings/inferences are kept in audit
- Outputs: text, json, html, PNG plots, aggregate CSVs (data-model)
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Helpers ----------
def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def _fmt(x):
    try:
        return f"{float(x):,.2f}"
    except Exception:
        return str(x)

def _safe_apply(f, *args, **kwargs):
    try:
        return f(*args, **kwargs)
    except Exception as e:
        return None

# ---------- Main Class ----------
class DescriptiveAnalyzer:
    """
    DescriptiveAnalyzer: comprehensive descriptive analytics + reporting.

    Usage:
      analyzer = DescriptiveAnalyzer(output_dir="reports", plot=True, verbose=True)
      report = analyzer.generate_report(df, out_prefix="run1")
    """

    DEFAULT_MAPPING = {
        "sales": ["sales", "revenue", "amount", "total", "price", "value", "sales_amount"],
        "profit": ["profit", "profit_val", "net_profit", "profit_amount"],
        "price": ["price", "unit_price", "unitprice"],
        "qty": ["qty", "quantity", "units"],
        "order_id": ["order_id", "orderid", "id", "transaction_id"],
        "customer_id": ["customer_id", "customer", "client_id", "user_id", "email"],
        "product": ["product", "product_name", "sku", "item", "product_id"],
        "category": ["category", "cat", "type", "group"],
        "city": ["city", "town", "location", "city_name"],
        "region": ["region", "state", "province"],
        "order_date": ["order_date", "date", "created_at", "timestamp", "orderdate"],
        "cogs": ["cogs", "cost", "cost_of_goods_sold", "unit_cost"],
        "visitors": ["visitors", "sessions", "traffic", "visits"],
        "campaign_cost": ["campaign_cost", "marketing_cost", "ad_cost", "cac"]
    }

    def __init__(self,
                 output_dir: str = "desc_reports",
                 plot: bool = True,
                 convert_locale_numbers: bool = True,
                 min_support_pairs: float = 0.01,
                 verbose: bool = True):
        self.output_dir = output_dir
        _ensure_dir(self.output_dir)
        self.fig_dir = os.path.join(self.output_dir, "figures")
        _ensure_dir(self.fig_dir)
        self.plot = plot
        self.plots: List[str] = []
        self.audit: Dict[str, Any] = {"warnings": [], "inferences": [], "mapping": {}}
        self.convert_locale_numbers = convert_locale_numbers
        self.min_support_pairs = float(min_support_pairs)

        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
            self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO if verbose else logging.WARNING)

    # ---------------- Smart mapping ----------------
    def smart_find_column(self, df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        cols = list(df.columns)
        lower_map = {c.lower(): c for c in cols}
        for cand in candidates:
            if cand.lower() in lower_map:
                return lower_map[cand.lower()]
        # partial match
        for c in cols:
            cl = c.lower()
            for cand in candidates:
                if cand.lower() in cl:
                    return c
        return None

    def auto_map_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        mapping = {}
        for key, candidates in self.DEFAULT_MAPPING.items():
            col = self.smart_find_column(df, candidates)
            if col:
                mapping[key] = col
        self.audit["mapping"] = mapping
        return mapping

    # ---------------- Locale numeric conversion ----------------
    def _convert_locale_numbers_series(self, s: pd.Series) -> pd.Series:
        # best-effort: detect European format vs US
        try:
            sample = s.dropna().astype(str).head(1000)
            if sample.empty:
                return s
            both = sample.apply(lambda x: ('.' in x) and (',' in x)).sum()
            comma_only = sample.apply(lambda x: (',' in x) and ('.' not in x)).sum()
            dot_only = sample.apply(lambda x: ('.' in x) and (',' not in x)).sum()
            if both > 0 or (comma_only > dot_only and comma_only > len(sample) * 0.1):
                def parse_eu(x):
                    try:
                        if pd.isna(x): return np.nan
                        t = str(x).strip()
                        t = t.replace('.', '').replace(',', '.')
                        return float(t)
                    except Exception:
                        return np.nan
                return sample.index.to_series().apply(lambda i: parse_eu(s.iat[i])) if False else s.apply(parse_eu)
            else:
                def parse_us(x):
                    try:
                        if pd.isna(x): return np.nan
                        t = str(x).strip().replace(',', '')
                        return float(t) if t not in ['', 'nan', 'None'] else np.nan
                    except Exception:
                        return np.nan
                return s.apply(parse_us)
        except Exception as e:
            self.audit["warnings"].append(f"Locale conversion failed: {e}")
            return s

    # ---------------- Ensure derived columns ----------------
    def ensure_derived_columns(self, df: pd.DataFrame, mapping: Dict[str,str]) -> Tuple[pd.DataFrame, Dict[str,str]]:
        df = df.copy()
        # find/parse order_date
        date_col = mapping.get("order_date")
        if date_col and date_col in df.columns:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            except Exception as e:
                try:
                    df[date_col] = df[date_col].apply(lambda x: pd.to_datetime(x, errors="coerce") if pd.notna(x) else x)
                except Exception:
                    self.audit["warnings"].append(f"Failed to parse date column '{date_col}': {e}")
        else:
            # attempt to infer a datetime-like column
            found = None
            for c in df.columns:
                try:
                    if pd.api.types.is_datetime64_any_dtype(df[c]):
                        found = c; break
                except Exception:
                    pass
                try:
                    parsed = pd.to_datetime(df[c].dropna().head(50), errors="coerce")
                    if parsed.notna().sum() > 0:
                        found = c; break
                except Exception:
                    pass
            if found:
                mapping["order_date"] = found
                self.audit["inferences"].append(f"Inferred order_date='{found}'")
            else:
                self.audit["warnings"].append("No order_date found; time analyses will be limited.")

        # derived time columns if order_date exists
        if mapping.get("order_date") and mapping["order_date"] in df.columns:
            try:
                col = mapping["order_date"]
                df["__year"] = df[col].dt.year
                df["__month"] = df[col].dt.to_period("M").astype(str)
                # week: try isocalendar, fallback to dt.week
                try:
                    df["__week"] = df[col].dt.isocalendar().week
                except Exception:
                    df["__week"] = df[col].dt.week
                df["__dayofweek"] = df[col].dt.day_name()
                self.audit["inferences"].append("Created time columns: __year, __month, __week, __dayofweek")
            except Exception as e:
                self.audit["warnings"].append(f"Failed to create time derived cols: {e}")

        # locale numeric conversion for candidates
        numeric_candidates = ["sales", "price", "qty", "profit", "cogs"]
        if self.convert_locale_numbers:
            for key in numeric_candidates:
                col = mapping.get(key)
                if col and col in df.columns and df[col].dtype == object:
                    df[col] = self._convert_locale_numbers_series(df[col])
                    self.audit["inferences"].append(f"Applied locale numeric conversion on '{col}'")

        # derive sales from price*qty if missing
        if "sales" not in mapping:
            p = mapping.get("price")
            q = mapping.get("qty")
            if p and q and p in df.columns and q in df.columns:
                try:
                    df["__sales_derived"] = pd.to_numeric(df[p], errors="coerce") * pd.to_numeric(df[q], errors="coerce")
                    mapping["sales"] = "__sales_derived"
                    self.audit["inferences"].append("Derived 'sales' as price * qty into __sales_derived")
                except Exception as e:
                    self.audit["warnings"].append(f"Failed deriving sales from price*qty: {e}")

        # derive profit net if missing and cogs available
        if "profit" not in mapping:
            s = mapping.get("sales")
            c = mapping.get("cogs")
            if s and c and s in df.columns and c in df.columns:
                try:
                    df["__profit_derived"] = pd.to_numeric(df[s], errors="coerce") - pd.to_numeric(df[c], errors="coerce")
                    mapping["profit"] = "__profit_derived"
                    self.audit["inferences"].append("Derived 'profit' as sales - cogs into __profit_derived")
                except Exception as e:
                    self.audit["warnings"].append(f"Failed deriving profit from sales-cogs: {e}")

        # customer_id fallback
        if "customer_id" not in mapping:
            for alt in ["email", "user_id", "client_id"]:
                if alt in df.columns:
                    mapping["customer_id"] = alt
                    self.audit["inferences"].append(f"Using '{alt}' as customer_id fallback")
                    break

        self.audit["mapping"] = mapping
        return df, mapping

    # ---------------- Plot saving ----------------
    def _save_plot(self, fig, name: str) -> Optional[str]:
        try:
            path = os.path.join(self.fig_dir, f"{name}.png")
            fig.tight_layout()
            fig.savefig(path, dpi=150)
            plt.close(fig)
            self.plots.append(path)
            return path
        except Exception as e:
            self.audit["warnings"].append(f"Failed saving plot {name}: {e}")
            try:
                plt.close(fig)
            except Exception:
                pass
            return None

    # ---------------- Basic metrics ----------------
    def basic_metrics(self, df: pd.DataFrame, mapping: Dict[str,str]) -> Dict[str,Any]:
        out = {}
        try:
            sales_col = mapping.get("sales")
            profit_col = mapping.get("profit")
            order_count = len(df)
            out["order_count"] = int(order_count)
            if sales_col and sales_col in df.columns:
                revenue = pd.to_numeric(df[sales_col], errors="coerce").sum()
                out["revenue"] = float(revenue if not pd.isna(revenue) else 0.0)
            else:
                out["revenue"] = 0.0
                self.audit["warnings"].append("sales column missing for basic metrics")
            # Net profit: if profit missing but cogs available, attempt derivation already done
            if profit_col and profit_col in df.columns:
                profit = pd.to_numeric(df[profit_col], errors="coerce").sum()
                out["net_profit"] = float(profit if not pd.isna(profit) else 0.0)
            else:
                out["net_profit"] = None
            out["aov"] = float(out["revenue"] / order_count) if order_count else 0.0
            # margins
            if out["revenue"] and out["net_profit"] is not None:
                out["net_margin"] = (out["net_profit"] / out["revenue"]) if out["revenue"] != 0 else None
            else:
                out["net_margin"] = None
        except Exception as e:
            self.audit["warnings"].append(f"basic_metrics failed: {e}")
        return out

    # ---------------- Time series analysis ----------------
    def time_series_analysis(self, df: pd.DataFrame, mapping: Dict[str,str]) -> Dict[str,Any]:
        res = {}
        try:
            sales_col = mapping.get("sales")
            date_col = mapping.get("order_date")
            if not (sales_col and date_col and date_col in df.columns):
                self.audit["warnings"].append("Skipping time_series_analysis due to missing date or sales column")
                return res
            ts = df.dropna(subset=[date_col]).copy()
            ts[sales_col] = pd.to_numeric(ts[sales_col], errors="coerce")
            ts["__period_month"] = ts[date_col].dt.to_period("M").astype(str)
            monthly = ts.groupby("__period_month")[sales_col].sum().sort_index()
            yearly = ts.groupby(ts[date_col].dt.year)[sales_col].sum().sort_index()
            res["monthly"] = monthly
            res["yearly"] = yearly
            # growth
            if len(monthly) >= 2:
                res["mom_growth_pct"] = float(((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]) * 100) if monthly.iloc[-2] != 0 else None
            else:
                res["mom_growth_pct"] = None
            if len(yearly) >= 2:
                res["yoy_growth_pct"] = float(((yearly.iloc[-1] - yearly.iloc[-2]) / yearly.iloc[-2]) * 100) if yearly.iloc[-2] != 0 else None
            else:
                res["yoy_growth_pct"] = None
            # top months
            try:
                res["top_months"] = monthly.sort_values(ascending=False).head(6)
                res["most_buy_months"] = list(monthly.groupby(monthly.index).sum().sort_values(ascending=False).index[:3])
            except Exception:
                pass

            # plotting (monthly & yearly)
            if self.plot:
                try:
                    fig = plt.figure(figsize=(9,4))
                    monthly.plot(marker="o", title="Monthly Revenue")
                    figax = self._save_plot(fig, "monthly_revenue")
                    fig2 = plt.figure(figsize=(9,4))
                    yearly.plot(marker="o", title="Yearly Revenue")
                    self._save_plot(fig2, "yearly_revenue")
                except Exception as e:
                    self.audit["warnings"].append(f"time_series plotting failed: {e}")
        except Exception as e:
            self.audit["warnings"].append(f"time_series_analysis failed: {e}")
        return res

    # ---------------- Geographic (sales & profit by city) ----------------
    def geographic_analysis(self, df: pd.DataFrame, mapping: Dict[str,str]) -> Dict[str,Any]:
        res = {}
        try:
            city_col = mapping.get("city")
            sales_col = mapping.get("sales")
            profit_col = mapping.get("profit")
            if city_col and sales_col and city_col in df.columns:
                agg = df.groupby(city_col)[sales_col].agg(lambda x: pd.to_numeric(x, errors="coerce").sum()).sort_values(ascending=False)
                res["sales_by_city"] = agg
                if self.plot:
                    try:
                        fig = plt.figure(figsize=(10,4))
                        agg.head(20).plot(kind="bar", title="Sales by City")
                        self._save_plot(fig, "sales_by_city")
                    except Exception as e:
                        self.audit["warnings"].append(f"geographic plotting failed: {e}")
            if city_col and profit_col and city_col in df.columns and profit_col in df.columns:
                agg_p = df.groupby(city_col)[profit_col].agg(lambda x: pd.to_numeric(x, errors="coerce").sum()).sort_values(ascending=False)
                res["profit_by_city"] = agg_p
                if self.plot:
                    try:
                        fig = plt.figure(figsize=(10,4))
                        agg_p.head(20).plot(kind="bar", title="Profit by City")
                        self._save_plot(fig, "profit_by_city")
                    except Exception as e:
                        self.audit["warnings"].append(f"geographic profit plotting failed: {e}")
        except Exception as e:
            self.audit["warnings"].append(f"geographic_analysis failed: {e}")
        return res

    # ---------------- Category analysis (sales & margin per category) ----------------
    def category_analysis(self, df: pd.DataFrame, mapping: Dict[str,str]) -> Dict[str,Any]:
        res = {}
        try:
            cat = mapping.get("category")
            prod = mapping.get("product")
            sales = mapping.get("sales")
            profit = mapping.get("profit")
            if cat and sales and cat in df.columns:
                agg = df.groupby(cat)[sales].agg(lambda x: pd.to_numeric(x, errors="coerce").sum()).sort_values(ascending=False)
                res["sales_by_category"] = agg
                if self.plot:
                    try:
                        fig = plt.figure(figsize=(9,4))
                        agg.head(20).plot(kind="bar", title="Sales by Category")
                        self._save_plot(fig, "sales_by_category")
                    except Exception as e:
                        self.audit["warnings"].append(f"category plotting failed: {e}")
                if profit and profit in df.columns:
                    try:
                        margin = df.groupby(cat).apply(lambda g: (pd.to_numeric(g[profit], errors="coerce").sum() / pd.to_numeric(g[sales], errors="coerce").sum()) if pd.to_numeric(g[sales], errors="coerce").sum() != 0 else np.nan)
                        res["margin_by_category"] = margin.sort_values(ascending=False)
                        if self.plot:
                            fig2 = plt.figure(figsize=(9,4))
                            (margin.dropna().head(20)*100).plot(kind="bar", title="Gross Margin % by Category")
                            self._save_plot(fig2, "margin_by_category")
                    except Exception as e:
                        self.audit["warnings"].append(f"category margin calc failed: {e}")
        except Exception as e:
            self.audit["warnings"].append(f"category_analysis failed: {e}")
        return res

    # ---------------- Ranking (top products, top cities) ----------------
    def ranking_analysis(self, df: pd.DataFrame, mapping: Dict[str,str]) -> Dict[str,Any]:
        res = {}
        try:
            sales = mapping.get("sales")
            prod = mapping.get("product")
            city = mapping.get("city")
            if prod and sales and prod in df.columns:
                top_prod = df.groupby(prod)[sales].agg(lambda x: pd.to_numeric(x, errors="coerce").sum()).sort_values(ascending=False).head(50)
                res["top_products"] = top_prod
                if self.plot:
                    try:
                        fig = plt.figure(figsize=(9,4))
                        top_prod.head(10).plot(kind="bar", title="Top 10 Products by Sales")
                        self._save_plot(fig, "top10_products")
                    except Exception as e:
                        self.audit["warnings"].append(f"ranking products plotting failed: {e}")
            if city and sales and city in df.columns:
                top_city = df.groupby(city)[sales].agg(lambda x: pd.to_numeric(x, errors="coerce").sum()).sort_values(ascending=False).head(50)
                res["top_cities"] = top_city
                if self.plot:
                    try:
                        fig = plt.figure(figsize=(9,4))
                        top_city.head(10).plot(kind="bar", title="Top 10 Cities by Sales")
                        self._save_plot(fig, "top10_cities")
                    except Exception as e:
                        self.audit["warnings"].append(f"ranking cities plotting failed: {e}")
        except Exception as e:
            self.audit["warnings"].append(f"ranking_analysis failed: {e}")
        return res

    # ---------------- Pareto & Contribution ----------------
    def contribution_and_pareto(self, df: pd.DataFrame, mapping: Dict[str,str]) -> Dict[str,Any]:
        res = {}
        try:
            sales = mapping.get("sales")
            prod = mapping.get("product")
            if prod and sales and prod in df.columns:
                prod_sales = df.groupby(prod)[sales].agg(lambda x: pd.to_numeric(x, errors="coerce").sum()).sort_values(ascending=False)
                cum_pct = (prod_sales.cumsum() / prod_sales.sum()) * 100
                res["pareto_cum_pct"] = cum_pct
                # plot Pareto
                if self.plot:
                    try:
                        fig = plt.figure(figsize=(9,4))
                        ax = prod_sales.plot(kind="bar", alpha=0.6, label="Sales")
                        ax2 = ax.twinx()
                        cum_pct.plot(ax=ax2, color="red", marker="o", label="Cumulative %")
                        ax.set_title("Pareto: Sales & Cumulative % by Product")
                        self._save_plot(fig, "pareto_products")
                    except Exception as e:
                        self.audit["warnings"].append(f"pareto plotting failed: {e}")
        except Exception as e:
            self.audit["warnings"].append(f"contribution_and_pareto failed: {e}")
        return res

    # ---------------- Customer behavior (RFM, cohort, CLV estimate, repeat rate) ----------------
    def customer_behavior(self, df: pd.DataFrame, mapping: Dict[str,str]) -> Dict[str,Any]:
        res = {}
        try:
            cust = mapping.get("customer_id")
            date_col = mapping.get("order_date")
            order_col = mapping.get("order_id")
            sales = mapping.get("sales")
            if not cust:
                self.audit["warnings"].append("No customer identifier -- skipping customer behavior")
                return res
            # build orders: if order_id exists aggregate rows into orders
            try:
                df_local = df.copy()
                # ensure date parsed
                if date_col and date_col in df_local.columns:
                    df_local = df_local.dropna(subset=[date_col])
                # order-level revenue
                if order_col and order_col in df_local.columns and sales and sales in df_local.columns:
                    orders = df_local.groupby(order_col).agg({date_col: "first", cust: "first", sales: lambda x: pd.to_numeric(x, errors="coerce").sum()}).rename(columns={sales: "order_revenue", date_col: "order_date"})
                else:
                    # fallback: treat each row as order
                    orders = df_local[[cust, date_col, sales]].rename(columns={sales: "order_revenue"})
                    orders = orders.dropna(subset=[cust])
                # RFM
                if date_col and date_col in df_local.columns:
                    now = df_local[date_col].max()
                    last_purchase = orders.groupby(cust)["order_date"].max()
                    frequency = orders.groupby(cust).size()
                    monetary = orders.groupby(cust)["order_revenue"].sum() if "order_revenue" in orders.columns else frequency
                    rfm = pd.concat([last_purchase, frequency, monetary], axis=1).rename(columns={"order_date": "last_order_date", 0: "frequency", "order_revenue": "monetary"})
                    rfm["recency_days"] = (now - rfm["last_order_date"]).dt.days
                else:
                    frequency = orders.groupby(cust).size()
                    monetary = orders.groupby(cust)["order_revenue"].sum() if "order_revenue" in orders.columns else frequency
                    rfm = pd.concat([frequency, monetary], axis=1).rename(columns={0: "frequency", "order_revenue": "monetary"})
                    rfm["recency_days"] = np.nan
                rfm = rfm.fillna(0)
                res["rfm_sample"] = rfm.head(5).to_dict(orient="index")
                # repeat rate and frequency
                repeat_rate = (rfm["frequency"] > 1).sum() / max(1, rfm.shape[0])
                res["repeat_rate"] = float(repeat_rate)
                res["avg_purchase_freq"] = float(rfm["frequency"].mean())
                res["avg_monetary"] = float(rfm["monetary"].mean())
                # CLV simple estimate
                aov = res["avg_monetary"] / (res["avg_purchase_freq"] if res["avg_purchase_freq"]>0 else 1)
                clv = aov * res["avg_purchase_freq"] * 12  # assume 12 months lifetime
                res["clv_simple"] = float(clv)
                # cohort basic (counts)
                if date_col and date_col in df_local.columns:
                    orders2 = orders.reset_index()
                    orders2["order_month"] = orders2["order_date"].dt.to_period("M").astype(str)
                    first_month = orders2.groupby(cust)["order_month"].min().rename("cohort")
                    cohorts = orders2.join(first_month, on=cust).groupby(["cohort", "order_month"]).size().unstack(fill_value=0)
                    res["cohort_head"] = cohorts.head(5).to_dict()
                    if self.plot:
                        try:
                            fig = plt.figure(figsize=(8,4))
                            cohorts_sum = cohorts.sum(axis=1)
                            cohorts_sum.plot(kind="bar", title="Cohort sizes (by first month)")
                            self._save_plot(fig, "cohort_sizes")
                        except Exception as e:
                            self.audit["warnings"].append(f"cohort plotting failed: {e}")
            except Exception as e:
                self.audit["warnings"].append(f"customer behavior calc failed: {e}")
        except Exception as e:
            self.audit["warnings"].append(f"customer_behavior failed: {e}")
        return res

    # ---------------- Association pairs (market basket) ----------------
    def association_pairs(self, df: pd.DataFrame, mapping: Dict[str,str], min_support: Optional[float]=None) -> Dict[str,Any]:
        res = {}
        try:
            order_col = mapping.get("order_id")
            prod_col = mapping.get("product")
            if not order_col or not prod_col or order_col not in df.columns or prod_col not in df.columns:
                self.audit["warnings"].append("Skipping association pairs — order_id or product missing")
                return res
            trans = df.groupby(order_col)[prod_col].apply(lambda s: list(set(s.dropna().astype(str)))).tolist()
            n = len(trans)
            item_counts = Counter()
            pair_counts = Counter()
            for t in trans:
                item_counts.update(t)
                tu = sorted(set(t))
                for i in range(len(tu)):
                    for j in range(i+1, len(tu)):
                        pair_counts[(tu[i], tu[j])] += 1
            pairs = []
            minsupp = self.min_support_pairs if min_support is None else min_support
            for pair, cnt in pair_counts.items():
                support = cnt / n
                if support < minsupp:
                    continue
                a, b = pair
                conf_a_b = cnt / item_counts[a] if item_counts[a] else 0
                conf_b_a = cnt / item_counts[b] if item_counts[b] else 0
                lift = (cnt * n) / (item_counts[a] * item_counts[b]) if (item_counts[a] * item_counts[b]) else 0
                pairs.append({"pair": pair, "support": support, "conf_a_b": conf_a_b, "conf_b_a": conf_b_a, "lift": lift, "count": int(cnt)})
            pairs_sorted = sorted(pairs, key=lambda x: (-x["lift"], -x["support"]))
            res["pairs"] = pairs_sorted[:200]
            # plot top pairs counts
            if self.plot and pairs_sorted:
                try:
                    top = pairs_sorted[:10]
                    names = [f"{p['pair'][0]} & {p['pair'][1']}" if False else f"{p['pair'][0]} & {p['pair'][1]}" for p in top]
                    counts = [p['count'] for p in top]
                    fig = plt.figure(figsize=(10,4))
                    plt.barh(range(len(names))[::-1], counts)
                    plt.yticks(range(len(names))[::-1], names)
                    plt.title("Top product pairs by co-occurrence (count)")
                    self._save_plot(fig, "top_product_pairs")
                except Exception as e:
                    self.audit["warnings"].append(f"association plotting failed: {e}")
        except Exception as e:
            self.audit["warnings"].append(f"association_pairs failed: {e}")
        return res

    # ---------------- Data modeling (aggregates and model table) ----------------
    def generate_model_table(self, df: pd.DataFrame, mapping: Dict[str,str]) -> Dict[str,str]:
        out = {}
        try:
            dfm = df.copy()
            sales = mapping.get("sales")
            profit = mapping.get("profit")
            date_col = mapping.get("order_date")
            prod = mapping.get("product")
            cust = mapping.get("customer_id")
            city = mapping.get("city")
            if date_col and date_col in dfm.columns:
                dfm["__period"] = dfm[date_col].dt.to_period("M").astype(str)
            else:
                dfm["__period"] = "unknown"
            # Ensure numeric cols
            if sales and sales in dfm.columns:
                dfm[sales] = pd.to_numeric(dfm[sales], errors="coerce")
            if profit and profit in dfm.columns:
                dfm[profit] = pd.to_numeric(dfm[profit], errors="coerce")
            # per period-product
            if prod and sales and prod in dfm.columns:
                try:
                    gp = dfm.groupby(["__period", prod]).agg(revenue=(sales, lambda x: pd.to_numeric(x, errors="coerce").sum()),
                                                              orders=(mapping.get("order_id") if mapping.get("order_id") in dfm.columns else sales, "count"))
                    p1 = os.path.join(self.output_dir, "by_month_product.csv") 
                    gp.reset_index().to_csv(p1, index=False)
                    out["by_month_product"] = p1
                except Exception as e:
                    self.audit["warnings"].append(f"by_month_product failed: {e}")
            # per period-customer
            if cust and sales and cust in dfm.columns:
                try:
                    gc = dfm.groupby(["__period", cust]).agg(revenue=(sales, lambda x: pd.to_numeric(x, errors="coerce").sum()),
                                                              orders=(mapping.get("order_id") if mapping.get("order_id") in dfm.columns else sales, "count"))
                    p2 = os.path.join(self.output_dir, "by_month_customer.csv")
                    gc.reset_index().to_csv(p2, index=False)
                    out["by_month_customer"] = p2
                except Exception as e:
                    self.audit["warnings"].append(f"by_month_customer failed: {e}")
            # per period-city
            if city and sales and city in dfm.columns:
                try:
                    gcity = dfm.groupby(["__period", city]).agg(revenue=(sales, lambda x: pd.to_numeric(x, errors="coerce").sum()),
                                                                 orders=(mapping.get("order_id") if mapping.get("order_id") in dfm.columns else sales, "count"))
                    p3 = os.path.join(self.output_dir, "by_month_city.csv")
                    gcity.reset_index().to_csv(p3, index=False)
                    out["by_month_city"] = p3
                except Exception as e:
                    self.audit["warnings"].append(f"by_month_city failed: {e}")
            # summary model table
            try:
                periods = dfm["__period"].unique()
                rows = []
                for p in periods:
                    sub = dfm[dfm["__period"]==p]
                    row = {"period": p, "revenue": float(pd.to_numeric(sub[sales], errors="coerce").sum()) if sales and sales in sub.columns else 0.0,
                           "profit": float(pd.to_numeric(sub[profit], errors="coerce").sum()) if profit and profit in sub.columns else 0.0,
                           "orders": int(sub[mapping.get("order_id")].nunique()) if mapping.get("order_id") in sub.columns else int(len(sub))}
                    rows.append(row)
                dfsum = pd.DataFrame(rows)
                p4 = os.path.join(self.output_dir, "summary_model_table.csv")
                dfsum.to_csv(p4, index=False)
                out["summary_model_table"] = p4
            except Exception as e:
                self.audit["warnings"].append(f"summary_model_table failed: {e}")
        except Exception as e:
            self.audit["warnings"].append(f"generate_model_table failed: {e}")
        return out

    # ---------------- Build final report (text/json/html + plots list) ----------------
    def generate_report(self, df: pd.DataFrame, out_prefix: str = "desc_report") -> Dict[str,Any]:
        results = {"mapping": {}, "analysis": {}, "plots": [], "files": [], "audit": self.audit}
        try:
            df0 = df.copy()
            mapping = self.auto_map_columns(df0)
            df1, mapping = self.ensure_derived_columns(df0, mapping)
            results["mapping"] = mapping

            # run analyses
            results["analysis"]["basic"] = self.basic_metrics(df1, mapping)
            results["analysis"]["time"] = self.time_series_analysis(df1, mapping)
            results["analysis"]["geo"] = self.geographic_analysis(df1, mapping)
            results["analysis"]["category"] = self.category_analysis(df1, mapping)
            results["analysis"]["ranking"] = self.ranking_analysis(df1, mapping)
            results["analysis"]["pareto"] = self.contribution_and_pareto(df1, mapping)
            results["analysis"]["customer"] = self.customer_behavior(df1, mapping)
            results["analysis"]["association"] = self.association_pairs(df1, mapping)
            # model tables
            files = self.generate_model_table(df1, mapping)
            results["files"].extend(list(files.values()))
            results["plots"] = self.plots

            # build text summary
            lines = []
            lines.append("===== DESCRIPTIVE ANALYSIS REPORT =====")
            lines.append(f"Rows: {len(df0)}, Columns: {len(df0.columns)}")
            lines.append(f"Auto-mapped columns: {json.dumps(mapping, ensure_ascii=False)}")
            # basic
            b = results["analysis"]["basic"]
            lines.append("\n--- BASIC METRICS ---")
            lines.append(f"Orders: {b.get('order_count')}")
            lines.append(f"Revenue: {_fmt(b.get('revenue',0))}")
            if b.get("net_profit") is not None:
                lines.append(f"Net Profit: {_fmt(b.get('net_profit',0))}")
                lines.append(f"Net Margin: {_fmt((b.get('net_margin') or 0)*100)} %")
            lines.append(f"AOV: {_fmt(b.get('aov',0))}")
            # time
            lines.append("\n--- TIME HIGHLIGHTS ---")
            ts = results["analysis"]["time"]
            if ts:
                lines.append(f"MoM Growth (%): {_fmt(ts.get('mom_growth_pct')) if ts.get('mom_growth_pct') is not None else 'N/A'}")
                lines.append(f"YoY Growth (%): {_fmt(ts.get('yoy_growth_pct')) if ts.get('yoy_growth_pct') is not None else 'N/A'}")
                topm = ts.get("top_months")
                if topm is not None:
                    lines.append("Top months (sample): " + ", ".join([str(x) for x in list(topm.index.astype(str))[:3]]))
            else:
                lines.append("Time analysis: skipped")

            lines.append("\n--- CUSTOMER HIGHLIGHTS ---")
            cust = results["analysis"]["customer"]
            if cust:
                lines.append(f"Repeat rate: {_fmt(cust.get('repeat_rate',0)*100)} %")
                lines.append(f"CLV (simple est): {_fmt(cust.get('clv_simple',0))}")
            else:
                lines.append("Customer analysis: skipped")

            if self.audit.get("warnings"):
                lines.append("\n--- WARNINGS ---")
                for w in self.audit["warnings"]:
                    lines.append(f"- {w}")
            if self.audit.get("inferences"):
                lines.append("\n--- INFERENCES ---")
                for inf in self.audit["inferences"]:
                    lines.append(f"- {inf}")

            # save text
            text_path = os.path.join(self.output_dir, f"{out_prefix}.txt")
            with open(text_path, "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
            results["text_path"] = text_path

            # save json
            json_path = os.path.join(self.output_dir, f"{out_prefix}.json")
            try:
                # ensure serializable: convert pandas objects to lists/strings
                serial = json.dumps(results, default=lambda o: (o.strftime("%Y-%m-%d") if hasattr(o, "strftime") else str(o)), ensure_ascii=False, indent=2)
                with open(json_path, "w", encoding="utf-8") as f:
                    f.write(serial)
                results["json_path"] = json_path
            except Exception:
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump({"mapping": mapping, "audit": self.audit}, f, ensure_ascii=False, indent=2)
                results["json_path"] = json_path

            # html
            html_path = os.path.join(self.output_dir, f"{out_prefix}.html")
            try:
                html_lines = ["<html><head><meta charset='utf-8'><title>Descriptive Report</title></head><body>"]
                html_lines.append("<h1>Descriptive Analysis Report</h1>")
                html_lines.append(f"<h3>Mapping</h3><pre>{json.dumps(mapping, ensure_ascii=False, indent=2)}</pre>")
                html_lines.append("<h3>Highlights</h3><pre>")
                html_lines.extend([f"{l}\n" for l in lines])
                html_lines.append("</pre>")
                if self.plots:
                    html_lines.append("<h3>Plots</h3>")
                    for p in self.plots:
                        html_lines.append(f"<div><img src='{os.path.basename(p)}' style='max-width:900px'></div>")
                html_lines.append("</body></html>")
                with open(html_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(html_lines))
                results["html_path"] = html_path
            except Exception as e:
                self.audit["warnings"].append(f"HTML report generation failed: {e}")

            results["audit"] = self.audit
            results["files"].extend(results.get("plots", []))
        except Exception as e:
            # never raise; capture and return what we have
            self.audit["warnings"].append(f"generate_report failed overall: {e}")
        return results

# ---------------- Demo when run as script ----------------
if __name__ == "__main__":
    print("DescriptiveAnalyzer demo (final unified). Producing sample report in ./desc_reports_demo")
    # small sample
    df_sample = pd.DataFrame({
        "order_id": [1,1,2,3,4,5,5],
        "created_at": ["2023-01-10","2023-01-10","2023-02-15","2023-02-20","2023-03-05","2023-03-05","2023-03-05"],
        "city_name": ["Cairo","Cairo","Alex","Cairo","Giza","Cairo","Cairo"],
        "product_name": ["A","B","C","A","B","A","D"],
        "sales_amount": ["1,200.50","800","1000","250","", "1.300,00", "200"],
        "profit_val": [200,100,150,50, None, 260, 30],
        "customer_email": ["a@x.com","a@x.com","b@x.com","c@x.com","d@x.com","e@x.com","e@x.com"]
    })
    analyzer = DescriptiveAnalyzer(output_dir="desc_reports_demo", plot=True, verbose=True)
    report = analyzer.generate_report(df_sample, out_prefix="demo_full")
    print("Generated:", report.get("text_path"), report.get("html_path"), report.get("json_path"))
800","1000","", "1300"],
        "profit_val": [200,100,150, None, 260]
    })

    # 1) Clean
    cleaner = DataCleaningEngine(
        fill_map={"sales_amount": 0, "profit_val": 0},
        force_id_columns=["order_id"],
        sample_size=1000,
        verbose=True
    )
    cleaned_df, audit = cleaner.clean(df_sample)

    # export audit
    cleaner.export_audit(os.path.join("reports","audit.json"), fmt="json")

    # 2) Analyze descriptively and generate report
    analyzer = DescriptiveAnalyzer(output_dir="reports", plot=True)
    report_info = analyzer.generate_report(cleaned_df, audit, out_prefix="sales_report")

    print("\nReport files generated:")
    print(report_info)
-------- aggregated_by_city_category(df, cols)
    ranking_analysis(df, cols)
    contribution_analysis(df, cols)
    pareto_analysis(df, cols)
    auto_insights(df, cols)
