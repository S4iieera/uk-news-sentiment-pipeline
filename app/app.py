from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from dashboard_io import (
    compute_coverage_for_run,
    default_run_id,
    get_metrics_for_display,
    infer_run_info,
    list_available_runs,
    load_features,
    load_global_coverage_report,
    load_inspection,
    load_model_dataset,
    load_prices,
)


st.set_page_config(page_title="FYP Dashboard", layout="wide")


def _find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    lowered = {str(c).lower(): str(c) for c in df.columns}
    for candidate in candidates:
        found = lowered.get(candidate.lower())
        if found:
            return found
    return None


def _run_selector() -> str:
    runs = list_available_runs()
    if not runs:
        st.error("No runs found under data/processed/.")
        st.stop()

    default_run = default_run_id(runs)
    default_index = runs.index(default_run) if default_run in runs else 0
    return st.sidebar.selectbox("Run ID", runs, index=default_index)


def _page_selector() -> str:
    return st.sidebar.radio(
        "Page",
        ["Overview", "Coverage", "Model Comparison", "Ticker Drilldown"],
        index=0,
    )


def page_overview(run_id: str) -> None:
    st.title("Overview")
    info = infer_run_info(run_id)

    col_run, col_window = st.columns(2)
    with col_run:
        st.subheader("Run")
        st.write(f"**Run ID:** `{run_id}`")
        st.write(f"**Provider:** {info.get('provider') or 'Unknown'}")
        st.write(f"**Timezone:** {info.get('timezone') or 'Unknown'}")
        st.write(f"**Cutoff rule:** {info.get('cutoff_time') or 'Unknown'}")

    with col_window:
        st.subheader("Window")
        st.write(f"**Start:** {info.get('window_start') or 'Unknown'}")
        st.write(f"**End:** {info.get('window_end') or 'Unknown'}")
        tickers = info.get("tickers") or []
        st.write(f"**Tickers:** {', '.join(tickers) if tickers else 'Unknown'}")

    st.subheader("Sentiment settings")
    sentiment = info.get("sentiment") or {}
    if sentiment:
        st.json(sentiment)
    else:
        st.info("No sentiment metadata found in run_metadata.json. Falling back to file-level inference only.")

    st.subheader("No-lookahead logic")
    st.write(
        "Guardian timestamps are converted to Europe/London time. Articles strictly after "
        "16:30 are assigned to the next effective date. Features at date d are used to "
        "predict d+1."
    )

    st.subheader("Read-only dashboard scope")
    st.write(
        "This dashboard is a presentation layer over existing saved artefacts. It does not "
        "run ingestion, FinBERT scoring, training, or evaluation."
    )


def page_coverage(run_id: str) -> None:
    st.title("Coverage")

    coverage = compute_coverage_for_run(run_id)
    if coverage is None or coverage.empty:
        st.warning("Could not compute coverage from run-specific files. Expected prices_daily.parquet.")
    else:
        st.subheader("Run-specific coverage")
        st.dataframe(coverage, use_container_width=True)

        fig = px.bar(
            coverage,
            x="ticker",
            y="coverage_pct",
            text="coverage_pct",
            title="Coverage percentage by ticker",
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="Coverage %", xaxis_title="Ticker")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Global report file check")
    global_cov = load_global_coverage_report()
    if global_cov is None or global_cov.empty:
        st.info("No reports/coverage_report.csv found.")
        return

    if "run_id" in global_cov.columns:
        run_ids = global_cov["run_id"].dropna().astype(str).unique().tolist()
        if len(run_ids) == 1 and run_ids[0] == run_id:
            st.success("reports/coverage_report.csv matches the selected run.")
            st.dataframe(global_cov, use_container_width=True)
        elif run_id in run_ids:
            st.success("reports/coverage_report.csv contains rows for the selected run.")
            st.dataframe(global_cov.loc[global_cov["run_id"].astype(str) == run_id], use_container_width=True)
        else:
            st.warning(
                "reports/coverage_report.csv appears to belong to a different run: "
                + ", ".join(run_ids)
            )
            st.dataframe(global_cov, use_container_width=True)
    else:
        st.warning("reports/coverage_report.csv has no run_id column, so it is treated as global/latest only.")
        st.dataframe(global_cov, use_container_width=True)


def page_model_comparison(run_id: str) -> None:
    st.title("Model Comparison")

    metrics, status = get_metrics_for_display(run_id)
    st.info(status)

    if metrics is None or metrics.empty:
        st.warning("No metrics available to display.")
        return

    variants_order = ["price_only", "price_plus_vader", "price_plus_finbert"]
    if "variant" in metrics.columns:
        metrics = metrics.loc[metrics["variant"].isin(variants_order)].copy()
        metrics["variant"] = pd.Categorical(metrics["variant"], categories=variants_order, ordered=True)
        metrics = metrics.sort_values("variant")

    st.dataframe(metrics, use_container_width=True)

    preferred_cols = [
        "variant",
        "policy_name",
        "policy_drop_no_news_days",
        "policy_min_ticker_coverage_pct",
        "roc_auc",
        "balanced_acc",
        "f1",
        "r2",
        "mae",
        "rmse",
        "status",
    ]
    metric_cols = [c for c in preferred_cols if c in metrics.columns]
    metrics_small = metrics[metric_cols].copy() if metric_cols else metrics.copy()

    plot_metric = next((c for c in ["roc_auc", "balanced_acc", "f1"] if c in metrics_small.columns), None)
    if plot_metric and "variant" in metrics_small.columns:
        fig = px.bar(
            metrics_small,
            x="variant",
            y=plot_metric,
            color="variant",
            title=f"{plot_metric} by variant",
        )
        st.plotly_chart(fig, use_container_width=True)

    if "policy_name" in metrics.columns:
        available_policies = metrics["policy_name"].dropna().astype(str).unique().tolist()
        st.write("**Policies present in loaded data:** " + ", ".join(available_policies))
    else:
        st.write("**Policy metadata not present** in loaded metrics file.")

    if "variant" in metrics.columns and "price_plus_finbert" not in metrics["variant"].astype(str).tolist():
        st.info("price_plus_finbert is not available in the currently loaded metrics file.")


def page_ticker_drilldown(run_id: str) -> None:
    st.title("Ticker Drilldown")

    prices = load_prices(run_id)
    features = load_features(run_id)
    model_dataset = load_model_dataset(run_id)
    inspection = load_inspection(run_id)

    if prices is None or prices.empty:
        st.warning("Missing prices_daily.parquet for this run.")
        return

    px_ticker = _find_col(prices, ["ticker", "symbol"])
    px_date = _find_col(prices, ["date", "trade_date", "datetime"])
    px_close = _find_col(prices, ["adj_close", "close"])

    if not px_ticker or not px_date or not px_close:
        st.warning("prices_daily.parquet does not have the expected ticker/date/price columns.")
        return

    prices = prices.copy()
    prices["ticker"] = prices[px_ticker].astype(str)
    prices["date"] = pd.to_datetime(prices[px_date], errors="coerce")
    prices["close"] = pd.to_numeric(prices[px_close], errors="coerce")
    prices = prices.sort_values(["ticker", "date"])
    prices["daily_return"] = prices.groupby("ticker")["close"].pct_change(fill_method=None)

    tickers = sorted(prices["ticker"].dropna().unique().tolist())
    selected_ticker = st.selectbox("Ticker", tickers)
    px_sub = prices.loc[prices["ticker"] == selected_ticker].copy()

    feat_sub = None
    if features is not None and not features.empty:
        ft_ticker = _find_col(features, ["ticker"])
        ft_date = _find_col(features, ["date"])
        if ft_ticker and ft_date:
            features = features.copy()
            features["ticker"] = features[ft_ticker].astype(str)
            features["date"] = pd.to_datetime(features[ft_date], errors="coerce")
            feat_sub = features.loc[features["ticker"] == selected_ticker].copy()

    target_sub = None
    if model_dataset is not None and not model_dataset.empty:
        md_ticker = _find_col(model_dataset, ["ticker"])
        md_date = _find_col(model_dataset, ["date"])
        if md_ticker and md_date and "target_ret_t1" in model_dataset.columns:
            model_dataset = model_dataset.copy()
            model_dataset["ticker"] = model_dataset[md_ticker].astype(str)
            model_dataset["date"] = pd.to_datetime(model_dataset[md_date], errors="coerce")
            keep_cols = ["ticker", "date", "target_ret_t1"]
            if "target_up_t1" in model_dataset.columns:
                keep_cols.append("target_up_t1")
            target_sub = model_dataset.loc[model_dataset["ticker"] == selected_ticker, keep_cols].copy()

    plot_df = px_sub[["date", "close", "daily_return"]].copy()
    if feat_sub is not None:
        feature_cols = ["date"]
        for candidate in ["n_articles", "sent_vader_mean", "finbert_score_mean", "finbert_pos_mean"]:
            if candidate in feat_sub.columns:
                feature_cols.append(candidate)
        plot_df = plot_df.merge(feat_sub[feature_cols], on="date", how="left")
    if target_sub is not None and not target_sub.empty:
        target_cols = ["date", "target_ret_t1"]
        if "target_up_t1" in target_sub.columns:
            target_cols.append("target_up_t1")
        plot_df = plot_df.merge(target_sub[target_cols], on="date", how="left")

    st.subheader("Chart controls")
    available_sentiments = {}
    if "sent_vader_mean" in plot_df.columns:
        available_sentiments["VADER"] = "sent_vader_mean"
    if "finbert_score_mean" in plot_df.columns:
        available_sentiments["FinBERT"] = "finbert_score_mean"
    elif "finbert_pos_mean" in plot_df.columns:
        available_sentiments["FinBERT positive probability"] = "finbert_pos_mean"

    selected_sentiments = st.multiselect(
        "Sentiment series to display",
        options=list(available_sentiments.keys()),
        default=list(available_sentiments.keys()),
        help="These are daily sentiment aggregates from saved features, not newly computed scores.",
    )
    overlay_mode = st.radio(
        "Price-panel sentiment overlay",
        ["Sentiment badges", "Sentiment markers", "Scaled signal line", "None"],
        index=0,
        horizontal=True,
        help="Badges/markers show positive or negative sentiment on dates with Guardian items. Scaled signal lines are visual guides only and are not predicted prices.",
    )
    signal_threshold = st.slider(
        "Minimum absolute sentiment signal for badges/markers",
        min_value=0.0,
        max_value=1.0,
        value=0.05,
        step=0.01,
        help="Higher values reduce visual clutter by hiding weaker sentiment signals.",
    )
    show_return_on_badges = st.checkbox(
        "Show realised next-day return % on sentiment badges",
        value=True,
        help="Shows only the actual saved target_ret_t1 percentage on badges. Sentiment source and sign are shown by colour/symbol and hover text.",
    )
    show_news_volume = st.checkbox("Show news volume panel", value=True)

    st.info(
        "The dashboard shows actual market prices and saved sentiment aggregates. It does not display "
        "model-predicted prices because the current evaluation stores summary metrics, not per-date "
        "prediction series. Sentiment badges and scaled overlays are visual sentiment indicators only, not trading advice or causal claims."
    )

    if "n_articles" in plot_df.columns:
        articles = pd.to_numeric(plot_df["n_articles"], errors="coerce").fillna(0)
    else:
        articles = pd.Series(0, index=plot_df.index)

    stat_cols = st.columns(5)
    start_price = pd.to_numeric(plot_df["close"], errors="coerce").dropna().iloc[0]
    end_price = pd.to_numeric(plot_df["close"], errors="coerce").dropna().iloc[-1]
    price_change_pct = ((end_price / start_price) - 1) * 100 if start_price else 0
    stat_cols[0].metric("Start price", f"{start_price:.2f}")
    stat_cols[1].metric("End price", f"{end_price:.2f}", f"{price_change_pct:.2f}%")
    stat_cols[2].metric("Total articles", int(articles.sum()))
    stat_cols[3].metric("News days", int((articles > 0).sum()))
    stat_cols[4].metric("Selected signals", len(selected_sentiments))

    row_count = 3 if show_news_volume else 2
    row_heights = [0.58, 0.27, 0.15] if show_news_volume else [0.68, 0.32]
    subplot_titles = ["Actual close price", "Daily sentiment aggregates"]
    if show_news_volume:
        subplot_titles.append("Guardian article count by assigned date")

    fig = make_subplots(
        rows=row_count,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.045,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["date"],
            y=plot_df["close"],
            mode="lines",
            name="Actual close price",
            line={"width": 2.5, "color": "#1f4e79"},
        ),
        row=1,
        col=1,
    )

    colors = {
        "VADER": "#e67e22",
        "FinBERT": "#16a085",
        "FinBERT positive probability": "#16a085",
    }
    marker_symbols = {"VADER": "triangle-up", "FinBERT": "circle", "FinBERT positive probability": "circle"}
    price_span = float(plot_df["close"].max() - plot_df["close"].min()) or 1.0
    price_mid = float(plot_df["close"].mean())
    if "target_ret_t1" in plot_df.columns:
        next_return_pct = pd.to_numeric(plot_df["target_ret_t1"], errors="coerce") * 100
    else:
        next_return_pct = pd.Series(pd.NA, index=plot_df.index, dtype="float64")
    signal_summaries = []

    for label in selected_sentiments:
        col_name = available_sentiments[label]
        score = pd.to_numeric(plot_df[col_name], errors="coerce")
        color = colors.get(label, "#6c757d")
        neutral_point = 0.5 if "probability" in label.lower() else 0.0
        directional_score = score - neutral_point
        signal_mask = score.notna() & (articles > 0) & (directional_score.abs() >= signal_threshold)

        if "target_up_t1" in plot_df.columns:
            valid = signal_mask & plot_df["target_up_t1"].notna()
            if valid.any():
                predicted_up = (directional_score.loc[valid] > 0).astype(int)
                actual_up = pd.to_numeric(plot_df.loc[valid, "target_up_t1"], errors="coerce").astype(int)
                matches = predicted_up.to_numpy() == actual_up.to_numpy()
                signal_summaries.append(
                    {
                        "sentiment_source": label,
                        "threshold": signal_threshold,
                        "labelled_dates_checked": int(valid.sum()),
                        "same_sign_as_next_day_direction": int(matches.sum()),
                        "match_rate_pct": round(float(matches.mean() * 100), 2),
                    }
                )

        if overlay_mode in {"Sentiment badges", "Sentiment markers"}:
            pos_mask = signal_mask & (directional_score > 0)
            neg_mask = signal_mask & (directional_score < 0)
            marker_mode = "markers+text" if overlay_mode == "Sentiment badges" else "markers"
            for mask, name, symbol, marker_color, text_label, text_position in [
                (pos_mask, f"{label} positive sentiment", marker_symbols.get(label, "triangle-up"), "#2ca02c", f"{label} +", "top center"),
                (neg_mask, f"{label} negative sentiment", "triangle-down", "#d62728", f"{label} -", "bottom center"),
            ]:
                returns_for_mask = next_return_pct.loc[mask]
                badge_text = []
                for ret_pct in returns_for_mask:
                    if overlay_mode == "Sentiment badges" and show_return_on_badges and pd.notna(ret_pct):
                        badge_text.append(f"{ret_pct:+.2f}%")
                    elif overlay_mode == "Sentiment badges":
                        badge_text.append(text_label)
                    else:
                        badge_text.append("")
                customdata = pd.DataFrame(
                    {
                        "score": score.loc[mask].to_numpy(),
                        "next_return_pct": returns_for_mask.to_numpy(),
                    }
                )
                fig.add_trace(
                    go.Scatter(
                        x=plot_df.loc[mask, "date"],
                        y=plot_df.loc[mask, "close"],
                        mode=marker_mode,
                        name=name,
                        text=badge_text,
                        textposition=text_position,
                        textfont={"size": 10, "color": marker_color},
                        marker={"symbol": symbol, "size": 9, "color": marker_color, "line": {"width": 0.5, "color": "white"}},
                        customdata=customdata,
                        hovertemplate=(
                            "Date=%{x}<br>Close=%{y:.2f}<br>Source=" + name +
                            "<br>Sentiment score=%{customdata[0]:.3f}"
                            "<br>Realised next-day return=%{customdata[1]:+.2f}%<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )
        elif overlay_mode == "Scaled signal line":
            scaled = price_mid + directional_score.fillna(0.0) * (price_span * 0.35)
            fig.add_trace(
                go.Scatter(
                    x=plot_df["date"],
                    y=scaled,
                    mode="lines",
                    name=f"{label} scaled signal (not price prediction)",
                    line={"dash": "dot", "width": 1.6, "color": color},
                ),
                row=1,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=plot_df["date"],
                y=score,
                mode="lines",
                name=f"{label} sentiment",
                line={"width": 2, "color": color},
            ),
            row=2,
            col=1,
        )

    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    if show_news_volume:
        fig.add_trace(
            go.Bar(
                x=plot_df["date"],
                y=articles,
                name="News volume",
                marker_color="#7f8c8d",
                opacity=0.65,
            ),
            row=3,
            col=1,
        )

    fig.update_yaxes(title_text="Close price", row=1, col=1)
    fig.update_yaxes(title_text="Sentiment", range=[-1, 1], row=2, col=1)
    if show_news_volume:
        fig.update_yaxes(title_text="Articles", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=row_count, col=1)
    fig.update_layout(
        title=f"{selected_ticker}: price, sentiment signals and news volume",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        hovermode="x unified",
        height=820 if show_news_volume else 690,
        margin={"l": 55, "r": 25, "t": 95, "b": 45},
    )
    st.plotly_chart(fig, use_container_width=True)

    if overlay_mode in {"Sentiment badges", "Sentiment markers"} and selected_sentiments:
        st.caption(
            "Badge text shows realised next-day return only. Colour shows sentiment sign: green = positive, red = negative. "
            "Symbol/legend identify whether the marker came from VADER or FinBERT."
        )

    if signal_summaries:
        st.subheader("Sentiment-implied direction check")
        st.dataframe(pd.DataFrame(signal_summaries), use_container_width=True)
        st.caption(
            "This descriptive check compares the sign of the selected daily sentiment signal against the saved next-day direction target. "
            "It is not a trading strategy backtest and does not account for transaction costs or portfolio construction."
        )

    with st.expander("How to read this chart"):
        st.write(
            "The top panel shows the actual close price. Sentiment badges mark dates where the selected "
            "daily aggregate was positive or negative and Guardian items were present. "
            "The middle panel shows the sentiment score itself on its own scale. The bottom panel shows article count. "
            "Scaled signal lines, if enabled, are visual guides only and should not be described as model-predicted prices."
        )
    st.subheader("Inspection items")
    if inspection is None or inspection.empty:
        st.info("No inspection_items.csv found for this run.")
        return

    insp = inspection.copy()
    if "ticker" not in insp.columns:
        st.info("inspection_items.csv exists but has no ticker column.")
        return

    insp = insp.loc[insp["ticker"].astype(str) == selected_ticker].copy()
    preferred_cols = [
        "ticker",
        "provider",
        "guardian_id",
        "item_hash",
        "url",
        "published_at_london",
        "assigned_date",
        "aligned_trade_date",
        "query_expr",
        "sentiment_score",
        "sentiment_label",
        "same_day_return",
        "next_day_return",
        "manual_relevance",
        "review_notes",
    ]
    show_cols = [c for c in preferred_cols if c in insp.columns]
    st.dataframe(insp[show_cols], use_container_width=True, height=350)


def main() -> None:
    st.sidebar.title("FYP Dashboard")
    run_id = _run_selector()
    page = _page_selector()

    st.sidebar.write("---")
    st.sidebar.write("Presentation layer only")
    st.sidebar.caption("Loads saved artefacts only. No ingestion, scoring, training, or evaluation.")

    if page == "Overview":
        page_overview(run_id)
    elif page == "Coverage":
        page_coverage(run_id)
    elif page == "Model Comparison":
        page_model_comparison(run_id)
    elif page == "Ticker Drilldown":
        page_ticker_drilldown(run_id)


if __name__ == "__main__":
    main()
