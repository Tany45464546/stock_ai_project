import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Institutional AI Trading Dashboard", layout="wide")

st.title("⚡ Institutional AI & LLM Trading Intelligence Dashboard")

df = pd.read_csv("final_stock_ai_dataset.csv", index_col=0, parse_dates=True)
latest = df.iloc[-1]

# Try loading backtest data if available
try:
    bt_df = pd.read_csv("backtest_results.csv", index_col=0, parse_dates=True)
    has_backtest = True
except Exception:
    has_backtest = False

# Executive KPI Bar
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Close Price", f"${latest['Close']:.2f}")
col2.metric("AI Composite Signal", latest['Composite_Signal'])
col3.metric("XGBoost Confidence", f"{latest['ML_Confidence']*100:.1f}%")
col4.metric("News Sentiment", latest['Sentiment_Label'])
col5.metric("14-Day ATR (Volatility)", f"${latest['ATR_14']:.2f}")

st.divider()

# Interactive Analytical Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📉 Price & Bollinger Bands", 
    "📊 MACD Indicator", 
    "🎯 RSI Momentum", 
    "💰 Strategy Backtest",
    "📋 Signal Log & Data"
])

with tab1:
    st.subheader("Price Action, Moving Averages & Bollinger Bands")
    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close Price', line=dict(color='white', width=2)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], mode='lines', name='SMA 20', line=dict(color='cyan', width=1)))
    fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Upper'], mode='lines', name='Bollinger Upper', line=dict(color='gray', dash='dash')))
    fig_price.add_trace(go.Scatter(x=df.index, y=df['BB_Lower'], mode='lines', name='Bollinger Lower', line=dict(color='gray', dash='dash')))
    fig_price.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig_price, use_container_width=True)

with tab2:
    st.subheader("MACD (Moving Average Convergence Divergence)")
    fig_macd = go.Figure()
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD'], mode='lines', name='MACD Line', line=dict(color='blue')))
    fig_macd.add_trace(go.Scatter(x=df.index, y=df['MACD_Signal'], mode='lines', name='Signal Line', line=dict(color='orange')))
    fig_macd.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_macd, use_container_width=True)

with tab3:
    st.subheader("Relative Strength Index (RSI)")
    fig_rsi = px.line(df, y='RSI_14', title="14-Day RSI Momentum")
    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
    fig_rsi.update_layout(template="plotly_dark", height=450)
    st.plotly_chart(fig_rsi, use_container_width=True)

with tab4:
    st.subheader("Quantitative Strategy Performance vs. Benchmark")
    if has_backtest:
        strat_ret = ((bt_df['Strategy_Equity'].iloc[-1] - bt_df['Strategy_Equity'].iloc[0]) / bt_df['Strategy_Equity'].iloc[0]) * 100
        bench_ret = ((bt_df['Bench_Equity'].iloc[-1] - bt_df['Bench_Equity'].iloc[0]) / bt_df['Bench_Equity'].iloc[0]) * 100
        
        b_col1, b_col2, b_col3 = st.columns(3)
        b_col1.metric("AI Strategy Return", f"{strat_ret:+.2f}%")
        b_col2.metric("Buy & Hold Benchmark", f"{bench_ret:+.2f}%")
        b_col3.metric("Ending Portfolio Value", f"${bt_df['Strategy_Equity'].iloc[-1]:,.2f}")

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Strategy_Equity'], mode='lines', name='AI Strategy Equity', line=dict(color='limegreen', width=2.5)))
        fig_bt.add_trace(go.Scatter(x=bt_df.index, y=bt_df['Bench_Equity'], mode='lines', name='Buy & Hold Benchmark', line=dict(color='orange', width=1.5, dash='dash')))
        fig_bt.update_layout(template="plotly_dark", height=450, title="Cumulative Portfolio Growth ($10,000 Initial)")
        st.plotly_chart(fig_bt, use_container_width=True)
    else:
        st.warning("Run 'python3 04_backtest.py' first to generate backtest performance metrics.")

with tab5:
    st.subheader("Recent Prediction Logs & Indicator Metrics")
    display_cols = ['Close', 'SMA_20', 'RSI_14', 'MACD', 'ATR_14', 'ML_Predicted_Signal', 'ML_Confidence', 'Composite_Signal']
    st.dataframe(df[display_cols].tail(15))
