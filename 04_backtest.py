import pandas as pd
import numpy as np

def run_backtest(data_path="final_stock_ai_dataset.csv", initial_capital=10000.0):
    print("--> Loading final AI dataset for backtesting...")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)

    # 1. Define Strategy Execution Rule:
    # Position = 1 (Hold Stock) when Composite Signal is BUY or STRONG BUY
    # Position = 0 (Cash) when Composite Signal is SELL or STRONG SELL
    df['Position'] = df['Composite_Signal'].apply(lambda x: 1 if 'BUY' in str(x) else 0)

    # Shift position by 1 day to simulate buying at today's close and holding for tomorrow's return
    df['Strategy_Return'] = df['Position'].shift(1) * df['Daily_Return']

    # Fill initial NaN value
    df['Strategy_Return'].fillna(0, inplace=True)
    df['Daily_Return'].fillna(0, inplace=True)

    # 2. Cumulative Equity Curves
    df['Bench_Equity'] = initial_capital * (1 + df['Daily_Return']).cumprod()
    df['Strategy_Equity'] = initial_capital * (1 + df['Strategy_Return']).cumprod()

    # 3. Calculate Key Quantitative Performance Metrics
    total_bench_return = ((df['Bench_Equity'].iloc[-1] - initial_capital) / initial_capital) * 100
    total_strat_return = ((df['Strategy_Equity'].iloc[-1] - initial_capital) / initial_capital) * 100

    # Annualized Sharpe Ratio (assuming 0% risk-free rate)
    strategy_daily_mean = df['Strategy_Return'].mean()
    strategy_daily_std = df['Strategy_Return'].std()
    sharpe_ratio = (strategy_daily_mean / strategy_daily_std) * np.sqrt(252) if strategy_daily_std != 0 else 0

    # Maximum Drawdown (MDD)
    df['Peak'] = df['Strategy_Equity'].cummax()
    df['Drawdown'] = (df['Strategy_Equity'] - df['Peak']) / df['Peak']
    max_drawdown = df['Drawdown'].min() * 100

    # Trade Signals & Win Rate
    trades = df['Position'].diff().abs().sum()
    winning_days = (df['Strategy_Return'] > 0).sum()
    total_active_days = (df['Position'].shift(1) == 1).sum()
    win_rate = (winning_days / total_active_days * 100) if total_active_days > 0 else 0

    # Print Performance Summary Report
    print("\n" + "="*55)
    print(f" 📊 QUANTITATIVE BACKTESTING REPORT (Starting: ${initial_capital:,.2f})")
    print("="*55)
    print(f" • Buy & Hold Benchmark Return : {total_bench_return:+.2f}%")
    print(f" • AI Strategy Return          : {total_strat_return:+.2f}%")
    print(f" • Final Strategy Portfolio    : ${df['Strategy_Equity'].iloc[-1]:,.2f}")
    print(f" • Annualized Sharpe Ratio     : {sharpe_ratio:.2f}")
    print(f" • Maximum Drawdown (MDD)      : {max_drawdown:.2f}%")
    print(f" • Winning Days Rate           : {win_rate:.1f}%")
    print(f" • Total Portfolio Rebalances  : {int(trades)}")
    print("="*55)

    # Save backtest results
    output_filename = "backtest_results.csv"
    df.to_csv(output_filename)
    print(f"--> Saved backtest timeline to '{output_filename}'.")
    return df

if __name__ == "__main__":
    run_backtest(initial_capital=10000.0)
