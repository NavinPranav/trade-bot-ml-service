"""Calculate backtest performance metrics."""
import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.065) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - risk_free_rate / 252
    return float(np.sqrt(252) * excess.mean() / excess.std())


def max_drawdown(equity_curve: pd.Series) -> float:
    if equity_curve.empty:
        return 0.0
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return float(drawdown.min())


def win_rate(trades: pd.DataFrame) -> float:
    if trades.empty or "pnl" not in trades.columns:
        return 0.0
    wins = (trades["pnl"] > 0).sum()
    return float(wins / len(trades) * 100)


def calmar_ratio(total_return: float, max_dd: float) -> float:
    if max_dd == 0:
        return 0.0
    return abs(total_return / max_dd)