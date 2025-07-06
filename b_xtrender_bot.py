"""Simplified translation of the B‑Xtrender trading bot.

This script mirrors the original Pine Script structure and includes placeholder logic. Replace these sections with real implementations to make the bot functional.
"""

import pandas as pd

class BXtrenderBot:
    """Container for strategy configuration and placeholder trading logic."""
    def __init__(self, data, initial_capital=30000):
        self.data = data
        self.initial_capital = initial_capital
        self.strategy = "B-Xtrender"
        self.trade_conditions = {
            "barstate_entry": "Bar Confirmed",
            "trade_before_close_setting": True,
            "include_no_data_setting": True,
            "trade_before_close_minute": 58,
            "enable_realtime_entries": False,
            "realtime_entries_setting": "Both"
        }
        self.bx_strategy = {
            "entry_setting": "Higher High",
            "exit_setting": "Lower Low"
        }
        self.weekly_watchlist = {
            "weekly_watchlist_exit_setting": "Lower Low"
        }
        self.hourly_swing = {}

    def apply_strategy(self):
        # Placeholder for strategy logic
        # This is where you would implement the logic based on the settings
        pass

# Example usage
data = pd.DataFrame()  # Replace with actual data
bot = BXtrenderBot(data)
bot.apply_strategy()

import pandas as pd

# Define input parameters
take_profit_percent = 2.0
tp_min_bars_to_exit = 2
stop_loss_percent = 4.0
sl_min_bars_to_exit = 2
enable_rsi_condition = True
candle_delay = 0

# Trend filter settings
enable_trend_filter_1 = True
daily_bxtrender_trend_settings_1 = "All"
weekly_bxtrender_trend_settings_1 = "All"
monthly_bxtrender_trend_settings_1 = "All"

enable_trend_filter_2 = False
daily_bxtrender_trend_settings_2 = "All"
weekly_bxtrender_trend_settings_2 = "All"
monthly_bxtrender_trend_settings_2 = "All"

enable_trend_filter_3 = False
daily_bxtrender_trend_settings_3 = "All"
weekly_bxtrender_trend_settings_3 = "All"
monthly_bxtrender_trend_settings_3 = "All"

enable_trend_filter_4 = False
daily_bxtrender_trend_settings_4 = "All"

# Example DataFrame setup
data = {
    'price': [100, 102, 104, 103, 105],
    'rsi': [30, 35, 40, 45, 50],
    'daily_trend': ['HH', 'HL', 'LL', 'LH', 'HH'],
    'weekly_trend': ['HH', 'HL', 'LL', 'LH', 'HH'],
    'monthly_trend': ['HH', 'HL', 'LL', 'LH', 'HH']
}

df = pd.DataFrame(data)

# Example logic for applying filters and conditions
def apply_trend_filter(row, enable_filter, daily_setting, weekly_setting, monthly_setting):
    """Check if a row meets the configured trend filters."""
    if not enable_filter:
        return True
    return (
        (daily_setting == "All" or row['daily_trend'] == daily_setting)
        and (weekly_setting == "All" or row['weekly_trend'] == weekly_setting)
        and (monthly_setting == "All" or row['monthly_trend'] == monthly_setting)
    )

df['trend_filter_1'] = df.apply(lambda row: apply_trend_filter(row, enable_trend_filter_1, daily_bxtrender_trend_settings_1, weekly_bxtrender_trend_settings_1, monthly_bxtrender_trend_settings_1), axis=1)

# Example logic for RSI condition
df['rsi_condition'] = df['rsi'] > 50 if enable_rsi_condition else True

# Example logic for trade signals
df['trade_signal'] = df['trend_filter_1'] & df['rsi_condition']

print(df)

import pandas as pd

# Define settings using dictionaries to mimic input options
settings = {
    "weekly_bxtrender_trend": "All",
    "monthly_bxtrender_trend": "All",
    "enable_weekly_rsi_filter": True,
    "enable_monthly_rsi_filter": False,
    "enable_mb_filter": False,
    "daily_mb_settings": "All",
    "weekly_mb_settings": "All",
    "monthly_mb_settings": "All",
    "repaint_daily": False,
    "repaint_weekly": False,
    "repaint_monthly": False,
    "short_l1": 5,
    "short_l2": 20,
    "short_l3": 5,
    "rsi_length": 14,
    "rsi_ma_length": 14,
    "mb_len": 20,
    "mb_smoothing": 7,
    "mb_osc_len": 7,
    "hourly_swing_sma_1_setting": 11,
    "hourly_swing_sma_2_setting": 100,
    "use_backtest_date": False,
    "backtest_time": pd.Timestamp("2020-01-01 19:00:00+0500"),
    "use_backtest_stop_date": False,
    "backtest_stop_time": pd.Timestamp("2025-01-01 19:00:00+0500")
}

# Example usage of settings
def apply_settings(df):
    """Apply optional RSI and market-bias filters to a DataFrame."""
    # Example: Apply RSI filter if enabled
    if settings["enable_weekly_rsi_filter"]:
        df['RSI'] = df['Close'].rolling(window=settings["rsi_length"]).mean()  # Simplified RSI calculation
        df['RSI_MA'] = df['RSI'].rolling(window=settings["rsi_ma_length"]).mean()
        df = df[df['RSI'] > df['RSI_MA']]
    
    # Example: Apply market bias filter if enabled
    if settings["enable_mb_filter"]:
        # Simplified market bias filter logic
        df['Market_Bias'] = df['Close'].rolling(window=settings["mb_len"]).mean()
        df = df[df['Market_Bias'] > df['Close']]
    
    return df

# Assuming df is your DataFrame with market data
# df = pd.read_csv('market_data.csv')
# df = apply_settings(df)

import pandas as pd

# Initial settings
initial_account_size = 30000.0
base_position_size = 100.0
enable_strategy = False

# Alert settings
alert_strategy_name = "B-Xtrender Bot"
alert_strategy_secret = ""
send_htf_mismatch_alerts = False

# Table settings
label_text_size = "small"
show_stats_table = True
show_bxtrender_rows = False
show_bx_filter_stats = False
stats_table_position = "top_right"
table_text_size = "small"
table_bg_color = (0, 0, 0)  # RGB color
table_border_color = "gray"
table_header_color = (0, 0, 255, 70)  # RGBA color

# Experimental settings
enable_entry_hl_rsi_above_ma = False
enable_entry_hl_down_candle = False
enable_bx_stop_loss = False
bx_stop_loss = 3.0

# Timeframe checks
def is_timeframe(period, target):
    """Utility to compare pandas time offsets."""
    return pd.to_timedelta(period) == pd.to_timedelta(target)

is_15m = is_timeframe('15T', '15T')
is_hourly = is_timeframe('1H', '1H')
is_daily = is_timeframe('1D', '1D')
is_weekly = is_timeframe('1W', '1W')
is_monthly = is_timeframe('1M', '1M')

# Use THT bot settings
use_tht_bot_settings = False  # This would be set based on your logic

if use_tht_bot_settings:
    table_header_color = (0, 188, 212, 70)  # RGBA color

    short_l1 = 5
    short_l2 = 20
    short_l3 = 5

    rsi_length = 14
    rsi_ma_length = 14

    enable_trend_filter_1 = True
    enable_trend_filter_2 = False
    enable_trend_filter_3 = False

import pandas as pd

def configure_strategy(strategy, timeframe, is_15m, use_bx_filters_with_tht_settings):
    """Return a configuration dictionary for the chosen strategy."""
    config = {}

    # Trend filter settings
    config['enable_trend_filter_4'] = False
    config['daily_bxtrender_trend_settings_1'] = "All" if not use_bx_filters_with_tht_settings else "All"
    config['weekly_bxtrender_trend_settings_1'] = "All" if not use_bx_filters_with_tht_settings else "Only Positive"
    config['monthly_bxtrender_trend_settings_1'] = "All" if not use_bx_filters_with_tht_settings else "Positive+HL"

    if strategy == "B-Xtrender":
        config['barstate_entry'] = "New Bar" if timeframe == 'daily' else "Bar Confirmed"
        config['entry_setting'] = "Higher Low" if timeframe == 'daily' else "Higher High"
        config['exit_setting'] = "Lower High" if timeframe == 'weekly' else "Lower Low"
        config['enable_entry_hl_down_candle'] = False
        config['enable_entry_hl_rsi_above_ma'] = False
        config['enable_bx_stop_loss'] = False
        config['enable_weekly_rsi_filter'] = False if timeframe == 'weekly' else True
        config['enable_monthly_rsi_filter'] = False

    elif strategy == "Hourly Swing/Quant Bot":
        config['barstate_entry'] = "Bar Confirmed"
        if is_15m:
            config['hourly_swing_sma_1_setting'] = 11
            config['hourly_swing_sma_2_setting'] = 50
            config['take_profit_percent'] = 0.5
            config['stop_loss_percent'] = 2
            config['tp_min_bars_to_exit'] = 1
            config['sl_min_bars_to_exit'] = 2
            config['candle_delay'] = 2
        else:
            config['hourly_swing_sma_1_setting'] = 11
            config['hourly_swing_sma_2_setting'] = 100
            config['take_profit_percent'] = 2
            config['stop_loss_percent'] = 4
            config['tp_min_bars_to_exit'] = 2
            config['sl_min_bars_to_exit'] = 2
            config['candle_delay'] = 0

        config['enable_weekly_rsi_filter'] = False
        config['enable_monthly_rsi_filter'] = False

    elif strategy == "Weekly Watchlist":
        config['barstate_entry'] = "New Bar"
        config['weekly_watchlist_exit_setting'] = "Lower Low"
        config['daily_bxtrender_trend_settings_1'] = "All"
        config['weekly_bxtrender_trend_settings_1'] = "Only Positive"
        config['monthly_bxtrender_trend_settings_1'] = "Positive+HL"
        config['enable_weekly_rsi_filter'] = False
        config['enable_monthly_rsi_filter'] = False

    config['include_no_data_setting'] = True
    config['enable_realtime_entries'] = False
    config['repaint_daily'] = False
    config['repaint_weekly'] = False
    config['repaint_monthly'] = False
    config['enable_mb_filter'] = False

    return config

# Example usage
strategy_config = configure_strategy("B-Xtrender", 'daily', False, True)
print(strategy_config)

import pandas as pd
import numpy as np

# Initialize trade statistics
total_trades = 0
winning_trades = 0
losing_trades = 0
cumulative_profit_loss = 0.0
max_profit = 0.0
max_loss = 0.0
avg_win = 0.0
avg_loss = 0.0
sum_wins = 0.0
sum_losses = 0.0
sum_profit = 0.0
sum_loss = 0.0
max_drawdown = 0.0
peak_value = 0.0
buy_and_hold_entry_price = np.nan

# Trend filter statistics
trend_filter_1_trades = 0
trend_filter_2_trades = 0
trend_filter_3_trades = 0
trend_filter_1_profit = 0.0
trend_filter_2_profit = 0.0
trend_filter_3_profit = 0.0
trend_filter_1_wins = 0
trend_filter_2_wins = 0
trend_filter_3_wins = 0
trend_filter_1_win_sum = 0.0
trend_filter_1_loss_sum = 0.0
trend_filter_2_win_sum = 0.0
trend_filter_2_loss_sum = 0.0
trend_filter_3_win_sum = 0.0
trend_filter_3_loss_sum = 0.0
trend_filter_1_losses = 0
trend_filter_2_losses = 0
trend_filter_3_losses = 0
trend_filter_4_trades = 0
trend_filter_4_wins = 0
trend_filter_4_losses = 0
trend_filter_4_profit = 0.0
trend_filter_4_win_sum = 0.0
trend_filter_4_loss_sum = 0.0

# Trade state variables
exit_reason = None
exit_percentage = np.nan
in_trade = False
entry_price = np.nan
bars_in_trade = 0
position_size = 0
take_profit_level = np.nan
stop_loss_level = np.nan

# Separate trade tracking for HTF repainting issues
independent_trade_tracking = False
independent_entry_price = np.nan
independent_bars_in_trade = 0

# Account tracking variables
current_account_size = initial_account_size
peak_account_size = initial_account_size
max_account_drawdown_pct = 0.0

def bxtrender(index, close, short_l1, short_l2, short_l3):
    """Calculate the BXtrender indicator and trend classification."""
    shortTermXtrender = pd.Series(close).ewm(span=short_l1).mean() - pd.Series(close).ewm(span=short_l2).mean()
    shortTermXtrender = shortTermXtrender.rolling(window=short_l3).apply(lambda x: np.mean(x)) - 50
    bx_trend = np.where(shortTermXtrender > 0, 
                        np.where(shortTermXtrender > shortTermXtrender.shift(1), 'higher_high', 'lower_high'), 
                        np.where(shortTermXtrender < shortTermXtrender.shift(1), 'lower_low', 'higher_low'))
    return shortTermXtrender, shortTermXtrender.shift(1), bx_trend

def rsi(index, close, rsi_length, rsi_ma_length):
    """Compute RSI and its moving average."""
    delta = pd.Series(close).diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_ma = rsi.rolling(window=rsi_ma_length).mean()
    return rsi, rsi_ma

def marketbias(index, open_, close, high, low, mb_len, mb_smoothing, mb_osc_len):
    """Approximation of the market bias indicator."""
    o = pd.Series(open_).ewm(span=mb_len).mean()
    c = pd.Series(close).ewm(span=mb_len).mean()
    h = pd.Series(high).ewm(span=mb_len).mean()
    l = pd.Series(low).ewm(span=mb_len).mean()

    haclose = (o + h + l + c) / 4
    xhaopen = (o + c) / 2
    haopen = (xhaopen.shift(1) + haclose.shift(1)) / 2 if not xhaopen.shift(1).isna().all() else (o + c) / 2
    hahigh = np.maximum(h, np.maximum(haopen, haclose))
    halow = np.minimum(l, np.minimum(haopen, haclose))

    o2 = pd.Series(haopen).ewm(span=mb_smoothing).mean()
    c2 = pd.Series(haclose).ewm(span=mb_smoothing).mean()
    h2 = pd.Series(hahigh).ewm(span=mb_smoothing).mean()
    l2 = pd.Series(halow).ewm(span=mb_smoothing).mean()

    ha_avg = (h2 + l2) / 2

    osc_bias = 100 * (c2 - o2)
    osc_smooth = osc_bias.ewm(span=mb_osc_len).mean()

    is_expansion = np.where(osc_bias > 0, osc_bias >= osc_smooth, osc_bias <= osc_smooth)

    return osc_bias, is_expansion

bxtrender_strategy_enabled = strategy == "B-Xtrender" and timeframe_in_seconds(timeframe_period) >= timeframe_in_seconds('60')
hourly_swing_strategy_enabled = strategy == "Hourly Swing/Quant Bot" and (is_15m or is_hourly)
weekly_watchlist_strategy_enabled = strategy == "Weekly Watchlist" and is_daily

enable_realtime_weekly_entries_condition = enable_realtime_entries and (realtime_entries_setting in ["Weekly", "Both"]) and not is_weekly
enable_realtime_monthly_entries_condition = enable_realtime_entries and (realtime_entries_setting in ["Monthly", "Both"]) and not is_monthly

import pandas as pd

def calculate_bxtrender(data, period):
    """Simplified BXtrender calculation placeholder."""
    return (
        data['close'].rolling(window=period).mean(),
        data['close'].shift(1).rolling(window=period).mean(),
        None,
    )

def calculate_rsi(data, period):
    """Simplified RSI calculation used in examples."""
    # Placeholder for RSI calculation logic
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi, rsi.rolling(window=period).mean()

def process_data(
    df,
    is_daily,
    is_weekly,
    is_monthly,
    enable_realtime_weekly_entries_condition,
    enable_realtime_monthly_entries_condition,
    repaint_daily,
    repaint_weekly,
    repaint_monthly,
    strategy,
):
    """Populate a DataFrame with indicator values for multiple timeframes."""
    hourly_bxtrender, hourly_bxtrender_prev, hourly_bxtrender_trend = calculate_bxtrender(df, 1)

    daily_bxtrender = None
    daily_bxtrender_prev = None
    daily_bxtrender_trend = None
    weekly_bxtrender = None
    weekly_bxtrender_prev = None
    weekly_bxtrender_trend = None
    monthly_bxtrender = None
    monthly_bxtrender_prev = None
    monthly_bxtrender_trend = None

    realtime_weekly_bxtrender = None
    realtime_weekly_bxtrender_prev = None
    realtime_weekly_bxtrender_trend = None

    realtime_monthly_bxtrender = None
    realtime_monthly_bxtrender_prev = None
    realtime_monthly_bxtrender_trend = None

    if is_daily:
        daily_bxtrender = hourly_bxtrender
        daily_bxtrender_prev = hourly_bxtrender_prev
        daily_bxtrender_trend = hourly_bxtrender_trend
    else:
        daily_bxtrender, daily_bxtrender_prev, daily_bxtrender_trend = calculate_bxtrender(df, 'D')

    if is_weekly:
        weekly_bxtrender = hourly_bxtrender
        weekly_bxtrender_prev = hourly_bxtrender_prev
        weekly_bxtrender_trend = hourly_bxtrender_trend
    else:
        weekly_bxtrender, weekly_bxtrender_prev, weekly_bxtrender_trend = calculate_bxtrender(df, 'W')
        if enable_realtime_weekly_entries_condition:
            realtime_weekly_bxtrender, realtime_weekly_bxtrender_prev, realtime_weekly_bxtrender_trend = calculate_bxtrender(df, 'W')

    if is_monthly:
        monthly_bxtrender = hourly_bxtrender
        monthly_bxtrender_prev = hourly_bxtrender_prev
        monthly_bxtrender_trend = hourly_bxtrender_trend
    else:
        monthly_bxtrender, monthly_bxtrender_prev, monthly_bxtrender_trend = calculate_bxtrender(df, 'M')
        if enable_realtime_monthly_entries_condition:
            realtime_monthly_bxtrender, realtime_monthly_bxtrender_prev, realtime_monthly_bxtrender_trend = calculate_bxtrender(df, 'M')

    current_rsi, current_rsi_ma = calculate_rsi(df, 14)

    weekly_rsi = None
    weekly_rsi_ma = None
    monthly_rsi = None
    monthly_rsi_ma = None

    realtime_weekly_rsi = None
    realtime_weekly_rsi_ma = None
    realtime_monthly_rsi = None
    realtime_monthly_rsi_ma = None

    if is_weekly:
        weekly_rsi = current_rsi
        weekly_rsi_ma = current_rsi_ma
    else:
        weekly_rsi, weekly_rsi_ma = calculate_rsi(df, 'W')
        if enable_realtime_weekly_entries_condition:
            realtime_weekly_rsi, realtime_weekly_rsi_ma = calculate_rsi(df, 'W')

    return {
        'daily_bxtrender': daily_bxtrender,
        'weekly_bxtrender': weekly_bxtrender,
        'monthly_bxtrender': monthly_bxtrender,
        'realtime_weekly_bxtrender': realtime_weekly_bxtrender,
        'realtime_monthly_bxtrender': realtime_monthly_bxtrender,
        'weekly_rsi': weekly_rsi,
        'monthly_rsi': monthly_rsi,
        'realtime_weekly_rsi': realtime_weekly_rsi,
        'realtime_monthly_rsi': realtime_monthly_rsi
    }

import pandas as pd
import numpy as np

def calculate_rsi(data, period=14):
    """Calculate a basic RSI series."""
    delta = data['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_market_bias(data, repaint=False):
    """Mock market bias calculation used for demonstration."""
    # Placeholder for market bias calculation
    bias = np.random.random(len(data))  # Random values for illustration
    expansion = np.random.choice([True, False], len(data))
    return bias, expansion

def calculate_sma(data, period):
    """Return a simple moving average."""
    return data['close'].rolling(window=period).mean()

def check_trend(trend_settings, bx, trend_value, include_no_data):
    """Evaluate BXtrender trend filters."""
    if include_no_data and pd.isna(bx):
        return True
    if not pd.isna(bx):
        if trend_settings == "All":
            return True
        if trend_settings == "Only Positive" and bx > 0:
            return True
        if trend_settings == "Only Negative" and bx <= 0:
            return True
        # Add other conditions as needed
    return False

# Example DataFrame
data = pd.DataFrame({
    'close': np.random.random(100),
    'low': np.random.random(100),
    'high': np.random.random(100),
    'month': np.random.randint(1, 13, 100)
})

# RSI calculations
data['monthly_rsi'] = calculate_rsi(data)
data['monthly_rsi_ma'] = data['monthly_rsi'].rolling(window=5).mean()

# Market Bias calculations
data['daily_mb_bias'], data['daily_mb_expansion'] = calculate_market_bias(data)
data['weekly_mb_bias'], data['weekly_mb_expansion'] = calculate_market_bias(data)
data['monthly_mb_bias'], data['monthly_mb_expansion'] = calculate_market_bias(data)

# SMA calculations
hourly_swing_sma_1_setting = 10
hourly_swing_sma_2_setting = 20
data['hourly_swing_sma_1'] = calculate_sma(data, hourly_swing_sma_1_setting)
data['hourly_swing_sma_2'] = calculate_sma(data, hourly_swing_sma_2_setting)

# Plotting logic (placeholder)
hourly_swing_strategy_enabled = True
if hourly_swing_strategy_enabled:
    data['plot_sma_1'] = data['hourly_swing_sma_1']
    data['plot_sma_2'] = data['hourly_swing_sma_2']

# Trend checking logic
trend_settings = "All"
bx = np.random.random(100)
trend_value = np.random.choice(['higher_high', 'higher_low', 'lower_low', 'lower_high'], 100)
include_no_data = True
data['trend_check'] = data.apply(lambda row: check_trend(trend_settings, row['close'], trend_value, include_no_data), axis=1)

# Note: This code is a simplified and illustrative translation of the Pine Script logic to Python using pandas.

import pandas as pd

def check_trend(settings, bxtrender, bxtrender_trend, include_no_data):
    """Placeholder for Pine Script trend logic."""
    pass

def check_mb(settings, bias, expansion):
    """Placeholder for market-bias checking logic."""
    pass

def evaluate_conditions(df):
    """Calculate filter conditions used for entries and exits."""
    df['mb_condition'] = (
        (df['mb_settings'] == "All") |
        ((df['mb_settings'] == "Green") & (df['mb_bias'] > 0)) |
        ((df['mb_settings'] == "DG") & (df['mb_bias'] > 0) & df['mb_expansion']) |
        ((df['mb_settings'] == "LG") & (df['mb_bias'] > 0) & ~df['mb_expansion']) |
        ((df['mb_settings'] == "Green+LR") & ((df['mb_bias'] > 0) | ~df['mb_expansion']))
    )

    df['is_post_backtest_trade'] = (
        (~df['use_backtest_date'] | (df['time'] > df['backtest_time'])) &
        (~df['use_backtest_stop_date'] | (df['time'] < df['backtest_stop_time']))
    )

    df['htf_bx_condition_1'] = (
        df['enable_trend_filter_1'] &
        check_trend(df['daily_bxtrender_trend_settings_1'], df['daily_bxtrender'], df['daily_bxtrender_trend'], False) &
        (check_trend(df['weekly_bxtrender_trend_settings_1'], df['weekly_bxtrender'], df['weekly_bxtrender_trend'], False) |
         (df['enable_realtime_weekly_entries_condition'] &
          check_trend(df['weekly_bxtrender_trend_settings_1'], df['realtime_weekly_bxtrender'], df['realtime_weekly_bxtrender_trend'], False))) &
        (check_trend(df['monthly_bxtrender_trend_settings_1'], df['monthly_bxtrender'], df['monthly_bxtrender_trend'], df['include_no_data_setting']) |
         (df['enable_realtime_monthly_entries_condition'] &
          check_trend(df['monthly_bxtrender_trend_settings_1'], df['realtime_monthly_bxtrender'], df['realtime_monthly_bxtrender_trend'], df['include_no_data_setting'])))
    )

    df['htf_bx_condition_2'] = (
        df['enable_trend_filter_2'] &
        check_trend(df['daily_bxtrender_trend_settings_2'], df['daily_bxtrender'], df['daily_bxtrender_trend'], False) &
        (check_trend(df['weekly_bxtrender_trend_settings_2'], df['weekly_bxtrender'], df['weekly_bxtrender_trend'], False) |
         (df['enable_realtime_weekly_entries_condition'] &
          check_trend(df['weekly_bxtrender_trend_settings_2'], df['realtime_weekly_bxtrender'], df['realtime_weekly_bxtrender_trend'], False))) &
        (check_trend(df['monthly_bxtrender_trend_settings_2'], df['monthly_bxtrender'], df['monthly_bxtrender_trend'], df['include_no_data_setting']) |
         (df['enable_realtime_monthly_entries_condition'] &
          check_trend(df['monthly_bxtrender_trend_settings_2'], df['realtime_monthly_bxtrender'], df['realtime_monthly_bxtrender_trend'], df['include_no_data_setting'])))
    )

    df['htf_bx_condition_3'] = (
        df['enable_trend_filter_3'] &
        check_trend(df['daily_bxtrender_trend_settings_3'], df['daily_bxtrender'], df['daily_bxtrender_trend'], False) &
        (check_trend(df['weekly_bxtrender_trend_settings_3'], df['weekly_bxtrender'], df['weekly_bxtrender_trend'], False) |
         (df['enable_realtime_weekly_entries_condition'] &
          check_trend(df['weekly_bxtrender_trend_settings_3'], df['realtime_weekly_bxtrender'], df['realtime_weekly_bxtrender_trend'], False))) &
        (check_trend(df['monthly_bxtrender_trend_settings_3'], df['monthly_bxtrender'], df['monthly_bxtrender_trend'], df['include_no_data_setting']) |
         (df['enable_realtime_monthly_entries_condition'] &
          check_trend(df['monthly_bxtrender_trend_settings_3'], df['realtime_monthly_bxtrender'], df['realtime_monthly_bxtrender_trend'], df['include_no_data_setting'])))
    )

    df['htf_bx_condition_4'] = (
        df['enable_trend_filter_4'] &
        check_trend(df['daily_bxtrender_trend_settings_4'], df['daily_bxtrender'], df['daily_bxtrender_trend'], False) &
        (check_trend(df['weekly_bxtrender_trend_settings_4'], df['weekly_bxtrender'], df['weekly_bxtrender_trend'], False) |
         (df['enable_realtime_weekly_entries_condition'] &
          check_trend(df['weekly_bxtrender_trend_settings_4'], df['realtime_weekly_bxtrender'], df['realtime_weekly_bxtrender_trend'], False))) &
        (check_trend(df['monthly_bxtrender_trend_settings_4'], df['monthly_bxtrender'], df['monthly_bxtrender_trend'], df['include_no_data_setting']) |
         (df['enable_realtime_monthly_entries_condition'] &
          check_trend(df['monthly_bxtrender_trend_settings_4'], df['realtime_monthly_bxtrender'], df['realtime_monthly_bxtrender_trend'], df['include_no_data_setting'])))
    )

    df['htf_bx_condition'] = (
        df['htf_bx_condition_1'] |
        df['htf_bx_condition_2'] |
        df['htf_bx_condition_3'] |
        df['htf_bx_condition_4']
    )

    df['weekly_rsi_condition'] = (
        ~df['enable_weekly_rsi_filter'] |
        (df['weekly_rsi'] > df['weekly_rsi_ma']) |
        (df['enable_realtime_weekly_entries_condition'] & (df['realtime_weekly_rsi'] > df['realtime_weekly_rsi_ma']))
    )

    df['monthly_rsi_condition'] = (
        ~df['enable_monthly_rsi_filter'] |
        (df['monthly_rsi'] > df['monthly_rsi_ma']) |
        (df['enable_realtime_monthly_entries_condition'] & (df['realtime_monthly_rsi'] > df['realtime_monthly_rsi_ma']))
    )

    df['mb_condition'] = (
        ~df['enable_mb_filter'] |
        (check_mb(df['daily_mb_settings'], df['daily_mb_bias'], df['daily_mb_expansion']) &
         check_mb(df['weekly_mb_settings'], df['weekly_mb_bias'], df['weekly_mb_expansion']) &
         check_mb(df['monthly_mb_settings'], df['monthly_mb_bias'], df['monthly_mb_expansion']))
    )

    return df

import pandas as pd
import numpy as np

# Entry conditions
def calculate_entry_conditions(df):
    """Determine entry signals for the active strategy."""
    bx_entry_condition = (
        (df['hourly_bxtrender_trend'] == 'higher_high') |
        ((df['entry_setting'] == "Higher Low") & (df['hourly_bxtrender_trend'] == 'higher_low')) |
        (df['enable_entry_hl_rsi_above_ma'] & (df['current_rsi'] > df['current_rsi_ma']) & (df['hourly_bxtrender_trend'] == 'higher_low')) |
        (df['enable_entry_hl_down_candle'] & (df['hourly_bxtrender_trend'] == 'higher_low') & (df['close'] < df['open']))
    )
    
    hourly_swing_entry_condition = (
        (df['close'].shift(df['candle_delay']) > df[['hourly_swing_sma_1', 'hourly_swing_sma_2']].shift(df['candle_delay']).max(axis=1)) &
        (~df['enable_rsi_condition'] | (df['current_rsi'].shift(df['candle_delay']) > df['current_rsi'].shift(df['candle_delay'] + 1)))
    )
    
    weekly_watchlist_entry_condition = df['month'] != df['month'].shift(1)
    
    strategy_entry_condition = np.where(
        df['hourly_swing_strategy_enabled'], hourly_swing_entry_condition,
        np.where(
            df['weekly_watchlist_strategy_enabled'], weekly_watchlist_entry_condition,
            np.where(df['bxtrender_strategy_enabled'], bx_entry_condition, False)
        )
    )
    
    return strategy_entry_condition

# Exit conditions
def calculate_exit_conditions(df):
    """Determine exit signals for the active strategy."""
    bx_exit_higher_low = (
        (df['exit_setting'] == "Higher Low") &
        ((df['hourly_bxtrender_trend'] == 'higher_low') & (df['hourly_bxtrender_trend'].shift(1) == 'lower_low')) |
        ((df['hourly_bxtrender_trend'] == 'lower_low') & (df['hourly_bxtrender_trend'].shift(1) == 'higher_low'))
    )
    
    bx_exit_condition = (
        ((df['exit_setting'] == "Lower Low") & (df['hourly_bxtrender_trend'] == 'lower_low')) |
        bx_exit_higher_low |
        ((df['exit_setting'] == "Lower High") & (df['hourly_bxtrender_trend'] == 'lower_high')) |
        (df['enable_bx_stop_loss'] & (df['close'] < df['stop_loss_level']))
    )
    
    hourly_swing_exit_condition = (
        ((df['bars_in_trade'] >= df['tp_min_bars_to_exit']) & (df['close'] > df['take_profit_level'])) |
        ((df['bars_in_trade'] >= df['sl_min_bars_to_exit']) & (df['close'] < df['stop_loss_level']))
    )
    
    weekly_watchlist_exit_condition = (
        (df['weekly_bxtrender_trend'] == np.where(df['weekly_watchlist_exit_setting'] == "Higher Low", 'higher_low', 'lower_low')) |
        ((df['weekly_watchlist_exit_setting'] == "Lower High") & (df['weekly_bxtrender_trend'] == 'lower_high'))
    )
    
    strategy_exit_condition = np.where(
        df['hourly_swing_strategy_enabled'], hourly_swing_exit_condition,
        np.where(
            df['weekly_watchlist_strategy_enabled'], weekly_watchlist_exit_condition,
            np.where(df['bxtrender_strategy_enabled'], bx_exit_condition, False)
        )
    )
    
    return strategy_exit_condition

# Entry and exit conditions
def calculate_entry_exit_conditions(df):
    """Combine entry and exit checks with appropriate index shifts."""
    index = np.where((df['barstate_entry'] == "New Bar") & (df['strategy'] != "Weekly Watchlist"), 1, 0)
    entry_condition = (
        df['strategy_entry_condition'].shift(index) &
        df['mb_condition'].shift(index) &
        df['weekly_rsi_condition'].shift(index) &
        df['monthly_rsi_condition'].shift(index) &
        df['htf_bx_condition'].shift(index)
    )
    exit_condition = df['strategy_exit_condition'].shift(index)
    
    return entry_condition, exit_condition

# End of day trade check
def eod_trade_check(df):
    """Identify if trading near the close of day should trigger an entry."""
    eod_trade_check = False
    eod_trade_check_done = False
    
    if (df['barstate_entry'] == "Bar Confirmed") & (df['timeframe_period'] == "60") & df['trade_before_close'] & df['barstate_isrealtime']:
        if (df['hour'] == 15) & (df['minute'] == df['trade_before_close_minute']) & ~eod_trade_check_done:
            eod_trade_check = True
            eod_trade_check_done = True
        elif (df['hour'] == 15) & (df['minute'] > df['trade_before_close_minute']):
            eod_trade_check = False
        elif (df['hour'] != 15) & eod_trade_check_done:
            eod_trade_check = False
            eod_trade_check_done = False
    
    return eod_trade_check, eod_trade_check_done

# Bar confirmed
def calculate_barconfirmed(df):
    """Check if the current bar should be treated as confirmed."""
    barconfirmed = (
        (df['trade_before_close'] & (((df['hour'] != 15) | ~df['barstate_islastconfirmedhistory']) & df['barstate_isconfirmed']) |
        (df['barstate_isrealtime'] & (df['hour'] == 15) & df['eod_trade_check'])) |
        (~df['trade_before_close'] & df['barstate_isconfirmed'])
    )
    
    return barconfirmed

# Entry timing
def calculate_entry_timing(df):
    """Determine if the current bar allows entering a trade."""
    entry_timing = (
        ((df['barstate_entry'] == "Bar Confirmed") & (df['barconfirmed'])) |
        ((df['barstate_entry'] == "New Bar") & (df['barstate_isnew'] | (df['strategy'] == "Weekly Watchlist")))
    )
    
    return entry_timing

# Get trade data
def get_trade_data(df):
    """Extract indicator values used when sending alerts."""
    trade_data = {
        "daily_bx": df['daily_bxtrender'].round(2),
        "daily_bx_previous": df['daily_bxtrender_prev'].round(2),
        "weekly_bx": df['weekly_bxtrender'].round(2),
        "weekly_bx_previous": df['weekly_bxtrender_prev'].round(2),
        "monthly_bx": df['monthly_bxtrender'].apply(lambda x: None if pd.isna(x) else round(x, 2)),
        "monthly_bx_previous": df['monthly_bxtrender_prev'].apply(lambda x: None if pd.isna(x) else round(x, 2))
    }
    
    return trade_data

# Example DataFrame setup
df = pd.DataFrame({
    # Add your data columns here
})

# Calculate conditions
df['strategy_entry_condition'] = calculate_entry_conditions(df)
df['strategy_exit_condition'] = calculate_exit_conditions(df)
df['entry_condition'], df['exit_condition'] = calculate_entry_exit_conditions(df)
df['eod_trade_check'], df['eod_trade_check_done'] = eod_trade_check(df)
df['barconfirmed'] = calculate_barconfirmed(df)
df['entry_timing'] = calculate_entry_timing(df)
trade_data = get_trade_data(df)

import pandas as pd
import numpy as np

# Initialize variables
trend_filter_2_matched = False
trend_filter_3_matched = False
trend_filter_4_matched = False

entry_daily_trend = np.nan
entry_weekly_trend = np.nan
entry_monthly_trend = np.nan

trade_entry_line = None
trade_exit_line = None

take_profit_line = None
stop_loss_line = None

# Sample DataFrame setup
df = pd.DataFrame({
    'bar_index': range(100),
    'low': np.random.rand(100),
    'open': np.random.rand(100),
    'close': np.random.rand(100),
    'htf_bx_condition_1': np.random.choice([True, False], 100),
    'htf_bx_condition_2': np.random.choice([True, False], 100),
    'htf_bx_condition_3': np.random.choice([True, False], 100),
    'htf_bx_condition_4': np.random.choice([True, False], 100),
    'entry_condition': np.random.choice([True, False], 100),
    'is_post_backtest_trade': np.random.choice([True, False], 100),
    'entry_timing': np.random.choice([True, False], 100),
    'strategy_entry_condition': np.random.choice([True, False], 100),
    'barstate_entry': np.random.choice(['New Bar', 'Old Bar'], 100),
    'daily_bxtrender_trend': np.random.rand(100),
    'weekly_bxtrender_trend': np.random.rand(100),
    'monthly_bxtrender_trend': np.random.rand(100),
})

# Handle trade entry
in_trade = False
independent_trade_tracking = False
base_position_size = 1  # Example value
total_trades = 0
enable_strategy = True
strategy = "B-Xtrender"
enable_bx_stop_loss = True
bx_stop_loss = 5  # Example percentage
take_profit_percent = 10  # Example percentage
stop_loss_percent = 5  # Example percentage
alert_strategy_name = "Example Strategy"
alert_strategy_secret = "Secret"
get_trade_data = lambda: "{}"  # Placeholder function

for index, row in df.iterrows():
    if not in_trade and row['entry_condition'] and row['is_post_backtest_trade']:
        position_size = base_position_size
        trend_filter_text = ""

        if row['htf_bx_condition_1']:
            trend_filter_text += "TF1"
        if row['htf_bx_condition_2']:
            trend_filter_text += "+TF2" if trend_filter_text else "TF2"
        if row['htf_bx_condition_3']:
            trend_filter_text += "+TF3" if trend_filter_text else "TF3"
        if row['htf_bx_condition_4']:
            trend_filter_text += "+TF4" if trend_filter_text else "TF4"

        entry_price = row['open'] if row['barstate_entry'] == "New Bar" else row['close']
        if row['entry_timing']:
            in_trade = True
            if total_trades == 0:
                buy_and_hold_entry_price = entry_price
            if enable_strategy:
                # Simulate strategy entry
                pass
            bars_in_trade = 0
            eod_trade_check = False

            if strategy == "B-Xtrender":
                if enable_bx_stop_loss:
                    stop_loss_level = entry_price * (1 - bx_stop_loss / 100)
                trade_entry_line = (index, entry_price)
                trade_exit_line = (index, entry_price)
            elif strategy == "Hourly Swing/Quant Bot":
                take_profit_level = entry_price * (1 + take_profit_percent / 100)
                stop_loss_level = entry_price * (1 - stop_loss_percent / 100)
                take_profit_line = (index, take_profit_level)
                stop_loss_line = (index, stop_loss_level)

            entry_daily_trend = row['daily_bxtrender_trend']
            entry_weekly_trend = row['weekly_bxtrender_trend']
            entry_monthly_trend = row['monthly_bxtrender_trend']

            trend_filter_1_matched = row['htf_bx_condition_1']
            trend_filter_2_matched = row['htf_bx_condition_2']
            trend_filter_3_matched = row['htf_bx_condition_3']
            trend_filter_4_matched = row['htf_bx_condition_4']

            # Simulate alert
            alert_msg = {
                "msg": "📈🔵 ENTRY",
                "type": "stock",
                "action": "buy",
                "price": entry_price,
                "ticker": "TICKER",
                "time": pd.Timestamp.now(),
                "strategy_name": alert_strategy_name,
                "strategy_secret": alert_strategy_secret,
                "trade_data": get_trade_data()
            }
            # print(alert_msg)

    # Independent tracking
    core_entry_condition = row['strategy_entry_condition']
    if not independent_trade_tracking and core_entry_condition and row['is_post_backtest_trade'] and row['entry_timing']:
        independent_trade_tracking = True
        independent_entry_price = row['open'] if row['barstate_entry'] == "New Bar" else row['close']
        independent_bars_in_trade = 0

    if in_trade:
        if strategy == "B-Xtrender":
            trade_entry_line = (trade_entry_line[0], index)
            trade_exit_line = (trade_exit_line[0], index)
            trade_exit_line = (trade_exit_line[0], row['close'])

import pandas as pd
import numpy as np

# Assuming df is your DataFrame with necessary columns
def handle_trade_exit(df, strategy, entry_price, current_account_size, position_size, peak_account_size, max_account_drawdown_pct, trend_filters):
    """Update statistics when an active trade closes."""
    for index, row in df.iterrows():
        if row['in_trade'] and row['exit_condition'] and row['bars_in_trade'] > 0:
            exit_price = row['open'] if row['barstate_entry'] == "New Bar" else row['close']
            exit_percentage = (exit_price - entry_price) / entry_price * 100

            # Create a label (for visualization purposes, not directly translatable)
            label_text = f"{exit_percentage:.2f}%\n{'O: ' if row['barstate_entry'] == 'New Bar' else 'C: '}{exit_price:.2f}"
            label_color = 'green' if exit_percentage > 0 else 'red' if row['barconfirmed'] or row['barstate_entry'] == "New Bar" else 'purple'

            if row['entry_timing']:
                row['in_trade'] = False
                if row['enable_strategy']:
                    # Close strategy logic here
                    pass
                row['eod_trade_check'] = False

                if strategy == "B-Xtrender":
                    row['trade_exit_line_y1'] = exit_price
                    row['trade_exit_line_y2'] = exit_price
                    row['trade_exit_line_color'] = 'green' if exit_price > entry_price else 'red'
                elif strategy == "Hourly Swing/Quant Bot":
                    row['take_profit_line'] = np.nan
                    row['stop_loss_line'] = np.nan
                    row['take_profit_level'] = np.nan
                    row['stop_loss_level'] = np.nan

                # Clear independent tracking
                row['independent_trade_tracking'] = False
                row['independent_entry_price'] = np.nan
                row['independent_bars_in_trade'] = 0

                # Calculate profit/loss in dollars
                trade_pnl = current_account_size * (position_size / 100) * (exit_percentage / 100)
                current_account_size += trade_pnl

                # Update peak and drawdown
                peak_account_size = max(peak_account_size, current_account_size)
                max_account_drawdown_pct = min(max_account_drawdown_pct, (current_account_size - peak_account_size) / peak_account_size * 100)

                # Update statistics for each matching trend filter
                for i, trend_filter in enumerate(trend_filters, start=1):
                    if row[f'trend_filter_{i}_matched']:
                        trend_filter['trades'] += 1
                        trend_filter['profit'] += exit_percentage
                        if exit_percentage > 0:
                            trend_filter['wins'] += 1
                            trend_filter['win_sum'] += exit_percentage
                        else:
                            trend_filter['losses'] += 1
                            trend_filter['loss_sum'] += exit_percentage

    return df, current_account_size, peak_account_size, max_account_drawdown_pct, trend_filters

# Example usage
trend_filters = [{'trades': 0, 'profit': 0, 'wins': 0, 'win_sum': 0, 'losses': 0, 'loss_sum': 0} for _ in range(4)]
df, current_account_size, peak_account_size, max_account_drawdown_pct, trend_filters = handle_trade_exit(
    df, "B-Xtrender", entry_price, current_account_size, position_size, peak_account_size, max_account_drawdown_pct, trend_filters
)

import pandas as pd
import numpy as np

# Initialize variables
trend_filter_4_wins = 0
trend_filter_4_win_sum = 0
trend_filter_4_losses = 0
trend_filter_4_loss_sum = 0
total_trades = 0
winning_trades = 0
losing_trades = 0
cumulative_profit_loss = 0
sum_wins = 0
sum_losses = 0
max_profit = -np.inf
max_loss = np.inf
sum_profit = 0
sum_loss = 0
peak_value = -np.inf
max_drawdown = np.inf
bars_in_trade = 0
independent_trade_tracking = False
independent_bars_in_trade = 0
send_htf_mismatch_alerts = True

# Example data
exit_percentage = 5  # Example value
exit_price = 100  # Example value
timenow = pd.Timestamp.now()
alert_strategy_name = "Example Strategy"
alert_strategy_secret = "Secret"
entry_timing = True
exit_condition = True
in_trade = True
independent_entry_price = 95  # Example value
barstate_entry = "New Bar"  # Example value

# Update statistics based on exit percentage
if exit_percentage > 0:
    trend_filter_4_wins += 1
    trend_filter_4_win_sum += exit_percentage
    exit_reason = "Take Profit"
    total_trades += 1
    winning_trades += 1
    cumulative_profit_loss += exit_percentage
    sum_wins += exit_percentage
    avg_win = sum_wins / winning_trades
    max_profit = max(max_profit, exit_percentage)
    sum_profit += exit_percentage
else:
    trend_filter_4_losses += 1
    trend_filter_4_loss_sum += exit_percentage
    exit_reason = "Stop Loss"
    total_trades += 1
    losing_trades += 1
    cumulative_profit_loss += exit_percentage
    sum_losses += exit_percentage
    avg_loss = sum_losses / losing_trades
    max_loss = min(max_loss, exit_percentage)
    sum_loss += exit_percentage

# Update peak value and max drawdown
peak_value = max(peak_value, cumulative_profit_loss)
max_drawdown = min(max_drawdown, cumulative_profit_loss - peak_value)

# Alert function (placeholder)
def alert(message):
    """Placeholder alert mechanism."""
    print(message)

# Send alert
alert(f"{{\"msg\":\"📈🟢 EXIT {exit_percentage:.2f}%\",\"type\":\"type\",\"action\":\"sell\",\"price\":{exit_price},\"ticker\":\"ticker\",\"time\":\"{timenow}\",\"strategy_name\":\"{alert_strategy_name}\",\"strategy_secret\":\"{alert_strategy_secret}\",\"profit\":{exit_percentage:.2f},\"bars_in_trade\":{bars_in_trade},\"trade_data\":\"trade_data\"}}")

# Update bars in trade
if in_trade:
    bars_in_trade += 1

# Detect HTF condition mismatch for repainting issues
if independent_trade_tracking and not in_trade and exit_condition and independent_bars_in_trade > 0 and entry_timing:
    mismatch_exit_price = open if barstate_entry == "New Bar" else close
    mismatch_exit_percentage = (mismatch_exit_price - independent_entry_price) / independent_entry_price * 100

    # Send exit alert for HTF condition mismatch
    if send_htf_mismatch_alerts:
        alert(f"{{\"msg\":\"📈🟡 HTF MISMATCH EXIT {mismatch_exit_percentage:.2f}%\",\"type\":\"type\",\"action\":\"sell\",\"price\":{mismatch_exit_price},\"ticker\":\"ticker\",\"time\":\"{timenow}\",\"strategy_name\":\"{alert_strategy_name}\",\"strategy_secret\":\"{alert_strategy_secret}\",\"profit\":{mismatch_exit_percentage:.2f},\"bars_in_trade\":{independent_bars_in_trade},\"reason\":\"HTF Condition Mismatch\",\"trade_data\":\"trade_data\"}}")

    # Clear independent tracking after sending alert
    independent_trade_tracking = False

import pandas as pd
import numpy as np

# Initialize variables
independent_entry_price = np.nan
independent_bars_in_trade = 0

# Sample data for demonstration
data = pd.DataFrame({
    'independent_trade_tracking': [True, False, True, True],
    'barstate_islast': [False, False, False, True],
    'strategy': ["Hourly Swing/Quant Bot", "Weekly Watchlist", "B-Xtrender", "Hourly Swing/Quant Bot"],
    'hourly_swing_strategy_enabled': [False, True, True, False],
    'weekly_watchlist_strategy_enabled': [True, False, True, True],
    'bxtrender_strategy_enabled': [True, True, False, True],
    'trend_filter_1_trades': [10, 0, 5, 8],
    'trend_filter_1_wins': [5, 0, 3, 4],
    'trend_filter_1_win_sum': [50, 0, 30, 40],
    'trend_filter_1_losses': [5, 0, 2, 4],
    'trend_filter_1_loss_sum': [-25, 0, -10, -20],
    'trend_filter_2_trades': [8, 0, 4, 6],
    'trend_filter_2_wins': [4, 0, 2, 3],
    'trend_filter_2_win_sum': [40, 0, 20, 30],
    'trend_filter_2_losses': [4, 0, 2, 3],
    'trend_filter_2_loss_sum': [-20, 0, -10, -15],
    'trend_filter_3_trades': [6, 0, 3, 5],
    'trend_filter_3_wins': [3, 0, 1, 2],
    'trend_filter_3_win_sum': [30, 0, 10, 20],
    'trend_filter_3_losses': [3, 0, 2, 3],
    'trend_filter_3_loss_sum': [-15, 0, -10, -15],
    'trend_filter_4_trades': [4, 0, 2, 3],
    'trend_filter_4_wins': [2, 0, 1, 1],
    'trend_filter_4_win_sum': [20, 0, 10, 10],
    'trend_filter_4_losses': [2, 0, 1, 2],
    'trend_filter_4_loss_sum': [-10, 0, -5, -10],
    'total_trades': [20, 0, 10, 15],
    'winning_trades': [10, 0, 5, 7],
    'show_stats_table': [True, False, True, True],
    'enable_trend_filter_1': [True, False, True, True],
    'enable_trend_filter_2': [True, False, True, True],
    'enable_trend_filter_3': [True, False, True, True],
    'enable_trend_filter_4': [True, False, True, True],
    'show_bxtrender_rows': [True, False, True, True],
    'stats_table_position': ["top_right", "top_left", "top_right", "top_left"]
})

# Update independent tracking
data['independent_bars_in_trade'] = data['independent_trade_tracking'].cumsum()

# Update table on each bar
def update_table(row):
    """Generate a row of statistics for the display table."""
    warning_text = ""
    if row['strategy'] == "Hourly Swing/Quant Bot" and not row['hourly_swing_strategy_enabled']:
        warning_text = "Hourly Swing requires 1h timeframe and Quant Bot requires 15m timeframe"
    elif row['strategy'] == "Weekly Watchlist" and not row['weekly_watchlist_strategy_enabled']:
        warning_text = "Weekly Watchlist requires daily timeframe"
    elif row['strategy'] == "B-Xtrender" and not row['bxtrender_strategy_enabled']:
        warning_text = "B-Xtrender requires at least 1h timeframe"

    trend_1_win_rate = (row['trend_filter_1_wins'] / row['trend_filter_1_trades'] * 100) if row['trend_filter_1_trades'] > 0 else 0.0
    trend_2_win_rate = (row['trend_filter_2_wins'] / row['trend_filter_2_trades'] * 100) if row['trend_filter_2_trades'] > 0 else 0.0
    trend_3_win_rate = (row['trend_filter_3_wins'] / row['trend_filter_3_trades'] * 100) if row['trend_filter_3_trades'] > 0 else 0.0
    trend_4_win_rate = (row['trend_filter_4_wins'] / row['trend_filter_4_trades'] * 100) if row['trend_filter_4_trades'] > 0 else 0.0

    trend_1_avg_win = (row['trend_filter_1_win_sum'] / row['trend_filter_1_wins']) if row['trend_filter_1_wins'] > 0 else 0.0
    trend_1_avg_loss = (row['trend_filter_1_loss_sum'] / row['trend_filter_1_losses']) if row['trend_filter_1_losses'] > 0 else 0.0
    trend_2_avg_win = (row['trend_filter_2_win_sum'] / row['trend_filter_2_wins']) if row['trend_filter_2_wins'] > 0 else 0.0
    trend_2_avg_loss = (row['trend_filter_2_loss_sum'] / row['trend_filter_2_losses']) if row['trend_filter_2_losses'] > 0 else 0.0
    trend_3_avg_win = (row['trend_filter_3_win_sum'] / row['trend_filter_3_wins']) if row['trend_filter_3_wins'] > 0 else 0.0
    trend_3_avg_loss = (row['trend_filter_3_loss_sum'] / row['trend_filter_3_losses']) if row['trend_filter_3_losses'] > 0 else 0.0
    trend_4_avg_win = (row['trend_filter_4_win_sum'] / row['trend_filter_4_wins']) if row['trend_filter_4_wins'] > 0 else 0.0
    trend_4_avg_loss = (row['trend_filter_4_loss_sum'] / row['trend_filter_4_losses']) if row['trend_filter_4_losses'] > 0 else 0.0

    trend_1_rr = abs(trend_1_avg_win) / abs(trend_1_avg_loss) if trend_1_avg_loss != 0 else 0.0
    trend_2_rr = abs(trend_2_avg_win) / abs(trend_2_avg_loss) if trend_2_avg_loss != 0 else 0.0
    trend_3_rr = abs(trend_3_avg_win) / abs(trend_3_avg_loss) if trend_3_avg_loss != 0 else 0.0
    trend_4_rr = abs(trend_4_avg_win) / abs(trend_4_avg_loss) if trend_4_avg_loss != 0 else 0.0

    win_rate = (row['winning_trades'] / row['total_trades'] * 100) if row['total_trades'] > 0 else 0.0

    if row['show_stats_table']:
        enabled_filters = (3 if row['enable_trend_filter_1'] else 0) + \
                          (3 if row['enable_trend_filter_2'] else 0) + \
                          (3 if row['enable_trend_filter_3'] else 0) + \
                          (3 if row['enable_trend_filter_4'] else 0)
        total_rows = 15 + (5 if row['show_bxtrender_rows'] else 0) + 1 + enabled_filters + 13 + 7

        # Simulate table creation
        stats_table_position = "top_right" if row['stats_table_position'] == "top_right" else "top_left"
        table_header = "B-Xtrender Bot - THT @chewy"

    return {
        'warning_text': warning_text,
        'trend_1_win_rate': trend_1_win_rate,
        'trend_2_win_rate': trend_2_win_rate,
        'trend_3_win_rate': trend_3_win_rate,
        'trend_4_win_rate': trend_4_win_rate,
        'trend_1_avg_win': trend_1_avg_win,
        'trend_1_avg_loss': trend_1_avg_loss,
        'trend_2_avg_win': trend_2_avg_win,
        'trend_2_avg_loss': trend_2_avg_loss,
        'trend_3_avg_win': trend_3_avg_win,
        'trend_3_avg_loss': trend_3_avg_loss,
        'trend_4_avg_win': trend_4_avg_win,
        'trend_4_avg_loss': trend_4_avg_loss,
    }

# Initialize current row
current_row = 1

# Create a DataFrame to simulate the table
stats_table = pd.DataFrame(columns=["Column1", "Column2"])

# Function to add a row to the DataFrame
def add_row(df, col1, col2, current_row):
    """Append a row to the stats table and return the next index."""
    df.loc[current_row] = [col1, col2]
    return current_row + 1

# Strategy row
strategy_color = {
    "Hourly Swing/Quant Bot": "rgb(89, 169, 235)",
    "Weekly Watchlist": "orange"
}.get(strategy, "rgb(235, 22, 206)")

current_row = add_row(stats_table, "Strategy", strategy, current_row)

# Ticker and Backtest Info
current_row = add_row(stats_table, "Symbol", syminfo.ticker, current_row)
current_row = add_row(stats_table, "Timeframe", timeframe.period, current_row)

backtest_date = f"{backtest_time:%Y-%m-%d}" if use_backtest_date else "Disabled"
backtest_stop_date = f"{backtest_stop_time:%Y-%m-%d}" if use_backtest_stop_date else "Disabled"
current_row = add_row(stats_table, "Backtest Date", f"{backtest_date} | {backtest_stop_date}", current_row)

# Entry and Exit Settings
current_row = add_row(stats_table, "Trade Timing", barstate_entry, current_row)

if strategy == "B-Xtrender":
    entry_exit_condition = f"{entry_setting} | {exit_setting}"
elif strategy == "Hourly Swing/Quant Bot":
    entry_exit_condition = f"{take_profit_percent:.2f}%/{tp_min_bars_to_exit} | -{stop_loss_percent:.2f}%/{sl_min_bars_to_exit} | {candle_delay}"
elif strategy == "Weekly Watchlist":
    entry_exit_condition = weekly_watchlist_exit_setting

current_row = add_row(stats_table, "Entry | Exit Condition", entry_exit_condition, current_row)

if enable_bx_stop_loss:
    current_row = add_row(stats_table, "Stop Loss", f"{bx_stop_loss:.2f}%", current_row)

# B-Xtrender Values (only shown if enabled)
if show_bxtrender_rows:
    current_row = add_row(stats_table, "B-Xtrender Values", "", current_row)
    hourly_bxtrender_color = "green" if hourly_bxtrender > 0 else "red"
    current_row = add_row(stats_table, "Hourly", f"{hourly_bxtrender:.2f}", current_row)

# Display the DataFrame
print(stats_table)

import pandas as pd

# Define colors
color_green = (0, 255, 0, 20)
color_red = (255, 0, 0, 20)
color_yellow = (255, 255, 0, 20)
color_white = (255, 255, 255)
color_teal = (0, 128, 128)
table_header_color = (50, 50, 50)
table_text_size = 12

# Define trend
trend_higher_low = "higher_low"

# Sample data
hourly_bxtrender = 1
daily_bxtrender = 1
weekly_bxtrender = 1
monthly_bxtrender = 1
weekly_bxtrender_trend = trend_higher_low
monthly_bxtrender_trend = trend_higher_low
enable_trend_filter_1 = True
enable_trend_filter_2 = True
show_bx_filter_stats = True
trend_filter_1_trades = 10
trend_1_win_rate = 75.0
trend_filter_1_profit = 5.0
trend_1_avg_win = 2.0
trend_1_avg_loss = 1.0
trend_1_rr = 2.0
daily_bxtrender_trend_settings_1 = "Setting1"
weekly_bxtrender_trend_settings_1 = "Setting2"
monthly_bxtrender_trend_settings_1 = "Setting3"
daily_bxtrender_trend_settings_2 = "Setting4"
weekly_bxtrender_trend_settings_2 = "Setting5"
monthly_bxtrender_trend_settings_2 = "Setting6"

# Initialize DataFrame for table
stats_table = pd.DataFrame(columns=["Label", "Value"])
current_row = 0

# Hourly
hourly_bxtrender_color = color_green if hourly_bxtrender > 0 else color_red
stats_table.loc[current_row] = ["Hourly", f"{hourly_bxtrender:.2f}"]
current_row += 1

# Daily
daily_bxtrender_color = color_green if daily_bxtrender > 0 else color_red
stats_table.loc[current_row] = ["Daily", f"{daily_bxtrender:.2f}"]
current_row += 1

# Weekly
weekly_bxtrender_color = color_green if weekly_bxtrender > 0 else (color_yellow if weekly_bxtrender_trend == trend_higher_low else color_red)
stats_table.loc[current_row] = ["Weekly", f"{weekly_bxtrender:.2f}"]
current_row += 1

# Monthly
monthly_bxtrender_color = color_green if monthly_bxtrender > 0 else (color_yellow if monthly_bxtrender_trend == trend_higher_low else color_red)
stats_table.loc[current_row] = ["Monthly", f"{monthly_bxtrender:.2f}"]
current_row += 1

# Filters
stats_table.loc[current_row] = ["Filters", ""]
current_row += 1

# Trend Filter 1
if enable_trend_filter_1:
    stats_table.loc[current_row] = ["BX Filter 1", f"{daily_bxtrender_trend_settings_1} | {weekly_bxtrender_trend_settings_1} | {monthly_bxtrender_trend_settings_1}"]
    current_row += 1
    if show_bx_filter_stats:
        stats_table.loc[current_row] = ["BXF1 Trades", f"{trend_filter_1_trades} trades, {trend_1_win_rate:.2f}% WR, {trend_filter_1_profit:.2f}% P/L"]
        current_row += 1
        stats_table.loc[current_row] = ["BXF1 Avg Win/Loss", f"{trend_1_avg_win:.2f}% | {trend_1_avg_loss:.2f}% (R/R: {trend_1_rr:.2f})"]
        current_row += 1

# Trend Filter 2
if enable_trend_filter_2:
    stats_table.loc[current_row] = ["BX Filter 2", f"{daily_bxtrender_trend_settings_2} | {weekly_bxtrender_trend_settings_2} | {monthly_bxtrender_trend_settings_2}"]
    current_row += 1

    if show_bx_filter_stats:
        stats_table.loc[current_row] = ["BXF2 Trades", ""]  # Add appropriate values here

# Display the table
print(stats_table)

import pandas as pd

def update_table(stats_table, current_row, trend_filter_2_trades, trend_2_win_rate, trend_filter_2_profit, trend_2_avg_win, trend_2_avg_loss, trend_2_rr,
                 enable_trend_filter_3, daily_bxtrender_trend_settings_3, weekly_bxtrender_trend_settings_3, monthly_bxtrender_trend_settings_3,
                 show_bx_filter_stats, trend_filter_3_trades, trend_3_win_rate, trend_filter_3_profit, trend_3_avg_win, trend_3_avg_loss, trend_3_rr,
                 enable_trend_filter_4, daily_bxtrender_trend_settings_4, weekly_bxtrender_trend_settings_4, monthly_bxtrender_trend_settings_4,
                 trend_filter_4_trades, trend_4_win_rate, trend_filter_4_profit, trend_4_avg_win, trend_4_avg_loss, trend_4_rr,
                 enable_weekly_rsi_filter, enable_monthly_rsi_filter, table_text_size):
    """Populate the statistics table with trend filter rows and settings."""

    def add_row(stats_table, current_row, col0, col1, text_color, text_size):
        stats_table.loc[current_row] = [col0, col1, text_color, text_size]
        return current_row + 1

    current_row = add_row(stats_table, current_row, "BXF2 Trades", 
                          f"{trend_filter_2_trades} trades, {trend_2_win_rate:.2f}% WR, {trend_filter_2_profit:.2f}% P/L", 
                          "white", table_text_size)
    
    current_row = add_row(stats_table, current_row, "BXF2 Avg Win/Loss", 
                          f"{trend_2_avg_win:.2f}% | {trend_2_avg_loss:.2f}% (R/R: {trend_2_rr:.2f})", 
                          "white", table_text_size)

    if enable_trend_filter_3:
        current_row = add_row(stats_table, current_row, "BX Filter 3", 
                              f"{daily_bxtrender_trend_settings_3} | {weekly_bxtrender_trend_settings_3} | {monthly_bxtrender_trend_settings_3}", 
                              "teal", table_text_size)
        
        if show_bx_filter_stats:
            current_row = add_row(stats_table, current_row, "BXF3 Trades", 
                                  f"{trend_filter_3_trades} trades, {trend_3_win_rate:.2f}% WR, {trend_filter_3_profit:.2f}% P/L", 
                                  "white", table_text_size)
            
            current_row = add_row(stats_table, current_row, "BXF3 Avg Win/Loss", 
                                  f"{trend_3_avg_win:.2f}% | {trend_3_avg_loss:.2f}% (R/R: {trend_3_rr:.2f})", 
                                  "white", table_text_size)

    if enable_trend_filter_4:
        current_row = add_row(stats_table, current_row, "BX Filter 4", 
                              f"{daily_bxtrender_trend_settings_4} | {weekly_bxtrender_trend_settings_4} | {monthly_bxtrender_trend_settings_4}", 
                              "teal", table_text_size)
        
        if show_bx_filter_stats:
            current_row = add_row(stats_table, current_row, "BXF4 Trades", 
                                  f"{trend_filter_4_trades} trades, {trend_4_win_rate:.2f}% WR, {trend_filter_4_profit:.2f}% P/L", 
                                  "white", table_text_size)
            
            current_row = add_row(stats_table, current_row, "BXF4 Avg Win/Loss", 
                                  f"{trend_4_avg_win:.2f}% | {trend_4_avg_loss:.2f}% (R/R: {trend_4_rr:.2f})", 
                                  "white", table_text_size)

    rsi_status = f"{'Enabled' if enable_weekly_rsi_filter else 'Disabled'} | {'Enabled' if enable_monthly_rsi_filter else 'Disabled'}"
    rsi_color = "teal" if enable_weekly_rsi_filter or enable_monthly_rsi_filter else "white"
    current_row = add_row(stats_table, current_row, "RSI Filter - W | M", rsi_status, rsi_color, table_text_size)

    current_row = add_row(stats_table, current_row, "MB Filter - D | W | M", "", "white", table_text_size)

    return stats_table

# Example usage
stats_table = pd.DataFrame(columns=["Column 0", "Column 1", "Text Color", "Text Size"])
current_row = 0
table_text_size = 12

# Call the function with appropriate parameters
stats_table = update_table(stats_table, current_row, 10, 75.5, 5.0, 2.5, 1.5, 1.67, 
                           True, "Daily Setting 3", "Weekly Setting 3", "Monthly Setting 3", 
                           True, 15, 80.0, 6.0, 3.0, 1.8, 1.75, 
                           True, "Daily Setting 4", "Weekly Setting 4", "Monthly Setting 4", 
                           20, 85.0, 7.0, 3.5, 2.0, 1.8, 
                           True, False, table_text_size)

print(stats_table)

import pandas as pd

# Initialize a DataFrame to simulate the table
stats_table = pd.DataFrame(columns=['Description', 'Value'])

# Function to add a row to the table
def add_row(description, value, text_color, text_size):
    """Helper used by the stats output examples."""
    global stats_table
    stats_table = stats_table.append({'Description': description, 'Value': value}, ignore_index=True)

# Function to merge cells (not applicable in pandas, but we can simulate by adding a header)
def merge_cells(description, bgcolor, text_size):
    """Add a header row to the stats table."""
    global stats_table
    stats_table = stats_table.append({'Description': description, 'Value': ''}, ignore_index=True)

# Variables (placeholders for actual values)
enable_mb_filter = True
daily_mb_settings = "Daily"
weekly_mb_settings = "Weekly"
monthly_mb_settings = "Monthly"
repaint_daily = True
repaint_weekly = False
enable_realtime_weekly_entries_condition = False
repaint_monthly = False
enable_realtime_monthly_entries_condition = False
total_trades = 100
winning_trades = 60
losing_trades = 40
avg_win = 5.5
avg_loss = -3.2
max_profit = 10.0
max_loss = -5.0
sum_profit = 550
sum_loss = -128
win_rate = 60.0
table_text_size = 12
table_header_color = 'gray'

# Adding rows to the table
add_row("Enabled - " + daily_mb_settings + " | " + weekly_mb_settings + " | " + monthly_mb_settings if enable_mb_filter else "Disabled", '', 'teal' if enable_mb_filter else 'white', table_text_size)

add_row("Repainting - D | W | M", '', 'white', table_text_size)
add_row(
    ("Enabled" if repaint_daily else "Disabled") + " | " +
    ("Enabled" if repaint_weekly or enable_realtime_weekly_entries_condition else "Disabled") + " | " +
    ("Enabled" if repaint_monthly or enable_realtime_monthly_entries_condition else "Disabled"),
    '',
    'yellow' if repaint_daily or repaint_weekly or repaint_monthly or enable_realtime_weekly_entries_condition or enable_realtime_monthly_entries_condition else 'green',
    table_text_size
)

merge_cells("Trade Statistics", table_header_color, table_text_size)

add_row("Total Trades", str(total_trades), 'white', table_text_size)
add_row("Wins", str(winning_trades), 'green', table_text_size)
add_row("Losses", str(losing_trades), 'red', table_text_size)
add_row("Avg Profit", f"{avg_win:.2f}%", 'green', table_text_size)
add_row("Avg Loss", f"{avg_loss:.2f}%", 'red', table_text_size)
add_row("Max Profit", f"{max_profit:.2f}%", 'green', table_text_size)
add_row("Max Loss", f"{max_loss:.2f}%", 'red', table_text_size)

merge_cells("Performance Metrics", table_header_color, table_text_size)

profit_factor = abs(sum_profit) / abs(sum_loss)
expected_value = (win_rate / 100 * avg_win) + ((100 - win_rate) / 100 * avg_loss)
risk_reward_ratio = abs(avg_win) / abs(avg_loss)

add_row("Win Rate", f"{win_rate:.2f}%", 'white', table_text_size)

# Display the table
print(stats_table)

import pandas as pd

def create_stats_table(data):
    """Return a small DataFrame summarizing account performance."""
    rows = []
    
    # Cumulative P/L
    cumulative_profit_loss = data['cumulative_profit_loss']
    text_color = 'green' if cumulative_profit_loss >= 0 else 'red'
    rows.append(["Cumulative P/L", f"{cumulative_profit_loss:.2f}%", text_color])
    
    # Max Drawdown
    max_drawdown = abs(data['max_drawdown'])
    rows.append(["Max Drawdown", f"{max_drawdown:.2f}%", 'red'])
    
    # Expected Value
    expected_value = data['expected_value']
    text_color = 'green' if expected_value >= 0 else 'red'
    rows.append(["Expected Value", f"{expected_value:.2f}%", text_color])
    
    # Risk/Reward Ratio
    risk_reward_ratio = data['risk_reward_ratio']
    text_color = 'green' if risk_reward_ratio >= 0 else 'red'
    rows.append(["Risk/Reward Ratio", f"{risk_reward_ratio:.2f}", text_color])
    
    # Profit Factor
    profit_factor = data['profit_factor']
    text_color = 'green' if profit_factor >= 1 else 'red'
    rows.append(["Profit Factor", f"{profit_factor:.2f}", text_color])
    
    # Account Statistics Section
    rows.append(["Account Statistics", "", 'white'])
    
    # Initial Account
    initial_account_size = data['initial_account_size']
    rows.append(["Initial Account", f"${initial_account_size:.2f}", 'white'])
    
    # Position Size
    base_position_size = data['base_position_size']
    rows.append(["Position Size", f"{base_position_size:.2f}%", 'white'])
    
    # Current Account
    current_account_size = data['current_account_size']
    text_color = 'green' if current_account_size >= initial_account_size else 'red'
    rows.append(["Current Account", f"${current_account_size:.2f}", text_color])
    
    # Peak Account
    peak_account_size = data['peak_account_size']
    rows.append(["Peak Account", f"${peak_account_size:.2f}", 'green'])
    
    # Max Drawdown
    max_account_drawdown_pct = abs(data['max_account_drawdown_pct'])
    rows.append(["Max Drawdown", f"{max_account_drawdown_pct:.2f}%", 'red'])
    
    # Total Return
    total_return = data['total_return']
    rows.append(["Total Return", f"{total_return:.2f}%", 'white'])
    
    return pd.DataFrame(rows, columns=["Metric", "Value", "Text Color"])

# Example usage
data = {
    'cumulative_profit_loss': 10.5,
    'max_drawdown': -5.2,
    'expected_value': 2.3,
    'risk_reward_ratio': 1.5,
    'profit_factor': 1.2,
    'initial_account_size': 10000,
    'base_position_size': 5,
    'current_account_size': 10500,
    'peak_account_size': 11000,
    'max_account_drawdown_pct': -3.5,
    'total_return': 5.0
}

stats_table = create_stats_table(data)
print(stats_table)

import pandas as pd

# Assuming the following variables are defined: current_account_size, initial_account_size, close, buy_and_hold_entry_price

# Calculate total return
total_return = ((current_account_size - initial_account_size) / initial_account_size) * 100

# Determine text color based on total return
total_return_color = 'green' if total_return >= 0 else 'red'

# Create a DataFrame to simulate the table
stats_table = pd.DataFrame(columns=['Description', 'Value', 'Text Color'])

# Add total return to the table
stats_table = stats_table.append({
    'Description': 'Total Return',
    'Value': f"{total_return:.2f}%",
    'Text Color': total_return_color
}, ignore_index=True)

# Calculate buy and hold return
buy_and_hold_return = ((close - buy_and_hold_entry_price) / buy_and_hold_entry_price) * 100

# Determine text color based on buy and hold return
buy_and_hold_return_color = 'green' if buy_and_hold_return >= 0 else 'red'

# Add buy and hold return to the table
stats_table = stats_table.append({
    'Description': 'Buy and Hold Return',
    'Value': f"{buy_and_hold_return:.2f}%",
    'Text Color': buy_and_hold_return_color
}, ignore_index=True)

# Display the table
print(stats_table)
