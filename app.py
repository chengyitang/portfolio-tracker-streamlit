import streamlit as st
import pandas as pd
import yfinance as yf
import os
import shutil
# --- Configuration ---
st.set_page_config(
    page_title="Portfolio Tracker MVP",
    page_icon="📈",
    layout="wide"
)

# --- Domain Logic ---

def normalize_ticker(ticker: str) -> str:
    """
    Normalizes stock ticker.
    - If all digits (e.g., '2330'), appends '.TW'.
    - Otherwise, returns as is.
    """
    ticker = str(ticker).strip().upper()
    if ticker.isdigit():
        return f"{ticker}.TW"
    return ticker

def detect_currency(symbol: str) -> str:
    """
    Detects currency based on suffix.
    - .TW / .TWO -> TWD
    - Others -> USD
    """
    if symbol.endswith('.TW') or symbol.endswith('.TWO'):
        return 'TWD'
    return 'USD'

def calculate_cash_flow(row):
    """
    Calculates Cash Flow if missing.
    """
    if pd.notna(row.get('Cash Flow')) and row['Cash Flow'] != 0:
        return row['Cash Flow']
    
    shares = row['Shares']
    price = row['Price']
    fee = row['Fee']
    
    if row['Type'] == 'Buy':
        return -(shares * price + fee)
    elif row['Type'] == 'Sell':
        return (shares * price - fee)
    elif row['Type'] == 'Dividend':
        return price 
    return 0.0

def process_ledger(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the raw ledger data.
    """
    df['Ticker'] = df['Ticker'].astype(str)
    df['Symbol'] = df['Ticker'].apply(normalize_ticker)
    df['Currency'] = df['Symbol'].apply(detect_currency)
    
    # Ensure Date
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce').fillna(0)
    df['Price'] = pd.to_numeric(df['Price'], errors='coerce').fillna(0.0)
    df['Fee'] = pd.to_numeric(df['Fee'], errors='coerce').fillna(0.0)
    df['Cash Flow'] = pd.to_numeric(df['Cash Flow'], errors='coerce')
    
    df['Cash Flow'] = df.apply(calculate_cash_flow, axis=1)
    
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_current_prices_and_fx(symbols: list, _log_func=None):
    """
    Fetch current prices and USDTWD exchange rate.
    _log_func: Optional callback for logging (underscore to denote it's not part of cache key if we wanted, 
               but st.cache_data hashes arguments. We can exclude it or just let it be None default).
    """
    if not symbols:
        return {}, 1.0
    
    # Add currency pair to fetch list
    fetch_list = symbols + ['USDTWD=X']
    tickers_str = " ".join(fetch_list)
    
    if _log_func:
        _log_func(f"Fetching data from Yahoo Finance: {tickers_str}")
    
    try:
        # Use 5d period to handle weekends/holidays better
        data = yf.download(tickers_str, period="5d", group_by='ticker', threads=True)
        prices = {}
        fx_rate = 30.0 # Default fallback
        
        # Helper to get last valid price from a series
        def get_last_valid(series):
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
            if series.empty:
                return 0.0
            valid = series.dropna()
            if valid.empty:
                return 0.0
            return float(valid.iloc[-1])

        if len(fetch_list) == 1:
            # Single ticker download structure is different (no top level ticker key)
            # data columns are 'Adj Close', 'Close', etc.
            # But fetch_list always has at least USDTWD=X. 
            # If symbols is empty, fetch_list=['USDTWD=X'], len=1.
            # So this block handles the case where ONLY FX is fetched (no stocks).
            try:
                rate_series = data['Close']
                fetched_rate = get_last_valid(rate_series)
                fx_rate = fetched_rate if fetched_rate > 0 else 30.0
            except Exception:
                pass
        else:
            for sym in symbols:
                try:
                    val = 0.0
                    if sym in data.columns or (sym in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else False):
                         # Handle MultiIndex
                         price_series = data[sym]['Close']
                         val = get_last_valid(price_series)
                    
                    prices[sym] = val
                    
                    if val == 0.0 and _log_func:
                        _log_func(f"⚠️ Warning: No recent price found for {sym} (Value is 0/NaN)")
                        
                except Exception as e:
                    prices[sym] = 0.0
                    if _log_func:
                         _log_func(f"❌ Error: Could not extract data for {sym}. Details: {e}")
            
            try:
                # FX
                if 'USDTWD=X' in data.columns or ('USDTWD=X' in data.columns.levels[0] if isinstance(data.columns, pd.MultiIndex) else False):
                    rate_series = data['USDTWD=X']['Close']
                    rate = get_last_valid(rate_series)
                    fx_rate = rate if rate > 0 else 30.0
            except (KeyError, IndexError):
                pass
                
        return prices, fx_rate
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return {}, 30.0

def aggregate_portfolio(ledger: pd.DataFrame, target_currency: str, fx_rate: float, log_func=None):
    """
    Aggregate ledger to portfolio view with currency conversion.
    Returns: (active_df, closed_df, applied_fx_rate)
    """
    if ledger.empty:
        return pd.DataFrame(), pd.DataFrame(), 30.0

    unique_symbols = ledger['Symbol'].unique().tolist()
    
    # Identify Active Symbols for Fetching (Shares > 0)
    buys = ledger[ledger['Type'] == 'Buy'].groupby('Symbol')['Shares'].sum()
    sells = ledger[ledger['Type'] == 'Sell'].groupby('Symbol')['Shares'].sum()
    net_shares = buys.sub(sells, fill_value=0)
    active_symbols = net_shares[net_shares > 0].index.tolist()
    
    if log_func:
        log_func(f"Found {len(active_symbols)} active symbols (out of {len(unique_symbols)} total). Preparing to fetch...")
        
    current_prices, fetched_fx = fetch_current_prices_and_fx(active_symbols, _log_func=log_func)
    
    real_fx = fetched_fx if fetched_fx > 0 else 30.0
    
    active_rows = []
    closed_rows = []
    
    grouped = ledger.groupby('Symbol')
    
    for symbol, group in grouped:
        stock_name = group.iloc[-1]['Stock Name'] if 'Stock Name' in group.columns else symbol
        native_currency = detect_currency(symbol)
        
        # Share aggregation
        total_buy_shares = group[group['Type'] == 'Buy']['Shares'].sum()
        total_sell_shares = group[group['Type'] == 'Sell']['Shares'].sum()
        current_shares = total_buy_shares - total_sell_shares
        
        # Common Calculations (Native)
        buy_txs = group[group['Type'] == 'Buy']
        sell_txs = group[group['Type'] == 'Sell']
        
        total_buy_cost_native = abs(buy_txs['Cash Flow'].sum())
        total_sell_proceeds_native = sell_txs['Cash Flow'].sum() # Sells are positive CF
        sum_cash_flows_native = group['Cash Flow'].sum()
        dividends_native = group[group['Type'] == 'Dividend']['Cash Flow'].sum()
        
        # Conversion Rate
        conversion_rate = 1.0
        if native_currency == 'USD' and target_currency == 'TWD':
            conversion_rate = real_fx
        elif native_currency == 'TWD' and target_currency == 'USD':
            conversion_rate = 1.0 / real_fx

        # --- Active Position Logic ---
        if current_shares > 0:
            total_buy_shares_all = buy_txs['Shares'].sum()
            avg_cost_native = total_buy_cost_native / total_buy_shares_all if total_buy_shares_all else 0.0
            
            curr_price_native = current_prices.get(symbol, 0.0)
            market_value_native = current_shares * curr_price_native
            
            # Capital Gain (Unrealized)
            capital_gain_native = (curr_price_native - avg_cost_native) * current_shares
            
            # Total Return (Unrealized + Realized + Divs) - This remains "Total P/L" for the position lifetime
            total_return_native = market_value_native + sum_cash_flows_native
            
            # Adjusted Avg Cost (Break Even Price)
            # (Total Spent - Total Returned) / Remaining Shares
            # Total Returned = Sell Proceeds + Dividends
            net_invested_native = total_buy_cost_native - total_sell_proceeds_native - dividends_native
            adj_avg_cost_native = net_invested_native / current_shares if current_shares else 0.0
            
            # Total % (Unrealized ROI)
            # (Current Price - Avg Cost) / Avg Cost
            total_pct = ((curr_price_native - avg_cost_native) / avg_cost_native * 100) if avg_cost_native else 0.0
            
            # Convert
            active_rows.append({
                'Symbol': symbol,
                'Stock Name': stock_name,
                'Currency': native_currency,
                'Shares': current_shares,
                'Avg. Cost': avg_cost_native * conversion_rate,
                'Current Price': curr_price_native * conversion_rate,
                'Market Value': market_value_native * conversion_rate,
                'Total Dividends': dividends_native * conversion_rate,
                'Capital Gain': capital_gain_native * conversion_rate,
                'Total Return': total_return_native * conversion_rate,
                'Adjusted Avg. Cost': adj_avg_cost_native * conversion_rate,
                'Total %': total_pct
            })
            
        # --- Closed Position Logic ---
        # If shares are 0, check if we had activity.
        # Ideally, we should check if total_buy_shares > 0 to ensure it was a real position.
        elif total_buy_shares > 0:
            # Fully Realized
            # Market Value is 0.
            # Total Return is just sum of cash flows.
            total_return_native = sum_cash_flows_native
            total_pct = (total_return_native / total_buy_cost_native * 100) if total_buy_cost_native else 0.0
            
            closed_rows.append({
                'Symbol': symbol,
                'Stock Name': stock_name,
                'Currency': native_currency,
                'Realized P/L': total_return_native * conversion_rate,
                'Total Return': total_return_native * conversion_rate, # Same as Realized P/L for closed
                'Total Dividends': dividends_native * conversion_rate,
                'Total Invested': total_buy_cost_native * conversion_rate,
                'Total %': total_pct
            })
        
    return pd.DataFrame(active_rows), pd.DataFrame(closed_rows), real_fx

def calculate_realized_trend(ledger: pd.DataFrame, target_currency: str, fx_rate: float):
    """
    Calculates realized gain trend by year for CLOSED positions only.
    """
    if ledger.empty:
        return pd.DataFrame()
        
    # 1. Identify Closed Symbols
    # Calculate Net Shares: Sum(Buy Shares) - Sum(Sell Shares)
    buys = ledger[ledger['Type'] == 'Buy'].groupby('Symbol')['Shares'].sum()
    sells = ledger[ledger['Type'] == 'Sell'].groupby('Symbol')['Shares'].sum()
    net_shares = buys.sub(sells, fill_value=0)
    
    # Filter for symbols with ~0 shares
    closed_symbols = net_shares[net_shares.abs() < 1e-6].index.tolist()
    
    if not closed_symbols:
        return pd.DataFrame()
        
    # 2. Filter ledger
    closed_df = ledger[ledger['Symbol'].isin(closed_symbols)].copy()
    closed_df = closed_df.sort_values('Date')
    
    yearly_gains = {}
    
    # 3. Calculate Realized P/L per transaction
    # We need to track cost basis per symbol
    symbol_state = {} # {symbol: {'shares': 0, 'total_cost': 0}}
    
    for _, row in closed_df.iterrows():
        sym = row['Symbol']
        year = row['Date'].year
        type_ = row['Type']
        
        if pd.isna(year):
            continue
            
        if sym not in symbol_state:
            symbol_state[sym] = {'shares': 0, 'total_cost': 0.0}
            
        state = symbol_state[sym]
        
        native_currency = row['Currency']
        conversion_rate = 1.0
        if native_currency == 'USD' and target_currency == 'TWD':
            conversion_rate = fx_rate
        elif native_currency == 'TWD' and target_currency == 'USD':
            conversion_rate = 1.0 / fx_rate

        if type_ == 'Buy':
            shares = row['Shares']
            cost = abs(row['Cash Flow']) # Buy is negative CF
            state['shares'] += shares
            state['total_cost'] += cost
            
        elif type_ == 'Sell':
            shares = row['Shares'] # Positive int from input? 
            # process_ledger doesn't change shares sign for input, but aggregate logic assumed checks type.
            # Let's assume input shares is positive.
            price = row['Price']
            fee = row['Fee']
            
            # Avg cost
            avg_cost = state['total_cost'] / state['shares'] if state['shares'] > 0 else 0.0
            
            # Calc Gain
            proceeds = (shares * price) - fee
            cost_basis = shares * avg_cost
            realized_gain_native = proceeds - cost_basis
            
            # Convert
            realized_gain = realized_gain_native * conversion_rate
            
            # Update State
            state['shares'] -= shares
            state['total_cost'] -= cost_basis # Remove proportional cost
            
            # Add to Year
            if year not in yearly_gains: yearly_gains[year] = 0.0
            yearly_gains[year] += realized_gain
            
        elif type_ == 'Dividend':
            # Dividend is pure gain
            gain_native = row['Cash Flow']
            gain = gain_native * conversion_rate
            
            if year not in yearly_gains: yearly_gains[year] = 0.0
            yearly_gains[year] += gain
            
    # Convert to DataFrame
    if not yearly_gains:
        return pd.DataFrame()
        
    trend_df = pd.DataFrame(list(yearly_gains.items()), columns=['Year', 'Realized Gain'])
    trend_df['Year'] = trend_df['Year'].astype(str) # For categorical axis
    trend_df = trend_df.set_index('Year').sort_index()
    return trend_df

# --- UI Layout ---

st.title("💰 Personal Portfolio Tracker")

import os
import shutil

# --- Configuration ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)

# ... (Previous imports code) ...

# --- Persistence Logic ---
def save_uploaded_files(uploaded_files):
    saved_paths = []
    for file in uploaded_files:
        file_path = os.path.join(DATA_DIR, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths

def load_saved_files():
    dataframes = []
    if os.path.exists(DATA_DIR):
        files = [f for f in os.listdir(DATA_DIR) if f.endswith(('.csv', '.xlsx'))]
        for file in files:
            file_path = os.path.join(DATA_DIR, file)
            try:
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path, dtype={'Ticker': str})
                else:
                    df = pd.read_excel(file_path, dtype={'Ticker': str})
                dataframes.append(df)
            except Exception as e:
                st.error(f"Error loading {file}: {e}")
    return dataframes

def clear_saved_data():
    if os.path.exists(DATA_DIR):
        for f in os.listdir(DATA_DIR):
            os.remove(os.path.join(DATA_DIR, f))
    st.cache_data.clear()
    if 'ledger' in st.session_state:
        del st.session_state['ledger']

# --- UI Layout ---

col_head1, col_head2 = st.columns([3, 1])
# Sidebar
st.sidebar.header("Settings")
target_currency = st.sidebar.radio("Display Currency", ["TWD", "USD"], index=0)


st.sidebar.header("Data Management")

# Download Template
with open("sample_us.csv", "rb") as file:
    st.sidebar.download_button(
        label="Download Sample Template (CSV)",
        data=file,
        file_name="ledger_template.csv",
        mime="text/csv"
    )

uploaded_files = st.sidebar.file_uploader("Add/Update Ledger(s)", type=["csv", "xlsx"], accept_multiple_files=True)

if uploaded_files:
    save_uploaded_files(uploaded_files)
    st.sidebar.success("Files saved!")
    st.rerun() # Reload to pick up new files

if st.sidebar.button("Clear Saved Data"):
    clear_saved_data()
    st.rerun()

# Auto-Load Logic
if 'ledger' not in st.session_state:
    dfs = load_saved_files()
    if dfs:
        combined_ledger = pd.concat(dfs, ignore_index=True)
        st.session_state['ledger'] = process_ledger(combined_ledger)
        st.sidebar.info(f"Loaded {len(dfs)} saved file(s).")

# Main Display
if 'ledger' in st.session_state:
    ledger = st.session_state['ledger']
    
    # Calc with Status
    with st.status("Updating Portfolio Data...", expanded=True) as status:
        st.write("Aggregating portfolio...")
        active_portfolio, closed_portfolio, fx_used = aggregate_portfolio(ledger, target_currency, 30.0, log_func=st.write)
        st.write("Calculations complete.")
        status.update(label="Portfolio Data Ready", state="complete", expanded=False)
    
    # Currency Symbol
    c_sym = "NT$" if target_currency == 'TWD' else "$"
    
    # 1. KPI Cards
    # Active Assets
    total_assets = active_portfolio['Market Value'].sum() if not active_portfolio.empty else 0.0
    
    # Total P/L (Active + Closed)
    active_pl = active_portfolio['Total Return'].sum() if not active_portfolio.empty else 0.0
    closed_pl = closed_portfolio['Total Return'].sum() if not closed_portfolio.empty else 0.0
    total_pl_all = active_pl + closed_pl
    
    # Dividends (Active + Closed)
    active_div = active_portfolio['Total Dividends'].sum() if not active_portfolio.empty else 0.0
    closed_div = closed_portfolio['Total Dividends'].sum() if not closed_portfolio.empty else 0.0
    total_div_all = active_div + closed_div
    
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Current Assets", f"{c_sym}{total_assets:,.2f}")
    kpi2.metric("Total Profit/Loss", f"{c_sym}{total_pl_all:,.2f}", delta_color="normal")
    kpi3.metric("Total Dividends", f"{c_sym}{total_div_all:,.2f}")
    kpi4.metric("FX Rate (USD/TWD)", f"{fx_used:,.2f}")
    
    # 2. Active Portfolio Table
    st.subheader(f"Active Portfolio ({target_currency})")
    
    if not active_portfolio.empty:
        column_config = {
            'Avg. Cost': st.column_config.NumberColumn(format="%.2f"),
            'Current Price': st.column_config.NumberColumn(format="%.2f"),
            'Market Value': st.column_config.NumberColumn(format="%.2f"),
            'Total Dividends': st.column_config.NumberColumn(format="%.2f"),
            'Capital Gain': st.column_config.NumberColumn(format="%.2f"),
            'Total Return': st.column_config.NumberColumn(format="%.2f"),
            'Adjusted Avg. Cost': st.column_config.NumberColumn(format="%.2f"),
            'Total %': st.column_config.NumberColumn(format="%.2f%%"),
        }
        
        def highlight_pl(val):
            color = '#2ECC71' if val > 0 else '#E74C3C' if val < 0 else 'inherit'
            return f'color: {color}'
            
        styled_active = active_portfolio.style.map(highlight_pl, subset=['Total Return', 'Capital Gain', 'Total %'])
        st.dataframe(styled_active, column_config=column_config, width='stretch')
        

    else:
        st.info("No active holdings.")

    # Charts Section (Placed between Active and Closed Tables)
    st.divider()
    st.subheader("Asset Allocation & Trends")
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        st.caption("Active Asset Allocation")
        if not active_portfolio.empty:
            import altair as alt
            base = alt.Chart(active_portfolio).encode(
                theta=alt.Theta("Market Value", stack=True),
                color=alt.Color("Symbol"),
                tooltip=["Symbol", "Market Value", "Total %", "Currency"]
            )
            pie = base.mark_arc(outerRadius=120)
            text = base.mark_text(radius=140).encode(
                text="Symbol",
                order=alt.Order("Market Value", sort="descending")
            )
            st.altair_chart(pie + text, width='stretch')
        else:
            st.info("No active assets to display.")
            
    with chart_col2:
        st.caption("Realized Gains Trend (Closed Positions Only)")
        trend_data = calculate_realized_trend(ledger, target_currency, fx_used)
        if not trend_data.empty:
            st.bar_chart(trend_data, width='stretch')
        else:
            st.info("No realized positions found.")


    # 3. Closed Portfolio Table
    if not closed_portfolio.empty:
        st.divider()
        st.subheader(f"Realized / Closed Portfolio ({target_currency})")
        
        closed_config = {
            'Realized P/L': st.column_config.NumberColumn(format="%.2f"),
            'Total Return': st.column_config.NumberColumn(format="%.2f"),
            'Total Dividends': st.column_config.NumberColumn(format="%.2f"),
            'Total Invested': st.column_config.NumberColumn(format="%.2f"),
            'Total %': st.column_config.NumberColumn(format="%.2f%%"),
        }
        
        def highlight_pl(val):
            color = '#2ECC71' if val > 0 else '#E74C3C' if val < 0 else 'inherit'
            return f'color: {color}'

        # Columns to show
        cols_to_show = ['Symbol', 'Stock Name', 'Realized P/L', 'Total Dividends', 'Total Invested', 'Total %']
        styled_closed = closed_portfolio[cols_to_show].style.map(highlight_pl, subset=['Realized P/L', 'Total %'])
        
        st.dataframe(styled_closed, column_config=closed_config, width='stretch')
        
else:
    st.info("Please upload a ledger file to get started.")
