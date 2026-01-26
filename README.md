# 💰 Personal Portfolio Tracker

A privacy-focused, local-first personal portfolio tracking tool.

## ✨ Features

*   **No Data Retention**: All your transaction data exists only in browser memory and is never stored on any server.
*   **Local Computation**: Ledger aggregation and profit/loss calculations are performed entirely on your computer.
*   **Real-time Quotes**: Integrates with Yahoo Finance API for real-time stock prices and exchange rates.
*   **US & TW Stocks Support**: Manages both US (USD) and Taiwan (TWD) stock portfolios.

## 🚀 Developer Instructions

Follow these steps to run the project locally:

### Prerequisites

Ensure your computer has Python 3.9 or higher installed.

### 1. Get the Code

```bash
git clone https://github.com/chengyitang/portfolio-tracker-streamlit.git
cd portfolio-tracker-streamlit
```

### 2. Create Virtual Environment (Recommended)

```bash
python3 -m venv venv
source venv/bin/activate  # macOS / Linux
# venv\Scripts\activate   # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Application

```bash
streamlit run app.py
```

Once the application starts, your browser will automatically open `http://localhost:8501`.

## 📂 File Structure

*   `app.py`: Main Streamlit application.
*   `convert_firstrade.py`: Tool script to convert Firstrade transaction records to the standard format.
*   `requirements.txt`: Project dependency list.
*   `sample_us.csv` / `sample_tw.csv`: Transaction record templates.
