from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_ollama import ChatOllama
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
import yfinance as yf # Library for stock prices.
from langchain_core.tools import tool

@tool
def price_plot(
    tickers: str,
    start_date: str = "2023-01-01",
    end_date: str = "2026-10-15",
    title: str = "Stock Price History"
) -> str:
    """Download historical stock prices and generate a comparison plot.
    
    Args:
        tickers: Stock ticker symbol or comma-separated symbols.
        start_date: Starting date in 'YYYY-MM-DD' format (e.g. '2023-01-01').
        end_date: Ending date in 'YYYY-MM-DD' format (e.g. '2026-10-15').
        title: Optional plot title.
    """
    if isinstance(tickers, list):
        tickers_list = tickers
    else:
        tickers_list = [t.strip().upper() for t in str(tickers).split(",")]

    fig = Figure(figsize=(10, 6))
    ax = fig.subplots()

    plots_created = 0
    for ticker in tickers_list:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if df.empty:
            continue
        ax.plot(df.index, df["Close"], label=f"{ticker} Close", linewidth=1.8)
        plots_created += 1

    if plots_created == 0:
        return f"Error: No data found for tickers {tickers_list} between {start_date} and {end_date}."

    plot_title = title if title != "Stock Price History" else f"Stock Prices: {', '.join(tickers_list)} ({start_date} to {end_date})"
    ax.set_title(plot_title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Price (USD)", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    filename = "price_plot.png"
    fig.savefig(filename, dpi=150)
    
    os.system(f"start {filename}")

    return f"Price plot created successfully and opened as '{filename}' for: {', '.join(tickers_list)}."


load_dotenv("../.env")

'''llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash",
    temperature=0,
)'''

llm = ChatOllama(model="qwen2.5:3b")

web_search = TavilySearch(
    max_results=5,
    tavily_api_key=os.getenv("TAVILY_API_KEY")
)

agent = create_agent(
    model=llm,
    tools=[web_search,price_plot],
    system_prompt = """
    You are a Financial Data Steward agent. Your primary duty is ensuring data integrity, accuracy, and clear governance for financial market analysis.

    When handling requests for stock symbols and prices:
    1. Validate Symbols: Verify that requested tickers are valid and clearly disambiguated.
    2. Date Integrity: Check that requested date ranges are logical (start_date < end_date) and account for trading days vs non-trading days.
    3. Data Governance & Provenance: Whenever pulling data or generating charts, state the data source (e.g. Yahoo Finance via yfinance), currency (USD), and data granularity (daily close).
    4. Anomaly Detection: Report any missing data points, extreme deviations, or warnings if a ticker returned empty records.

    If asked for a plot, call the price_plot tool with the ticker symbol as a string.
    """
)

result = agent.invoke({
    "messages": [
        {
            "role": "user",
            "content": "Give me a table of prices and a plot of the prices of stock ticker 'GOOG' from 2023 to 2026."
        }
    ]
})

print(result["messages"][-1].content)