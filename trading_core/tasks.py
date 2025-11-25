# trading_core/tasks.py
import os
import yfinance as yf
import pandas as pd
import requests
from django.db import transaction
from decimal import Decimal
from dotenv import load_dotenv
from trading_core.models import ConfiguredStock, Trade
from bot_app.views import calculate_rsi # Re-use the RSI calculation function
import logging

# --- CRITICAL FIXES FOR IMPORTS ---
# 1. Matplotlib backend for non-GUI environments (even if not directly plotting here, it's good practice)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt # Needed for the 'Agg' backend to be effective

# 2. Django's timezone for database-related timestamps (e.g., DateTimeField)
from django.utils import timezone as django_timezone 

# 3. Python's standard datetime and timezone for creating tz-aware datetime objects
from datetime import datetime, timezone as python_timezone 
# --- END CRITICAL FIXES ---

# Get an instance of a logger for this module
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

TRADING_API_KEY = os.getenv('TRADING_API_KEY')

TRADING_API_SECRET = os.getenv('TRADING_API_SECRET') # Load the secret key
TRADING_212_ORDERS_URL = "https://demo.trading212.com/api/v0/equity/orders/market"

def execute_automated_trading_task():
    """
    Core logic for the automated trading bot.
    Processes all active configured stocks, generates signals, and executes trades.
    This function is designed to be called periodically.
    """
    logger.info(f"[{django_timezone.now()}] Starting automated trading run...")

    # headers = {"Authorization": TRADING_API_KEY} if TRADING_API_KEY else {}
    
    is_simulation_mode = not TRADING_API_KEY

    active_configs = ConfiguredStock.objects.filter(is_active_for_trading=True)

    if not active_configs:
        logger.warning("No active configured stocks found for automated trading.")
        return

    for config in active_configs:
        logger.info(f"Processing {config.yf_ticker} ({config.trading212_ticker})...")

        with transaction.atomic():
            try:
                # 1. Fetch Data
                # Fetch a reasonable window (e.g., 5 days) to ensure enough history for RSI
                hist = yf.Ticker(config.yf_ticker).history(period="5d", interval="5m")

                if not hist.empty:
                    if hist.index.tz is None: hist.index = hist.index.tz_localize('UTC')
                    else: hist.index = hist.index.tz_convert('UTC')

                realtime_info = yf.Ticker(config.yf_ticker).info
                realtime_price = realtime_info.get('regularMarketPrice')

                if realtime_price is None:
                    logger.error(f"Could not fetch real-time price for {config.yf_ticker}. Skipping trade for this stock.")
                    config.last_run = django_timezone.now()
                    config.save()
                    continue
                
                current_price = Decimal(str(realtime_price))

                # Append real-time data
                realtime_data = {
                    'Open': realtime_price, 'High': realtime_price, 'Low': realtime_price,
                    'Close': realtime_price, 'Volume': 0, 'Dividends': 0, 'Stock Splits': 0
                }
                realtime_timestamp = pd.to_datetime(datetime.now(python_timezone.utc))
                realtime_df = pd.DataFrame([realtime_data], index=[realtime_timestamp])
                realtime_df.index.name = 'Date'

                # Concatenate, drop duplicates (to handle cases where yfinance's last interval overlaps with new real-time)
                # and sort to maintain chronological order
                if not hist.empty: hist = pd.concat([hist, realtime_df]).drop_duplicates(keep='last').sort_index()
                else: hist = realtime_df # If history was empty, realtime_df becomes the history
                
                if hist.empty:
                    logger.error(f"No valid historical or real-time data for {config.yf_ticker}. Skipping trade.")
                    config.last_run = django_timezone.now()
                    config.save()
                    continue

                # 2. Calculate RSI
                if 'Close' in hist.columns and len(hist) > config.rsi_window:
                    hist['RSI'] = calculate_rsi(hist['Close'], config.rsi_window)
                    latest_rsi = hist['RSI'].iloc[-1]
                    if pd.isna(latest_rsi):
                         logger.warning(f"RSI calculated as NaN for {config.yf_ticker}. Skipping trade.")
                         config.last_run = django_timezone.now()
                         config.save()
                         continue
                else:
                    logger.warning(f"Not enough data ({len(hist)} rows) to calculate RSI for {config.yf_ticker} (window: {config.rsi_window}). Skipping trade.")
                    config.last_run = django_timezone.now()
                    config.save()
                    continue

                # 3. Generate Signal
                signal = 'Hold'
                if latest_rsi < config.buy_threshold:
                    signal = 'Buy'
                elif latest_rsi > config.sell_threshold:
                    signal = 'Sell'

                logger.info(f"Latest RSI for {config.yf_ticker}: {latest_rsi:.2f}, Signal: {signal}")

                trade_response_text = "N/A"
                trade_status = "Simulated"
                trading212_order_id = None
                realized_pnl = None

                # Determine current net position (simplified)
                net_position_quantity = 0
                open_buy_trades = Trade.objects.filter(stock_config=config, signal_generated='Buy', status='Success', realized_pnl_for_trade__isnull=True)
                open_sell_trades = Trade.objects.filter(stock_config=config, signal_generated='Sell', status='Success', realized_pnl_for_trade__isnull=True)

                net_position_quantity = sum(t.quantity for t in open_buy_trades) - sum(t.quantity for t in open_sell_trades)
                
                avg_buy_price = Decimal('0.00')
                if net_position_quantity > 0:
                    total_buy_value = sum(t.quantity * t.price_at_execution for t in open_buy_trades if t.price_at_execution is not None)
                    if total_buy_value > 0 and net_position_quantity > 0:
                        avg_buy_price = total_buy_value / Decimal(net_position_quantity)


                # 4. Execute Trade if signal is Buy/Sell
                if signal == 'Buy':
                    # Simplified logic: always buy if signal, assuming you manage risk elsewhere
                    payload = {
                        "quantity": config.trade_quantity,
                        "ticker": config.trading212_ticker,
                    }
                    logger.debug(f"DEBUG: hist DataFrame is empty for {config.trade_quantity}.")
                    if not is_simulation_mode:
                        try:
                            response = requests.post(TRADING_212_ORDERS_URL, json=payload, auth=(TRADING_API_KEY,TRADING_API_SECRET), timeout=10)
                            response.raise_for_status()
                            trade_response_text = response.text
                            response_json = response.json()
                            trading212_order_id = response_json.get('orderId')
                            trade_status = 'Success'
                            logger.info(f"Successfully placed BUY order for {config.yf_ticker}. Order ID: {trading212_order_id}")
                        except requests.exceptions.RequestException as e:
                            trade_response_text = f"API Request Error: {e}"
                            trade_status = 'Failed'
                            logger.error(f"API BUY request failed for {config.yf_ticker}: {e}")
                        except Exception as e:
                            trade_response_text = f"Processing BUY response failed: {e}"
                            trade_status = 'Failed'
                            logger.error(f"Processing BUY response for {config.yf_ticker} failed: {e}")
                    else:
                        trade_response_text = f"Simulated BUY order for {config.trade_quantity} shares of {config.trading212_ticker} at {current_price}."
                        logger.warning(f"SIMULATION MODE: {trade_response_text}")

                elif signal == 'Sell':
                    if net_position_quantity >= config.trade_quantity: # Only sell if holding enough shares
                        payload = {
                            "quantity": -config.trade_quantity,
                            "ticker": config.trading212_ticker,
                        }
                        if not is_simulation_mode:
                            try:
                                response = requests.post(TRADING_212_ORDERS_URL, json=payload, headers=headers, timeout=10)
                                response.raise_for_status()
                                trade_response_text = response.text
                                response_json = response.json()
                                trading212_order_id = response_json.get('orderId')
                                trade_status = 'Success'
                                logger.info(f"Successfully placed SELL order for {config.yf_ticker}. Order ID: {trading212_order_id}")

                                if avg_buy_price > 0:
                                    realized_pnl = (current_price - avg_buy_price) * Decimal(config.trade_quantity)
                                    config.total_realized_pnl += realized_pnl
                                    config.save()
                                    logger.info(f"Realized P&L from sell: {realized_pnl:.2f}")

                            except requests.exceptions.RequestException as e:
                                trade_response_text = f"API Request Error: {e}"
                                trade_status = 'Failed'
                                logger.error(f"API SELL request failed for {config.yf_ticker}: {e}")
                            except Exception as e:
                                trade_response_text = f"Processing SELL response failed: {e}"
                                trade_status = 'Failed'
                                logger.error(f"Processing SELL response for {config.yf_ticker} failed: {e}")
                        else:
                            trade_response_text = f"Simulated SELL order for {config.trade_quantity} shares of {config.trading212_ticker} at {current_price}."
                            logger.warning(f"SIMULATION MODE: {trade_response_text}")
                            if avg_buy_price > 0:
                                realized_pnl = (current_price - avg_buy_price) * Decimal(config.trade_quantity)
                                config.total_realized_pnl += realized_pnl
                                config.save()
                                logger.warning(f"SIMULATION P&L: {realized_pnl:.2f}")

                    else:
                        trade_response_text = f"Skipped SELL for {config.yf_ticker}: Not enough shares to sell. Currently holding {net_position_quantity} shares."
                        signal = 'Hold'
                        logger.warning(trade_response_text)
                
                # 5. Log Trade (even if it was a 'Hold' or skipped)
                Trade.objects.create(
                    stock_config=config,
                    timestamp=django_timezone.now(),
                    signal_generated=signal,
                    quantity=config.trade_quantity,
                    price_at_execution=current_price if signal != 'Hold' else None,
                    trading212_order_id=trading212_order_id,
                    status=trade_status if signal != 'Hold' else 'Simulated',
                    response_text=trade_response_text,
                    realized_pnl_for_trade=realized_pnl,
                )

                config.last_run = django_timezone.now()
                config.save()

            except Exception as e:
                logger.exception(f"Unexpected error processing {config.yf_ticker}: {e}")
                config.last_run = django_timezone.now()
                config.save()

    logger.info(f"[{django_timezone.now()}] Automated trading run finished.")