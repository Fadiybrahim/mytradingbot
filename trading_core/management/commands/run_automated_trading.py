# trading_core/management/commands/run_automated_trading.py
import os
import yfinance as yf
import pandas as pd
import requests
from django.core.management.base import BaseCommand
from django.utils import timezone
from django.db import transaction
from decimal import Decimal # For precise monetary calculations
from dotenv import load_dotenv
from trading_core.models import ConfiguredStock, Trade
from bot_app.views import calculate_rsi # Re-use the RSI calculation function
from trading_core.actions import place_market_order # Import the centralized trading action
from datetime import datetime

# Load environment variables from .env file
load_dotenv()

TRADING_API_KEY = os.getenv('TRADING_API_KEY')
# TRADING_212_ORDERS_URL is no longer directly used here as place_market_order handles it

class Command(BaseCommand):
    help = 'Runs the automated trading bot for all active configured stocks.'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS(f"[{timezone.now()}] Starting automated trading run..."))

        headers = {"Authorization": TRADING_API_KEY} if TRADING_API_KEY else {}
        is_simulation_mode = not TRADING_API_KEY # If no API key, it's simulation mode

        # Keep track of open positions for P&L calculation (in-memory for this run)
        # In a real system, this would be persistent (e.g., in a DB model or external state)
        # For simplicity, we'll assume a basic P&L where each sell is intended to close a buy.
        # This is a very simplified model and needs expansion for multiple open positions.
        open_positions = {} # {yf_ticker: {'quantity': N, 'avg_buy_price': X}}

        # First, load any existing "open" positions from previous runs (if you had a more advanced Trade model)
        # For this example, we'll re-calculate on the fly based on existing trades.
        # A proper PnL system would track individual buy orders to match with sell orders.

        active_configs = ConfiguredStock.objects.filter(is_active_for_trading=True)

        if not active_configs:
            self.stdout.write(self.style.WARNING("No active configured stocks found for automated trading."))
            return

        for config in active_configs:
            self.stdout.write(self.style.HTTP_INFO(f"Processing {config.yf_ticker} ({config.trading212_ticker})..."))

            with transaction.atomic(): # Ensure all DB changes for this stock are atomic
                try:
                    # 1. Fetch Data
                    stock_info = yf.Ticker(config.yf_ticker)
                    hist = stock_info.history(period="5d", interval="5m")

                    # --- Timezone Normalization for historical data ---
                    if not hist.empty:
                        if hist.index.tz is None:
                            hist.index = hist.index.tz_localize('UTC')
                        else:
                            hist.index = hist.index.tz_convert('UTC')
                    # --- End Timezone Normalization ---

                    realtime_info = stock_info.info
                    realtime_price = realtime_info.get('regularMarketPrice')

                    if realtime_price is None:
                        self.stdout.write(self.style.ERROR(f"Could not fetch real-time price for {config.yf_ticker}. Skipping trade for this stock."))
                        config.last_run = timezone.now()
                        config.save()
                        continue
                    
                    # Convert realtime_price to Decimal for consistent calculations
                    current_price = Decimal(str(realtime_price))

                    # Append real-time data
                    realtime_data = {
                        'Open': realtime_price, 'High': realtime_price, 'Low': realtime_price,
                        'Close': realtime_price, 'Volume': 0, 'Dividends': 0, 'Stock Splits': 0
                    }
                    realtime_timestamp = pd.to_datetime(datetime.now(timezone.utc))
                    realtime_df = pd.DataFrame([realtime_data], index=[realtime_timestamp])
                    realtime_df.index.name = 'Date'

                    if not hist.empty:
                        hist = pd.concat([hist, realtime_df])
                    else:
                        hist = realtime_df
                    
                    if hist.empty:
                        self.stdout.write(self.style.ERROR(f"No valid historical or real-time data for {config.yf_ticker}. Skipping trade."))
                        config.last_run = timezone.now()
                        config.save()
                        continue

                    # 2. Calculate RSI
                    if 'Close' in hist.columns and len(hist) > config.rsi_window: # Ensure enough data for RSI
                        hist['RSI'] = calculate_rsi(hist['Close'], config.rsi_window)
                        latest_rsi = hist['RSI'].iloc[-1]
                        if pd.isna(latest_rsi):
                             self.stdout.write(self.style.WARNING(f"RSI calculated as NaN for {config.yf_ticker}. Skipping trade."))
                             config.last_run = timezone.now()
                             config.save()
                             continue
                    else:
                        self.stdout.write(self.style.WARNING(f"Not enough data ({len(hist)} rows) to calculate RSI for {config.yf_ticker} (window: {config.rsi_window}). Skipping trade."))
                        config.last_run = timezone.now()
                        config.save()
                        continue

                    # 3. Generate Signal
                    signal = 'Hold'
                    if latest_rsi < config.buy_threshold:
                        signal = 'Buy'
                    elif latest_rsi > config.sell_threshold:
                        signal = 'Sell'

                    self.stdout.write(self.style.NOTICE(f"Latest RSI for {config.yf_ticker}: {latest_rsi:.2f}, Signal: {signal}"))

                    trade_response_text = "N/A"
                    trade_status = "Simulated" # Default for logging
                    trading212_order_id = None
                    realized_pnl = None # PnL for this specific trade

                    # Determine current position to refine trade logic (simplified)
                    # This assumes you only have ONE open position for a given stock, or net position.
                    # A more complex system would track individual buy orders.
                    net_position_quantity = 0
                    buy_trades = Trade.objects.filter(stock_config=config, signal_generated='Buy', status='Success', realized_pnl_for_trade__isnull=True)
                    sell_trades = Trade.objects.filter(stock_config=config, signal_generated='Sell', status='Success', realized_pnl_for_trade__isnull=True)
                    
                    net_position_quantity = sum(t.quantity for t in buy_trades) - sum(t.quantity for t in sell_trades)
                    
                    # Average buy price for open position
                    if net_position_quantity > 0:
                        total_buy_value = sum(t.quantity * t.price_at_execution for t in buy_trades)
                        avg_buy_price = total_buy_value / net_position_quantity if net_position_quantity > 0 else Decimal('0.00')
                    else:
                        avg_buy_price = Decimal('0.00')


                    # 4. Execute Trade if signal is Buy/Sell
                    if signal == 'Buy':
                        # Check if current holdings are more than 0
                        if net_position_quantity > 0:
                            trade_response_text = f"Skipped AUTO BUY for {config.yf_ticker}: Already holding {net_position_quantity} shares."
                            signal = 'Hold' # Change signal to Hold as no buy action is taken
                            trade_status = 'Skipped'
                            self.stdout.write(self.style.WARNING(trade_response_text))
                        else:
                            # Only buy if current holdings are 0
                            success, message = place_market_order(config.id, 'buy', config.trade_quantity, trade_type='auto')
                            if success:
                                self.stdout.write(self.style.SUCCESS(f"Successfully placed AUTO BUY order for {config.yf_ticker}. Message: {message}"))
                            else:
                                self.stdout.write(self.style.ERROR(f"Failed to place AUTO BUY order for {config.yf_ticker}: {message}"))
                            trade_response_text = message
                            trade_status = 'Success' if success else 'Failed'

                    elif signal == 'Sell':
                        if net_position_quantity >= config.trade_quantity:
                            success, message = place_market_order(config.id, 'sell', config.trade_quantity, trade_type='auto')
                            if success:
                                self.stdout.write(self.style.SUCCESS(f"Successfully placed AUTO SELL order for {config.yf_ticker}. Message: {message}"))
                                if avg_buy_price > 0:
                                    realized_pnl = (current_price - avg_buy_price) * config.trade_quantity
                                    config.total_realized_pnl += realized_pnl
                                    config.save()
                                    self.stdout.write(self.style.NOTICE(f"Realized P&L from sell: {realized_pnl:.2f}"))
                            else:
                                self.stdout.write(self.style.ERROR(f"Failed to place AUTO SELL order for {config.yf_ticker}: {message}"))
                            trade_response_text = message
                            trade_status = 'Success' if success else 'Failed'
                        else:
                            trade_response_text = f"Skipped AUTO SELL for {config.yf_ticker}: Not enough shares to sell. Currently holding {net_position_quantity} shares."
                            signal = 'Hold'
                            trade_status = 'Skipped'
                            self.stdout.write(self.style.WARNING(trade_response_text))
                    
                    # 5. Log Trade (even if it was a 'Hold' or skipped)
                    # The place_market_order function already logs the trade, so we only need to log if it was a 'Hold' or 'Skipped' signal
                    if signal == 'Hold' or trade_status == 'Skipped':
                        Trade.objects.create(
                            stock_config=config,
                            signal_generated=signal,
                            quantity=config.trade_quantity,
                            price_at_execution=current_price if signal != 'Hold' else None,
                            status=trade_status,
                            response_text=trade_response_text,
                            realized_pnl_for_trade=realized_pnl,
                        )
                    # If a trade was placed (Buy/Sell), place_market_order already logged it.

                    # Update last run time for the config
                    config.last_run = timezone.now()
                    config.save()

                except Exception as e:
                    self.stdout.write(self.style.ERROR(f"Unexpected error processing {config.yf_ticker}: {e}"))
                    # Still update last_run even if there was an error
                    config.last_run = timezone.now()
                    config.save()

        self.stdout.write(self.style.SUCCESS(f"[{timezone.now()}] Automated trading run finished."))
