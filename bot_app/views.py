import os
import io
import base64
import yfinance as yf
import pandas as pd
import mplfinance as mpf

# --- Matplotlib backend for non-GUI environments ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --- Plotly imports ---
import plotly.graph_objects as go
import json
import numpy as np

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        return super(NpEncoder, self).default(obj)
# --- END Plotly imports ---

import requests
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.conf import settings
from dotenv import load_dotenv
from trading_core.models import ConfiguredStock, Trade, HistoricalData, Trading212Instrument # Import Trading212Instrument
from portfolio.models import PortfolioHolding
from django.urls import reverse
from django.db import transaction
from django.views.decorators.http import require_POST
from django.contrib import messages
from trading_core.actions import place_market_order
import logging


from django.utils import timezone as django_timezone 
from datetime import datetime, timezone as python_timezone 



logger = logging.getLogger(__name__)

load_dotenv()

TRADING_API_KEY = os.getenv('TRADING_API_KEY')
TRADING_212_METADATA_URL = "https://demo.trading212.com/api/v0/equity/metadata/instruments"


def calculate_rsi(data, window):
    diff = data.diff(1).dropna()
    up, down = diff.copy(), diff.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    avg_gain = up.ewm(com=window - 1, adjust=False).mean()
    avg_loss = down.abs().ewm(com=window - 1, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def get_trading212_instruments():
    # This function is no longer used for searching in stock_search_view,
    # but might be used elsewhere. Keeping it for now.
    if not TRADING_API_KEY:
        logger.warning("TRADING_API_KEY not set. Cannot fetch Trading 212 instruments.")
        return []
        
    headers = {"Authorization": TRADING_API_KEY}
    try:
        response = requests.get(TRADING_212_METADATA_URL, headers=headers, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Trading 212 instruments: {e}")
        return []


def main_view(request):
    return render(request, 'bot_app/main.html')

def stock_search_view(request):
    search_results = {'yf': [], 't212': []}
    query = ""
    error_message = ""

    if request.method == 'POST':
        query = request.POST.get('query', '').strip()
        if query:
            # Yahoo Finance Search
            yf_search_results_raw = yf.Search(query, max_results=10).quotes
            search_results['yf'] = [
                {'symbol': q['symbol'], 'longname': q.get('longname', 'N/A')}
                for q in yf_search_results_raw if 'symbol' in q
            ]

            # Trading 212 Instrument Search from DB
            # Search by shortName and ticker, case-insensitive
            t212_instruments_db = Trading212Instrument.objects.filter(
                shortName__icontains=query
            ) | Trading212Instrument.objects.filter(
                ticker__icontains=query
            )
            
            search_results['t212'] = [
                {'ticker': inst.ticker, 'shortName': inst.shortName, 'name': inst.name}
                for inst in t212_instruments_db
            ]
        else:
            error_message = "Please enter a stock symbol or name to search."
    
    configured_stocks = ConfiguredStock.objects.all().order_by('yf_ticker')

    context = {
        'search_results': search_results,
        'query': query,
        'error_message': error_message,
        'configured_stocks': configured_stocks,
        'has_api_key': TRADING_API_KEY is not None
    }
    return render(request, 'bot_app/stock_search.html', context)

def update_historical_data_gaps(stock_config):
    """
    Fetches historical data from yfinance and appends new, non-duplicate 5-minute interval data
    to fill gaps in the HistoricalData model for a given stock_config.
    """
    logger.info(f"Starting historical data gap fill for {stock_config.yf_ticker}...")
    
    latest_db_record = HistoricalData.objects.filter(stock_config=stock_config).order_by('-timestamp').first()
    
    if latest_db_record:
        # Fetch data from the day before the latest record to ensure overlap and recalculate RSI correctly
        start_date = latest_db_record.timestamp - pd.Timedelta(days=1)
        # Ensure start_date is timezone-aware if latest_db_record.timestamp is
        if start_date.tzinfo is None:
            start_date = python_timezone.localize(start_date)
        else:
            start_date = start_date.astimezone(python_timezone.utc)
        
        # Fetch up to current time
        end_date = datetime.now(python_timezone.utc)
        logger.debug(f"Fetching yfinance history from {start_date} to {end_date} for {stock_config.yf_ticker}")
        hist = yf.Ticker(stock_config.yf_ticker).history(start=start_date, end=end_date, interval="5m")
    else:
        # If no data exists, fetch a longer period (e.g., 60 days)
        logger.debug(f"No existing historical data. Fetching 60 days history for {stock_config.yf_ticker}")
        hist = yf.Ticker(stock_config.yf_ticker).history(period="60d", interval="5m")

    if hist.empty:
        logger.warning(f"No historical data found from yfinance for {stock_config.yf_ticker} to fill gaps.")
        return

    # Timezone Normalization
    if hist.index.tz is None: hist.index = hist.index.tz_localize('UTC')
    else: hist.index = hist.index.tz_convert('UTC')

    # Filter out data that is already in the database (based on timestamp)
    existing_timestamps = set(HistoricalData.objects.filter(
        stock_config=stock_config,
        timestamp__in=hist.index
    ).values_list('timestamp', flat=True))

    new_data_to_save = []
    for index, row in hist.iterrows():
        if index not in existing_timestamps:
            new_data_to_save.append({
                'timestamp': index,
                'open_price': row['Open'] if pd.notna(row['Open']) else None,
                'high_price': row['High'] if pd.notna(row['High']) else None,
                'low_price': row['Low'] if pd.notna(row['Low']) else None,
                'close_price': row['Close'] if pd.notna(row['Close']) else None,
                'volume': row['Volume'] if pd.notna(row['Volume']) else None,
            })
    
    if not new_data_to_save:
        logger.info(f"No new historical data to append for {stock_config.yf_ticker}.")
        return

    # Convert new data to DataFrame to calculate RSI/Signal
    new_hist_df = pd.DataFrame(new_data_to_save).set_index('timestamp').sort_index()

    # Combine with existing data for RSI calculation
    # Fetch enough existing data to ensure accurate RSI calculation for new points
    all_relevant_data = HistoricalData.objects.filter(
        stock_config=stock_config
    ).order_by('-timestamp')[:stock_config.rsi_window * 2] # Fetch more than needed
    
    existing_df_list = []
    for record in all_relevant_data:
        existing_df_list.append({
            'timestamp': record.timestamp,
            'close_price': record.close_price
        })
    
    if existing_df_list:
        existing_df = pd.DataFrame(existing_df_list).set_index('timestamp').sort_index()
    else:
        existing_df = pd.DataFrame(columns=['close_price'], index=pd.DatetimeIndex([])) # Initialize empty DataFrame with expected column
        existing_df.index.name = 'timestamp' # Set index name for consistency
    
    # Combine existing and new data for RSI calculation
    combined_df = pd.concat([existing_df[['close_price']], new_hist_df[['close_price']]]).drop_duplicates().sort_index()
    
    if 'close_price' in combined_df.columns and len(combined_df) > stock_config.rsi_window:
        combined_df['RSI'] = calculate_rsi(combined_df['close_price'], stock_config.rsi_window)
        combined_df['Signal'] = 'Hold'
        combined_df.loc[combined_df['RSI'] < stock_config.buy_threshold, 'Signal'] = 'Buy'
        combined_df.loc[combined_df['RSI'] > stock_config.sell_threshold, 'Signal'] = 'Sell'
    else:
        combined_df['RSI'] = None
        combined_df['Signal'] = 'N/A'

    # Prepare HistoricalData objects for bulk creation/update
    historical_data_objects = []
    for index, row in new_hist_df.iterrows():
        # Get RSI and Signal from the combined_df
        rsi_val = combined_df.loc[index, 'RSI'] if index in combined_df.index and pd.notna(combined_df.loc[index, 'RSI']) else None
        signal_val = combined_df.loc[index, 'Signal'] if index in combined_df.index and pd.notna(combined_df.loc[index, 'Signal']) else None

        historical_data_objects.append(
            HistoricalData(
                stock_config=stock_config,
                timestamp=index,
                open_price=row['Open'],
                high_price=row['High'],
                low_price=row['Low'],
                close_price=row['Close'],
                volume=row['Volume'],
                rsi=rsi_val,
                signal=signal_val,
            )
        )
    
    HistoricalData.objects.bulk_create(historical_data_objects, ignore_conflicts=True)
    logger.info(f"Successfully appended {len(historical_data_objects)} new historical data points for {stock_config.yf_ticker}.")
    return # Ensure the function returns after processing

def configure_bot_view(request, config_id=None):
    context = {}
    
    # --- Default values for new bots ---
    default_rsi_window = 14
    default_buy_threshold = 35
    default_sell_threshold = 65
    default_trade_quantity = 1
    default_allocated_funds = 0.0
    default_is_active_for_trading = False

    # --- Get parameters ---
    # These are used for initial form population or if no POST data is submitted
    initial_yf_ticker = ""
    initial_t212_ticker = ""
    initial_rsi_window = default_rsi_window
    initial_buy_threshold = default_buy_threshold
    initial_sell_threshold = default_sell_threshold
    initial_trade_quantity = default_trade_quantity
    initial_allocated_funds = default_allocated_funds
    initial_is_active_for_trading = default_is_active_for_trading
    
    stock_config_for_form = None # To hold the config object if editing
    error_messages = []
    success_messages = []
    created_configs = [] # To store successfully created/updated ConfiguredStock objects

    if config_id:
        try:
            stock_config_for_form = get_object_or_404(ConfiguredStock, id=config_id)
            initial_yf_ticker = stock_config_for_form.yf_ticker
            initial_t212_ticker = stock_config_for_form.trading212_ticker
            initial_rsi_window = stock_config_for_form.rsi_window
            initial_buy_threshold = stock_config_for_form.buy_threshold
            initial_sell_threshold = stock_config_for_form.sell_threshold
            initial_trade_quantity = stock_config_for_form.trade_quantity
            initial_is_active_for_trading = stock_config_for_form.is_active_for_trading
            
            # Fetch portfolio holding for allocated funds if T212 ticker exists
            if stock_config_for_form.trading212_ticker:
                try:
                    portfolio_holding = PortfolioHolding.objects.get(ticker=stock_config_for_form.trading212_ticker)
                    initial_allocated_funds = portfolio_holding.allocated_funds
                except PortfolioHolding.DoesNotExist:
                    pass # allocated_funds remains default
                except Exception as e:
                    logger.error(f"Error fetching portfolio holding for {stock_config_for_form.trading212_ticker}: {e}")
        except ConfiguredStock.DoesNotExist:
            error_messages.append(f"Configuration with ID {config_id} not found.")
        except Exception as e:
            logger.exception(f"Error loading configuration {config_id}: {e}")
            error_messages.append(f"Error loading configuration {config_id}: {e}")

    # Get GET parameters for multi-ticker creation
    yf_tickers_list = request.GET.getlist('yf_tickers')
    t212_tickers_list = request.GET.getlist('t212_tickers')

    # Determine current values for form submission (from POST, then GET, then initial)
    current_yf_ticker_post = request.POST.get('yf_ticker', '').strip().upper()
    current_t212_ticker_post = request.POST.get('t212_ticker', '').strip()
    current_rsi_window = int(request.POST.get('rsi_window', request.GET.get('rsi_window', initial_rsi_window)))
    current_buy_threshold = int(request.POST.get('buy_threshold', request.GET.get('buy_threshold', initial_buy_threshold)))
    current_sell_threshold = int(request.POST.get('sell_threshold', request.GET.get('sell_threshold', initial_sell_threshold)))
    current_trade_quantity = int(request.POST.get('trade_quantity', request.GET.get('trade_quantity', initial_trade_quantity)))
    current_allocated_funds = float(request.POST.get('allocated_funds', request.GET.get('allocated_funds', initial_allocated_funds)))
    current_is_active = request.POST.get('is_active_for_trading', request.GET.get('is_active_for_trading')) == 'on'

    # --- Process POST or GET multi-ticker requests ---
    # Process POST for editing an existing config, or GET for creating new configs (if not editing)
    if (request.method == 'POST' and config_id) or (yf_tickers_list and not config_id):
        
        if config_id: # Editing an existing configuration (POST request)
            if not stock_config_for_form: # If config_id was invalid
                error_messages.append(f"Configuration with ID {config_id} not found for editing.")
            else:
                # Use POST values for editing, fallback to initial values if POST is empty
                edit_yf_ticker = current_yf_ticker_post or initial_yf_ticker
                edit_t212_ticker = current_t212_ticker_post or initial_t212_ticker

            if not edit_yf_ticker or not edit_t212_ticker:
                error_messages.append("Both Yahoo Finance and Trading 212 tickers are required for editing.")
                logger.error(f"Validation error: Missing YF or T212 ticker for editing config ID {config_id}.")
            else:
                try:
                    logger.debug(f"configure_bot_view: Saving changes for config ID {config_id}. YF: {edit_yf_ticker}, T212: {edit_t212_ticker}")
                    with transaction.atomic():
                        stock_config_for_form.yf_ticker = edit_yf_ticker
                        stock_config_for_form.trading212_ticker = edit_t212_ticker
                        stock_config_for_form.rsi_window = current_rsi_window
                        stock_config_for_form.buy_threshold = current_buy_threshold
                        stock_config_for_form.sell_threshold = current_sell_threshold
                        stock_config_for_form.trade_quantity = current_trade_quantity
                        stock_config_for_form.is_active_for_trading = current_is_active
                        stock_config_for_form.save()
                        logger.debug(f"configure_bot_view: ConfiguredStock saved for {edit_yf_ticker}.")
                        
                        # Update or create PortfolioHolding with allocated_funds
                        if stock_config_for_form.trading212_ticker:
                            logger.debug(f"configure_bot_view: Updating/creating PortfolioHolding for T212 ticker: {stock_config_for_form.trading212_ticker}")
                            portfolio_holding, created_holding = PortfolioHolding.objects.get_or_create(
                                ticker=stock_config_for_form.trading212_ticker,
                                defaults={'allocated_funds': current_allocated_funds, 'quantity': 0, 'average_price': 0, 'current_price': 0, 'value': 0, 'profit': 0, 'currency_code': 'USD'}
                            )
                            if not created_holding:
                                portfolio_holding.allocated_funds = current_allocated_funds
                                portfolio_holding.save()
                                logger.debug(f"configure_bot_view: PortfolioHolding updated for {stock_config_for_form.trading212_ticker}.")
                            else:
                                logger.debug(f"configure_bot_view: PortfolioHolding created for {stock_config_for_form.trading212_ticker}.")
                        
                        logger.debug(f"configure_bot_view: Calling update_historical_data_gaps for {stock_config_for_form.yf_ticker}")
                        update_historical_data_gaps(stock_config_for_form)
                        logger.debug(f"configure_bot_view: update_historical_data_gaps completed.")

                        success_messages.append(f"Configuration for {stock_config_for_form.yf_ticker} updated successfully!")
                        created_configs.append(stock_config_for_form) # Add to list for context
                except ValueError as e:
                    error_messages.append(f"Configuration error: {e}. Please correct the inputs.")
                    logger.error(f"ValueError during save for config ID {config_id}: {e}")
                except Exception as e:
                    logger.exception(f"An unexpected error occurred while saving configuration for {config_id}: {e}")
                    error_messages.append(f"An unexpected error occurred while saving configuration: {e}")

        elif yf_tickers_list: # Creating new configurations from GET parameters (not editing)
            if not t212_tickers_list:
                error_messages.append("At least one Trading 212 ticker is required when selecting Yahoo Finance tickers.")
            else:
                for i, yf_ticker in enumerate(yf_tickers_list):
                    yf_ticker = yf_ticker.strip().upper()
                    if not yf_ticker: 
                        logger.warning("Skipping empty YF ticker.")
                        continue

                    # Determine the corresponding T212 ticker
                    t212_ticker = t212_tickers_list[i % len(t212_tickers_list)].strip() if t212_tickers_list else ""

                    if not t212_ticker:
                        error_messages.append(f"Missing Trading 212 ticker for Yahoo Finance ticker '{yf_ticker}'. Skipping.")
                        logger.warning(f"Missing T212 ticker for YF ticker '{yf_ticker}'.")
                        continue

                    try:
                        logger.debug(f"configure_bot_view: Processing YF ticker '{yf_ticker}' and T212 ticker '{t212_ticker}'")
                        logger.debug(f"configure_bot_view: Attempting get_or_create for YF ticker: {yf_ticker}")
                        # Add logging before get_or_create
                        logger.debug(f"configure_bot_view: Before ConfiguredStock.objects.get_or_create for YF: {yf_ticker}")
                        stock_config, created = ConfiguredStock.objects.get_or_create(
                            yf_ticker=yf_ticker,
                            defaults={
                                'trading212_ticker': t212_ticker,
                                'rsi_window': current_rsi_window,
                                'buy_threshold': current_buy_threshold,
                                'sell_threshold': current_sell_threshold,
                                'trade_quantity': current_trade_quantity,
                                'is_active_for_trading': current_is_active,
                            }
                        )
                        # Add logging after get_or_create
                        logger.debug(f"configure_bot_view: After ConfiguredStock.objects.get_or_create for YF: {yf_ticker}, created={created}")

                        if not created:
                            # If already exists, update it with new parameters
                            logger.debug(f"configure_bot_view: ConfiguredStock already exists for {yf_ticker}. Updating.")
                            stock_config.trading212_ticker = t212_ticker
                            stock_config.rsi_window = current_rsi_window
                            stock_config.buy_threshold = current_buy_threshold
                            stock_config.sell_threshold = current_sell_threshold
                            stock_config.trade_quantity = current_trade_quantity
                            stock_config.is_active_for_trading = current_is_active
                            logger.debug(f"configure_bot_view: About to save ConfiguredStock for {yf_ticker}")
                            stock_config.save()
                            logger.info(f"Updated existing ConfiguredStock for {yf_ticker}.")
                        else:
                            logger.info(f"Created new ConfiguredStock for {yf_ticker}.")
                        
                        # Update or create PortfolioHolding with allocated_funds
                        if stock_config.trading212_ticker:
                            logger.debug(f"configure_bot_view: Updating/creating PortfolioHolding for T212 ticker: {stock_config.trading212_ticker}")
                            portfolio_holding, created_holding = PortfolioHolding.objects.get_or_create(
                                ticker=stock_config.trading212_ticker,
                                defaults={'allocated_funds': current_allocated_funds, 'quantity': 0, 'average_price': 0, 'current_price': 0, 'value': 0, 'profit': 0, 'currency_code': 'USD'}
                            )
                            if not created_holding:
                                portfolio_holding.allocated_funds = current_allocated_funds
                                portfolio_holding.save()
                                logger.debug(f"configure_bot_view: PortfolioHolding updated for {stock_config.trading212_ticker}.")
                            else:
                                logger.debug(f"configure_bot_view: PortfolioHolding created for {stock_config.trading212_ticker}.")
                        
                        logger.debug(f"configure_bot_view: Calling update_historical_data_gaps for {stock_config.yf_ticker}")
                        update_historical_data_gaps(stock_config)
                        logger.debug(f"configure_bot_view: update_historical_data_gaps completed.")

                        success_messages.append(f"Successfully configured bot for {yf_ticker}.")
                        created_configs.append(stock_config)
                        
                    except ValueError as e:
                        error_messages.append(f"Configuration error for {yf_ticker}: {e}. Skipping.")
                        logger.error(f"ValueError during save for YF ticker {yf_ticker}: {e}")
                    except Exception as e:
                        logger.exception(f"An unexpected error occurred while saving configuration for {yf_ticker}: {e}")
                        error_messages.append(f"An unexpected error occurred for {yf_ticker}: {e}")
        
            # --- Redirection Logic ---
            logger.debug(f"configure_bot_view: Redirection logic. Errors: {error_messages}, Created configs count: {len(created_configs)}")
            if error_messages:
                # If there were any errors, re-render the form with the errors
                # The form will be rendered with existing values and error messages
                pass # Fall through to rendering the form
            elif created_configs:
                # If new bots were created successfully, redirect to the specific bot's configuration page
                last_created_config_id = created_configs[-1].id
                logger.debug(f"configure_bot_view: Redirecting to configure_bot with config_id={last_created_config_id}")
                return redirect('configure_bot', config_id=last_created_config_id)
            else:
                # If no errors and no configs created (e.g., empty input), fall through to render form
                pass # Fall through to rendering the form

    # --- Rendering Logic ---
    # If we reach here, it means no redirect occurred. Render the form.
    # Populate the form with initial values.
    
    # If editing a specific config, stock_config_for_form is already populated.
    # If creating new bots and it failed, we still want to show the form with submitted values.
    # If no GET tickers and no POST, we show defaults.

    # Use POST values if available, otherwise initial values for form fields
    form_yf_ticker = current_yf_ticker_post or initial_yf_ticker
    form_t212_ticker = current_t212_ticker_post or initial_t212_ticker
    
    # If no config_id was provided and no GET tickers were processed, and no POST data,
    # then we are showing the initial form for manual entry.
    # In this case, form_yf_ticker and form_t212_ticker should be empty strings if no initial values were set.
    if not config_id and not yf_tickers_list and not request.method == 'POST':
        form_yf_ticker = ""
        form_t212_ticker = ""

    context.update({
        'stock_config': stock_config_for_form, # Will be None if not editing or if config_id was invalid
        'yf_ticker': form_yf_ticker,
        't212_ticker': form_t212_ticker,
        'rsi_window': current_rsi_window,
        'buy_threshold': current_buy_threshold,
        'sell_threshold': current_sell_threshold,
        'trade_quantity': current_trade_quantity,
        'allocated_funds': current_allocated_funds,
        'is_active_for_trading': current_is_active,
        'has_api_key': TRADING_API_KEY is not None,
        'error_messages': error_messages,
        'success_messages': success_messages,
    })

    # If editing a specific config, fetch historical data and generate plots for the dashboard view.
    if stock_config_for_form:
        try:
            # Fetch historical data from DB for display and charts
            historical_records = HistoricalData.objects.filter(
                stock_config=stock_config_for_form
            ).order_by('-timestamp')[:30]
            logger.debug(f"configure_bot_view: Fetched {len(historical_records)} historical records from DB for {stock_config_for_form.yf_ticker}.")

            hist_data_list = []
            if historical_records.exists():
                for record in historical_records:
                    hist_data_list.append({
                        'Date': record.timestamp,
                        'Open': record.open_price,
                        'High': record.high_price,
                        'Low': record.low_price,
                        'Close': record.close_price,
                        'Volume': record.volume,
                        'RSI': record.rsi,
                        'Signal': record.signal,
                    })
                hist = pd.DataFrame(hist_data_list).set_index('Date').sort_index()
                context['rsi_calculated'] = hist['RSI'].notna().any()
                context['latest_signal'] = hist['Signal'].iloc[-1] if not hist.empty else 'N/A'
                context['realtime_price'] = hist['Close'].iloc[-1] if not hist.empty else 'N/A'
            else:
                hist = pd.DataFrame()
                context['rsi_calculated'] = False
                context['latest_signal'] = 'N/A (No historical data)'
                context['realtime_price'] = 'N/A'

            if not hist.empty:
                price_rsi_table = hist[['Open', 'High', 'Low', 'Close', 'RSI', 'Signal']].copy().sort_index(ascending=False)
                context['price_rsi_table'] = price_rsi_table.to_html(classes='table table-striped table-sm')
            else:
                context['price_rsi_table'] = "<p class='text-muted'>No data for Price, RSI, and Signals table.</p>"

            # Plotly Chart Generation
            candlestick_plot_json = None
            rsi_plot_json = None

            if not hist.empty and not hist[['Open', 'High', 'Low', 'Close']].isnull().all().all():
                fig_candle = go.Figure(data=[go.Candlestick(
                    x=hist.index, open=hist['Open'], high=hist['High'], low=hist['Low'], close=hist['Close']
                )])
                fig_candle.update_layout(title=f'{stock_config_for_form.yf_ticker} 5-Minute Candles', xaxis_rangeslider_visible=False, template="plotly_dark", height=400)
                candlestick_plot_json = json.dumps(fig_candle.to_dict(), cls=NpEncoder)
                
            if context.get('rsi_calculated') and hist['RSI'].notna().any():
                fig_rsi = go.Figure(data=[go.Scatter(x=hist.index, y=hist['RSI'], mode='lines', name='RSI')])
                fig_rsi.add_hline(y=stock_config_for_form.buy_threshold, line_dash="dash", line_color="green", annotation_text=f"Buy ({stock_config_for_form.buy_threshold})")
                fig_rsi.add_hline(y=stock_config_for_form.sell_threshold, line_dash="dash", line_color="red", annotation_text=f"Sell ({stock_config_for_form.sell_threshold})")
                fig_rsi.update_layout(title=f'{stock_config_for_form.yf_ticker} 5-Minute RSI', template="plotly_dark", height=300)
                rsi_plot_json = json.dumps(fig_rsi.to_dict(), cls=NpEncoder)

            context['candlestick_plot_json'] = candlestick_plot_json
            context['rsi_plot_json'] = rsi_plot_json

            # News
            news = yf.Search(stock_config_for_form.yf_ticker, news_count=10).news
            context['news'] = news

            # Trades
            context['trades'] = stock_config_for_form.trades.filter(signal_generated__in=['Buy', 'Sell']).order_by('-timestamp')[:10]
        
        except Exception as e:
            logger.exception(f"An error occurred during analysis for {stock_config_for_form.yf_ticker}: {e}")
            error_messages.append(f"An error occurred during analysis for {stock_config_for_form.yf_ticker}: {e}")

    # If no specific config is loaded and no errors, show the form for manual entry.
    # The 'dashboard.html' template should handle displaying the form when 'stock_config' is None.
    return render(request, 'bot_app/dashboard.html', context)


def get_bot_status_json(request, config_id):
    """
    Returns the latest status updates for a specific ConfiguredStock as JSON.
    Used by AJAX calls from the dashboard.
    """
    stock_config = get_object_or_404(ConfiguredStock, id=config_id)

    realtime_price = None
    latest_signal = "N/A"
    price_rsi_table_html = "<p class='text-muted'>No data for Price, RSI, and Signals table.</p>"

    try:
        yf_ticker = stock_config.yf_ticker
        
        # --- Real-time Data Fetch and Append for AJAX ---
        stock_info = yf.Ticker(yf_ticker)
        realtime_info = stock_info.info
        realtime_price = realtime_info.get('regularMarketPrice')

        if realtime_price is None:
            logger.warning(f"Could not fetch real-time price for {yf_ticker} during AJAX update.")
            # Fallback to latest historical data if real-time price is unavailable
            latest_historical = HistoricalData.objects.filter(stock_config=stock_config).order_by('-timestamp').first()
            if latest_historical:
                realtime_price = latest_historical.close_price
            else:
                return JsonResponse({
                    'realtime_price': 'N/A',
                    'latest_signal': 'N/A',
                    'last_run': stock_config.last_run.strftime("%Y-%m-%d %H:%M:%S") if stock_config.last_run else 'Never',
                    'total_realized_pnl': str(stock_config.total_realized_pnl),
                    'is_active_for_trading': stock_config.is_active_for_trading,
                    'trades': [],
                    'price_rsi_table_html': price_rsi_table_html,
                })

        # Get enough historical data from DB to calculate RSI for the new point
        # Need at least rsi_window + 1 points for RSI calculation
        required_history_count = stock_config.rsi_window + 5 # A bit extra for robustness
        recent_historical_records = HistoricalData.objects.filter(
            stock_config=stock_config
        ).order_by('-timestamp')[:required_history_count]

        hist_data_list = []
        for record in reversed(recent_historical_records): # Reverse to get ascending order for DataFrame
            hist_data_list.append({
                'Date': record.timestamp,
                'Open': record.open_price,
                'High': record.high_price,
                'Low': record.low_price,
                'Close': record.close_price,
                'Volume': record.volume,
            })
        
        # Initialize hist DataFrame with proper columns, even if empty
        if hist_data_list:
            hist = pd.DataFrame(hist_data_list).set_index('Date')
            logger.debug(f"get_bot_status_json: Initial hist DataFrame from DB. Head:\n{hist.head()}")
        else:
            hist = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'], index=pd.DatetimeIndex([]))
            hist.index.name = 'Date'
            logger.debug("get_bot_status_json: Initial hist DataFrame is empty.")

        # Create a new DataFrame for the real-time data point
        realtime_timestamp = pd.to_datetime(datetime.now(python_timezone.utc))
        
        # Check if a record for this exact timestamp already exists to prevent duplicates
        # If within a small delta (e.g., 1 minute), update the last record
        if not hist.empty and (realtime_timestamp - hist.index[-1]).total_seconds() < 60:
            logger.debug(f"get_bot_status_json: Updating existing record for {hist.index[-1]} with new realtime price {realtime_price}.")
            latest_record_db = HistoricalData.objects.filter(stock_config=stock_config).order_by('-timestamp').first()
            if latest_record_db:
                latest_record_db.close_price = realtime_price
                latest_record_db.open_price = latest_record_db.open_price or realtime_price
                latest_record_db.high_price = max(latest_record_db.high_price or realtime_price, realtime_price)
                latest_record_db.low_price = min(latest_record_db.low_price or realtime_price, realtime_price)
                latest_record_db.save()
                # Update the DataFrame too
                hist.loc[hist.index[-1], 'Close'] = realtime_price
                hist.loc[hist.index[-1], 'Open'] = hist.loc[hist.index[-1], 'Open'] or realtime_price
                hist.loc[hist.index[-1], 'High'] = max(hist.loc[hist.index[-1], 'High'] or realtime_price, realtime_price)
                hist.loc[hist.index[-1], 'Low'] = min(hist.loc[hist.index[-1], 'Low'] or realtime_price, realtime_price)
        else:
            logger.debug(f"get_bot_status_json: Adding new realtime record for {realtime_timestamp} with price {realtime_price}.")
            # Add new real-time data point
            realtime_df = pd.DataFrame([{
                'Open': realtime_price, 'High': realtime_price, 'Low': realtime_price,
                'Close': realtime_price, 'Volume': 0
            }], index=[realtime_timestamp])
            realtime_df.index.name = 'Date'
            
            # Concatenate, drop duplicates, and sort
            hist = pd.concat([hist, realtime_df]).drop_duplicates(keep='last').sort_index()

            # Save the new real-time data point to DB
            HistoricalData.objects.create(
                stock_config=stock_config,
                timestamp=realtime_timestamp,
                open_price=realtime_price,
                high_price=realtime_price,
                low_price=realtime_price,
                close_price=realtime_price,
                volume=0,
                rsi=None, # Will be calculated below
                signal=None # Will be calculated below
            )
        
        if hist.empty: 
            logger.error(f"get_bot_status_json: hist DataFrame is empty after attempting to add real-time data for {yf_ticker}.")
            raise ValueError(f"No historical or real-time data found for {yf_ticker}.")

        logger.debug(f"get_bot_status_json: hist DataFrame after real-time update. Tail:\n{hist.tail()}")

        # Recalculate RSI and Signal for the updated historical data
        if 'Close' in hist.columns and len(hist) > stock_config.rsi_window:
            hist['RSI'] = calculate_rsi(hist['Close'], stock_config.rsi_window)
            if hist['RSI'].notna().any():
                hist['Signal'] = 'Hold'
                hist.loc[hist['RSI'] < stock_config.buy_threshold, 'Signal'] = 'Buy'
                hist.loc[hist['RSI'] > stock_config.sell_threshold, 'Signal'] = 'Sell'
                latest_signal = hist['Signal'].iloc[-1]
                logger.debug(f"get_bot_status_json: RSI calculated. Latest RSI: {hist['RSI'].iloc[-1]}, Signal: {latest_signal}")
            else:
                latest_signal = 'N/A (RSI NaN or insufficient data after calc.)'
                logger.debug("get_bot_status_json: RSI not calculated (NaN or insufficient data).")
        else:
            latest_signal = 'N/A (Not enough data for RSI or No Close data)'
            logger.debug("get_bot_status_json: Not enough data for RSI calculation.")

        # Update the latest HistoricalData object with calculated RSI and Signal
        latest_record_db = HistoricalData.objects.filter(stock_config=stock_config).order_by('-timestamp').first()
        if latest_record_db and not hist.empty:
            latest_hist_row = hist.iloc[-1]
            latest_record_db.rsi = latest_hist_row['RSI'] if 'RSI' in latest_hist_row and pd.notna(latest_hist_row['RSI']) else None
            latest_record_db.signal = latest_hist_row['Signal'] if 'Signal' in latest_hist_row and pd.notna(latest_hist_row['Signal']) else None
            latest_record_db.save()
            logger.debug(f"get_bot_status_json: Updated latest HistoricalData record in DB with RSI: {latest_record_db.rsi}, Signal: {latest_record_db.signal}")
        else:
            logger.debug("get_bot_status_json: Could not update latest HistoricalData record in DB.")

        # Retrieve the last 30 entries from DB for the table display
        historical_records_for_display = HistoricalData.objects.filter(
            stock_config=stock_config
        ).order_by('-timestamp')[:30]

        if historical_records_for_display.exists():
            display_data_list = []
            for record in historical_records_for_display:
                display_data_list.append({
                    'Date': record.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'Open': record.open_price,
                    'High': record.high_price,
                    'Low': record.low_price,
                    'Close': record.close_price,
                    'RSI': record.rsi,
                    'Signal': record.signal,
                })
            price_rsi_table = pd.DataFrame(display_data_list).to_html(classes='table table-striped table-sm')
            price_rsi_table_html = price_rsi_table
        else:
            price_rsi_table_html = "<p class='text-muted'>No data for Price, RSI, and Signals table.</p>"
        # --- END Real-time Data Fetch and Append ---
        
    except Exception as e:
        logger.error(f"Error fetching real-time data or calculating signal/table for JSON update for {stock_config.yf_ticker}: {e}")

    trades = stock_config.trades.filter(signal_generated__in=['Buy', 'Sell']).order_by('-timestamp')[:5]
    trade_list = []
    for trade in trades:
        trade_list.append({
            'timestamp': trade.timestamp.strftime("%Y-%m-%d %H:%M:%S") if trade.timestamp else None,
            'signal_generated': trade.signal_generated,
            'quantity': trade.quantity,
            'price_at_execution': str(trade.price_at_execution) if trade.price_at_execution else None,
            'status': trade.status,
            'realized_pnl_for_trade': str(trade.realized_pnl_for_trade) if trade.realized_pnl_for_trade else None,
        })

    data = {
        'realtime_price': str(realtime_price) if realtime_price else 'N/A',
        'latest_signal': latest_signal,
        'last_run': stock_config.last_run.strftime("%Y-%m-%d %H:%M:%S") if stock_config.last_run else 'Never',
        'total_realized_pnl': str(stock_config.total_realized_pnl),
        'is_active_for_trading': stock_config.is_active_for_trading,
        'trades': trade_list,
        'price_rsi_table_html': price_rsi_table_html,
    }
    return JsonResponse(data)


def list_configured_bots(request):
    configured_stocks = ConfiguredStock.objects.all().order_by('yf_ticker')
    context = {
        'configured_stocks': configured_stocks,
    }
    return render(request, 'bot_app/configured_bots.html', context)


def toggle_trading_status(request, config_id):
    if request.method == 'POST':
        stock_config = get_object_or_404(ConfiguredStock, id=config_id)
        stock_config.is_active_for_trading = not stock_config.is_active_for_trading
        stock_config.save()
        return redirect('list_configured_bots')
    return redirect('list_configured_bots')


@require_POST
def manual_trade_view(request, config_id, order_type):
    """
    Handles manual buy and sell requests from the dashboard.
    """
    stock_config = get_object_or_404(ConfiguredStock, id=config_id)
    
    try:
        quantity_str = request.POST.get('quantity')
        if not quantity_str:
            raise ValueError("Quantity is required for manual trade.")
        quantity = float(quantity_str)
        if quantity <= 0:
            raise ValueError("Quantity must be a positive number.")
    except (ValueError, TypeError) as e:
        messages.error(request, f"Invalid quantity provided: {e}")
        return redirect('configure_bot', config_id=config_id)

    success, message = place_market_order(config_id, order_type, quantity, trade_type='manual')

    if success:
        messages.success(request, message)
    else:
        messages.error(request, f"Order failed: {message}")

    return redirect('configure_bot', config_id=config_id)
