# trading_core/actions.py
import os
import requests
import base64
from decimal import Decimal
from django.utils import timezone
from dotenv import load_dotenv
from .models import Trade, ConfiguredStock

# Load environment variables
load_dotenv()
TRADING_API_KEY = os.getenv('TRADING_API_KEY')
TRADING_API_SECRET = os.getenv('TRADING_API_SECRET')
TRADING_212_ORDERS_URL = "https://demo.trading212.com/api/v0/equity/orders/market"

def place_market_order(config_id, order_type, quantity, trade_type='manual'):
    """
    Places a market order (buy or sell) for a given stock configuration.
    
    Args:
        config_id (int): The ID of the ConfiguredStock.
        order_type (str): 'buy' or 'sell'.
        quantity (Decimal): The quantity to trade.
        trade_type (str): 'manual' or 'auto', indicating how the trade was initiated.

    Returns:
        tuple: (success (bool), message (str))
    """
    is_simulation_mode = not (TRADING_API_KEY and TRADING_API_SECRET)
    headers = {}
    if not is_simulation_mode:
        credentials_string = f"{TRADING_API_KEY}:{TRADING_API_SECRET}"
        encoded_credentials = base64.b64encode(credentials_string.encode('utf-8')).decode('utf-8')
        headers["Authorization"] = f"Basic {encoded_credentials}"

    try:
        config = ConfiguredStock.objects.get(id=config_id)
    except ConfiguredStock.DoesNotExist:
        return False, "Configuration not found."

    # For sells, quantity should be negative in the payload
    payload_quantity = quantity if order_type == 'buy' else -quantity

    payload = {
        "ticker": config.trading212_ticker,
        "quantity": payload_quantity,
    }

    trade_status = 'Simulated'
    trade_response_text = f"Simulated {order_type.upper()} order for {quantity} shares of {config.trading212_ticker}."
    trading212_order_id = None

    if not is_simulation_mode:
        try:
            response = requests.post(TRADING_212_ORDERS_URL, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            response_json = response.json()
            trading212_order_id = response_json.get('orderId')
            trade_status = 'Success'
            trade_response_text = response.text
        except requests.exceptions.RequestException as e:
            response_text = "No response body."
            if e.response is not None:
                response_text = e.response.text
            error_message = f"API Request Error: {e} with payload: {payload}. Response: {response_text}"
            trade_response_text = error_message
            trade_status = 'Failed'
            
            # Log the failed trade attempt
            Trade.objects.create(
                stock_config=config,
                signal_generated=f'{trade_type.capitalize()} {order_type.upper()}',
                quantity=quantity,
                status=trade_status,
                response_text=trade_response_text,
            )
            return False, error_message
    
    # Log the successful or simulated trade
    Trade.objects.create(
        stock_config=config,
        signal_generated=f'{trade_type.capitalize()} {order_type.upper()}',
        quantity=quantity,
        trading212_order_id=trading212_order_id,
        status=trade_status,
        response_text=trade_response_text,
    )

    return True, f"Successfully placed {order_type.upper()} order for {config.yf_ticker}."
