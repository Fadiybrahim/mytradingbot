# trading_core/actions.py
import os
import requests
import base64
from decimal import Decimal
from django.utils import timezone
from dotenv import load_dotenv
from .models import Trade, ConfiguredStock, PortfolioHolding # Import PortfolioHolding

# Load environment variables
load_dotenv()
TRADING_API_KEY = os.getenv('TRADING_API_KEY')
TRADING_API_SECRET = os.getenv('TRADING_API_SECRET')
TRADING_212_ORDERS_URL = "https://demo.trading212.com/api/v0/equity/orders/market"
TRADING_212_POSITIONS_URL = "https://demo.trading212.com/api/v0/equity/positions" # URL for fetching positions

def get_auth_headers():
    """
    Generates the Authorization header for Trading 212 API requests.
    Uses Basic Authentication with API Key and Secret.
    """
    if not TRADING_API_KEY or not TRADING_API_SECRET:
        # In simulation mode or if credentials are not set
        return {}
    
    credentials_string = f"{TRADING_API_KEY}:{TRADING_API_SECRET}"
    encoded_credentials = base64.b64encode(credentials_string.encode('utf-8')).decode('utf-8')
    return {"Authorization": f"Basic {encoded_credentials}"}

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
    headers = get_auth_headers()

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

def fetch_and_update_portfolio_holdings():
    """
    Fetches current portfolio holdings from Trading212 and updates the database.
    """
    headers = get_auth_headers()
    if not headers:
        print("API credentials not set. Cannot fetch portfolio holdings.")
        return False, "API credentials not set."

    all_holdings_data = []
    configured_stocks = ConfiguredStock.objects.all()

    for stock_config in configured_stocks:
        query = {"ticker": stock_config.trading212_ticker}
        try:
            response = requests.get(TRADING_212_POSITIONS_URL, headers=headers, params=query, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # The response is a list, even if it contains only one position for the ticker
            for position_data in data:
                all_holdings_data.append(position_data)

        except requests.exceptions.RequestException as e:
            print(f"Error fetching data for {stock_config.trading212_ticker}: {e}")
            # Continue to next stock even if one fails
            continue
        except ValueError: # Includes JSONDecodeError
            print(f"Error decoding JSON response for {stock_config.trading212_ticker}.")
            continue

    # Update or create PortfolioHolding records
    updated_count = 0
    created_count = 0
    for holding_data in all_holdings_data:
        instrument = holding_data.get('instrument', {})
        wallet_impact = holding_data.get('walletImpact', {})

        # Use update_or_create for atomic upsert operation
        obj, created = PortfolioHolding.objects.update_or_create(
            instrument_ticker=instrument.get('ticker'),
            defaults={
                'instrument_currency': instrument.get('currency'),
                'instrument_isin': instrument.get('isin'),
                'instrument_name': instrument.get('name'),
                'instrument_short_name': instrument.get('shortName'),
                'average_price_paid': Decimal(holding_data.get('averagePricePaid', 0)),
                'created_at': timezone.make_aware(timezone.datetime.strptime(holding_data.get('createdAt'), '%Y-%m-%dT%H:%M:%SZ')) if holding_data.get('createdAt') else None,
                'current_price': Decimal(holding_data.get('currentPrice', 0)),
                'quantity': Decimal(holding_data.get('quantity', 0)),
                'quantity_available_for_trading': Decimal(holding_data.get('quantityAvailableForTrading', 0)),
                'quantity_in_pies': Decimal(holding_data.get('quantityInPies', 0)),
                'wallet_impact_currency': wallet_impact.get('currency'),
                'wallet_impact_current_value': Decimal(wallet_impact.get('currentValue', 0)),
                'wallet_impact_fx_impact': Decimal(wallet_impact.get('fxImpact', 0)),
                'wallet_impact_total_cost': Decimal(wallet_impact.get('totalCost', 0)),
                'wallet_impact_unrealized_profit_loss': Decimal(wallet_impact.get('unrealizedProfitLoss', 0)),
                'fetched_at': timezone.now(),
            }
        )
        if created:
            created_count += 1
        else:
            updated_count += 1
    
    return True, f"Portfolio holdings updated. Created: {created_count}, Updated: {updated_count}."
