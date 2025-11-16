from django.shortcuts import render
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from .models import PortfolioHolding


# Load environment variables from .env file
load_dotenv()

def portfolio_view(request):
    # Retrieve API key and headers from environment variables
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    TRADING_API_KEY = os.getenv('TRADING_API_KEY')
    TRADING_API_SECRET = os.getenv('TRADING_API_SECRET')
    

    # URLs
    url_portfolio = "https://demo.trading212.com/api/v0/equity/portfolio"
    url_instro = "https://demo.trading212.com/api/v0/equity/metadata/instruments"

    # Fetch and save data if refresh is requested
    if request.method == 'GET' or not PortfolioHolding.objects.exists():
        try:
            # Fetch portfolio data - Use 'headers' instead of 'auth' for Basic authentication with Base64 encoded credentials
            response_portfolio = requests.get(url_portfolio, auth=(TRADING_API_KEY,TRADING_API_SECRET))
            response_portfolio.raise_for_status()  # Raise an exception for bad status codes
            data_portfolio = response_portfolio.json()
            df_portfolio = pd.DataFrame(data_portfolio)

            # Fetch instruments data - This also needs the headers if it's part of the same authenticated API
            response_instro = requests.get(url_instro, auth=(TRADING_API_KEY,TRADING_API_SECRET))
            response_instro.raise_for_status()
            data_instro = response_instro.json()
            df_instro = pd.DataFrame(data_instro)
        except requests.exceptions.RequestException as e:
            # Handle connection errors, timeouts, etc.
            context = {'error_message': f"Error fetching data from API: {e}"}
            return render(request, 'portfolio/portfolio_detail.html', context)
        except ValueError:
            # Handle JSON decoding errors
            context = {'error_message': "Error decoding API response. The API key may be invalid or the service may be down."}
            return render(request, 'portfolio/portfolio_detail.html', context)

        # Process portfolio data
        df_portfolio['value'] = df_portfolio['quantity'] * df_portfolio['currentPrice']
        df_portfolio['initialFillDate'] = pd.to_datetime(df_portfolio['initialFillDate'], errors='coerce')
        df_portfolio['year-month'] = [date.strftime('%Y-%m') if pd.notna(date) else None for date in df_portfolio['initialFillDate']]
        df_portfolio.rename(columns={'ppl': 'profit', 'fxPpl': 'fx profit'}, inplace=True)
        df_portfolio.drop(columns=['initialFillDate', 'frontend'], inplace=True)

        # Merge dataframes
        df_merged = pd.merge(df_portfolio, df_instro, on='ticker', how='left')
        df_merged.drop(columns=['workingScheduleId','isin','pieQuantity','maxBuy','maxSell','maxOpenQuantity','addedOn'], inplace=True)

        # Convert GBX to GBP
        gbx_rows = df_merged['currencyCode'] == 'GBX'
        df_merged.loc[gbx_rows, ['averagePrice', 'currentPrice', 'value']] *= 0.01
        df_merged.loc[gbx_rows, 'currencyCode'] = 'GBP'

        # Fetch exchange rates
        currency_list = df_merged['currencyCode'].unique()
        main_currency = "EUR"
        other_currencies = [currency for currency in currency_list if currency != main_currency and currency is not None]

        exchange_rate_data = []
        for currency in other_currencies:
            url_ex = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={currency}&to_currency={main_currency}&apikey={api_key}'
            r_ex = requests.get(url_ex)
            data_ex = r_ex.json()
            if "Realtime Currency Exchange Rate" in data_ex:
                rate_info = data_ex["Realtime Currency Exchange Rate"]
                exchange_rate_data.append({
                    'From_Currency Code': rate_info.get('1. From_Currency Code'),
                    'Exchange Rate': rate_info.get('5. Exchange Rate'),
                })

        exchange_rates_df = pd.DataFrame(exchange_rate_data)
        
        if not exchange_rates_df.empty:
            exchange_rate_dict = exchange_rates_df.set_index('From_Currency Code')['Exchange Rate'].astype(float).to_dict()
            
            columns_to_convert = ['averagePrice', 'currentPrice', 'value']
            for col in columns_to_convert:
                df_merged[f'{col}_eur'] = df_merged.apply(
                    lambda row: row[col] * exchange_rate_dict.get(row['currencyCode'], 1.0) 
                    if row['currencyCode'] != main_currency else row[col], 
                    axis=1
                )

        df_merged['fx profit'] = df_merged['fx profit'].fillna(0)

        # Clear old data and save new data
        PortfolioHolding.objects.all().delete()
        for _, row in df_merged.iterrows():
            PortfolioHolding.objects.create(
                ticker=row.get('ticker'),
                name=row.get('name'),
                quantity=row.get('quantity'),
                average_price=row.get('averagePrice'),
                current_price=row.get('currentPrice'),
                value=row.get('value'),
                profit=row.get('profit'),
                fx_profit=row.get('fx profit'),
                year_month=row.get('year-month'),
                currency_code=row.get('currencyCode'),
                instrument_type=row.get('type'),
                average_price_eur=row.get('averagePrice_eur'),
                current_price_eur=row.get('currentPrice_eur'),
                value_eur=row.get('value_eur'),
            )

    # Fetch all holdings from the database
    holdings = PortfolioHolding.objects.all()

    context = {
        'holdings': holdings,
    }
    return render(request, 'portfolio/portfolio_detail.html', context)