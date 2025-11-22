from django.shortcuts import render
import requests
import pandas as pd
import os
import base64 # Import base64 for encoding
from dotenv import load_dotenv
from .models import PortfolioHolding
from trading_core.models import Trading212Instrument # Import Trading212Instrument
import json # Import json for serializing data
from django.http import JsonResponse # Import JsonResponse
from google import genai
from google.genai import types
import traceback # Import traceback for detailed error logging

# Import plotting libraries
# The data for charts is prepared and passed to the frontend for rendering using JavaScript libraries.
# import matplotlib.pyplot as plt
# import seaborn as sns

# Load environment variables from .env file
load_dotenv()

def portfolio_view(request):
    # Retrieve API keys from environment variables
    alphavantage_api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    trading_api_key = os.getenv('TRADING_API_KEY')
    trading_api_secret = os.getenv('TRADING_API_SECRET') # Load the secret key

    # Initialize the context dictionary to be passed to the template
    context = {}

    # URLs for the Trading 212 API
    url_portfolio = "https://demo.trading212.com/api/v0/equity/portfolio"
 

    # --- Start of Data Fetching and Processing Logic ---
    # This block runs if it's a POST request (from the refresh button) or if it's a GET request and the database is empty.
    if request.method == 'POST' or (request.method == 'GET' and not PortfolioHolding.objects.exists()):
        
        # Check if the necessary API keys are configured
        if not trading_api_key or not trading_api_secret: # Check for both keys
            context['error_message'] = "Trading 212 API Key or Secret is not configured in the .env file."
            # Still render the page but with an error
            return render(request, 'portfolio/portfolio_detail.html', context)

        # Set the correct authorization header for Trading 212 API using Basic Auth
        credentials_string = f"{trading_api_key}:{trading_api_secret}"
        encoded_credentials = base64.b64encode(credentials_string.encode('utf-8')).decode('utf-8')
        headers = {'Authorization': f'Basic {encoded_credentials}'}
        
        try:
            # --- Fetch data from APIs ---
            response_portfolio = requests.get(url_portfolio, headers=headers)
            response_portfolio.raise_for_status()  # Raise an exception for bad status codes (e.g., 401, 500)
            data_portfolio = response_portfolio.json()
            df_portfolio = pd.DataFrame(data_portfolio)

            # Fetch instrument data from the database
            instruments_from_db = Trading212Instrument.objects.all().values()
            df_instruments = pd.DataFrame(list(instruments_from_db))
            # Ensure column names match the expected API response for merging
            df_instruments.rename(columns={
                'name': 'shortName', # The API returns 'shortName', DB has 'name'
                'type': 'type', # The API returns 'type', DB has 'type'
                'currencyCode': 'currencyCode', # The API returns 'currencyCode', DB has 'currencyCode'
                'workingScheduleId': 'workingScheduleId', # The API returns 'workingScheduleId', DB has 'workingScheduleId'
                'isin': 'isin', # The API returns 'isin', DB has 'isin'
                'maxOpenQuantity': 'maxOpenQuantity', # The API returns 'maxOpenQuantity', DB has 'maxOpenQuantity'
                'addedOn': 'addedOn', # The API returns 'addedOn', DB has 'addedOn'
            }, inplace=True)

            # --- Process and merge the data using Pandas ---
            df_portfolio['value'] = df_portfolio['quantity'] * df_portfolio['currentPrice']
            df_portfolio['initialFillDate'] = pd.to_datetime(df_portfolio['initialFillDate'], errors='coerce', utc=True)
            df_portfolio['year-month'] = df_portfolio['initialFillDate'].dt.strftime('%Y-%m').where(pd.notna(df_portfolio['initialFillDate']), None)
            df_portfolio.rename(columns={'ppl': 'profit', 'fxPpl': 'fx_profit'}, inplace=True)
            df_portfolio.drop(columns=['initialFillDate', 'frontend'], inplace=True)

            df_merged = pd.merge(df_portfolio, df_instruments, on='ticker', how='left')
            df_merged.drop(columns=['workingScheduleId', 'isin', 'pieQuantity', 'maxBuy', 'maxSell', 'maxOpenQuantity', 'addedOn'], inplace=True)

            # Convert prices from GBX (pence) to GBP (pounds)
            gbx_rows = df_merged['currencyCode'] == 'GBX'
            df_merged.loc[gbx_rows, ['averagePrice', 'currentPrice', 'value']] /= 100
            df_merged.loc[gbx_rows, 'currencyCode'] = 'GBP'

            # --- Fetch and apply currency exchange rates to convert to EUR ---
            main_currency = "EUR"
            other_currencies = [c for c in df_merged['currencyCode'].unique() if c != main_currency and c is not None]

            exchange_rate_dict = {}
            for currency in other_currencies:
                url_ex = f'https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency={currency}&to_currency={main_currency}&apikey={alphavantage_api_key}'
                r_ex = requests.get(url_ex)
                data_ex = r_ex.json()
                if rate_info := data_ex.get("Realtime Currency Exchange Rate"):
                    exchange_rate_dict[currency] = float(rate_info.get('5. Exchange Rate', 1.0))
            
            # Apply exchange rates to calculate EUR values
            df_merged['value_eur'] = df_merged.apply(
                lambda row: row['value'] * exchange_rate_dict.get(row['currencyCode'], 1.0),
                axis=1
            )
            df_merged['averagePrice_eur'] = df_merged.apply(
                lambda row: row['averagePrice'] * exchange_rate_dict.get(row['currencyCode'], 1.0),
                axis=1
            )
            df_merged['currentPrice_eur'] = df_merged.apply(
                lambda row: row['currentPrice'] * exchange_rate_dict.get(row['currencyCode'], 1.0),
                axis=1
            )
            df_merged['fx_profit'] = df_merged['fx_profit'].fillna(0)



            # --- Update the database ---
            PortfolioHolding.objects.all().delete()  # Clear old data
            for _, row in df_merged.iterrows():
                PortfolioHolding.objects.create(
                    ticker=row.get('ticker'),
                    name=row.get('name'),
                    quantity=row.get('quantity'),
                    average_price=row.get('averagePrice'),
                    current_price=row.get('currentPrice'),
                    value=row.get('value'),
                    profit=row.get('profit'),
                    fx_profit=row.get('fx_profit'),
                    year_month=row.get('year-month'),
                    currency_code=row.get('currencyCode'),
                    instrument_type=row.get('type'),
                    average_price_eur=row.get('averagePrice_eur'),
                    current_price_eur=row.get('currentPrice_eur'),
                    value_eur=row.get('value_eur'),
                )
            # --- Calculate summary statistics ---
            total_investment_eur = df_merged['value_eur'].sum()
            total_profit_eur = df_merged['profit'].sum() 
            total_fx_profit_eur = df_merged['fx_profit'].sum() 
            total_gain_loss_percent = (total_profit_eur / total_investment_eur * 100) if total_investment_eur else 0


            # --- Prepare data for JavaScript chart (Pie Chart) ---
            # This data is suitable for a pie chart showing value distribution by instrument type.
            chart_data_for_js = {}
            if not df_merged.empty:
                value_by_type = df_merged.groupby('type')['value'].sum()
                if not value_by_type.empty:
                    # Convert to a dictionary for JSON serialization
                    chart_data_for_js = value_by_type.to_dict()
            
            # Pass the pie chart data to the context for frontend rendering
            context['portfolio_chart_data'] = json.dumps(chart_data_for_js)            
            
            # --- Prepare data for JavaScript chart (Bar Chart: Profit by Ticker) ---
            # Group by ticker and type, and sum the profit
            profit_by_ticker_type = df_merged.groupby(['shortName', 'type'])['profit'].sum().reset_index()

            # Sort the data by profit in descending order to get the order for the bars
            profit_order = profit_by_ticker_type.groupby('shortName')['profit'].sum().sort_values(ascending=False).index

            # Prepare data for the bar chart
            bar_chart_data = {
                'shortName': profit_by_ticker_type['shortName'].tolist(),
                'type': profit_by_ticker_type['type'].tolist(),
                'profit': profit_by_ticker_type['profit'].tolist(),
                'profit_order': list(profit_order) # Convert index to list
            }
            context['profit_bar_chart_data'] = json.dumps(bar_chart_data)

            # --- Prepare summary statistics for the template ---
            # These details will be displayed alongside the charts.
            context.update({
                'success_message': 'Portfolio data has been successfully updated.',
                'total_investment': total_investment_eur,
                'total_profit': total_profit_eur,
                'total_fx_profit': total_fx_profit_eur,
                'total_gain_loss_percent': total_gain_loss_percent,
            })

        # --- Handle potential errors during the process ---
        except requests.exceptions.RequestException as e:
            context['error_message'] = f"Error fetching data from an API: {e}"
            print(f"ERROR: RequestException in portfolio_view: {e}") # Error print
        except (ValueError, KeyError) as e:
            context['error_message'] = f"Error processing API response. The API key may be invalid or the data format may have changed. Details: {e}"
            print(f"ERROR: ValueError/KeyError in portfolio_view: {e}") # Error print
        except Exception as e:
            context['error_message'] = f"An unexpected error occurred: {e}"
            print(f"ERROR: Unexpected Exception in portfolio_view: {e}") # Error print
    
    # --- Final step: Fetch all holdings from the database and render the page ---
    # This runs regardless of whether the data was refreshed or not.
    context['holdings'] = PortfolioHolding.objects.all()
    
    return render(request, 'portfolio/portfolio_detail.html', context)

def generate_analysis_view(request):
    print("--- Starting generate_analysis_view ---")
    try:
        # Load GEMINI API Key
        GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
        if not GOOGLE_API_KEY:
            print("ERROR: GEMINI_API_KEY not configured")
            return JsonResponse({'error': 'GEMINI_API_KEY not configured'}, status=500)
        print("GEMINI_API_KEY loaded successfully.")

        # Load Topics from JSON file
        try:
            print("Attempting to load trading_topics.json...")
            with open('trading_topics.json', 'r') as f:
                data = json.load(f)
                Topics = data.get("Topics", {})
            print("trading_topics.json loaded successfully.")
        except FileNotFoundError:
            print("ERROR: trading_topics.json not found")
            return JsonResponse({'error': 'trading_topics.json not found'}, status=500)
        except json.JSONDecodeError:
            print("ERROR: Error decoding trading_topics.json")
            return JsonResponse({'error': 'Error decoding trading_topics.json'}, status=500)

        # Reconstruct df_merged from PortfolioHolding
        print("Reconstructing DataFrame from PortfolioHolding...")
        holdings = PortfolioHolding.objects.all()
        if not holdings.exists():
            print("ERROR: No portfolio holdings found.")
            return JsonResponse({'error': 'No portfolio holdings found. Please refresh portfolio data first.'}, status=400)

        # Convert holdings to a DataFrame
        df_portfolio_holdings = pd.DataFrame(list(holdings.values()))

        # Rename columns to match expected names in the original code if necessary
        df_portfolio_holdings.rename(columns={
            'average_price': 'averagePrice',
            'current_price': 'currentPrice',
            'instrument_type': 'type',
            'currency_code': 'currencyCode',
            'name': 'shortName'
        }, inplace=True)

        # Ensure required columns for prompt are present, fill missing with defaults if necessary
        # The original code was missing 'averagePrice_eur' and 'currentPrice_eur' from this list,
        # leading to a KeyError when trying to select them later.
        required_cols = ['ticker', 'shortName', 'quantity', 'averagePrice', 'currentPrice', 'value', 'profit', 'fx_profit', 'currencyCode', 'type', 'averagePrice_eur', 'currentPrice_eur', 'value_eur']
        print(f"Ensuring required columns are present: {required_cols}")
        for col in required_cols:
            if col not in df_portfolio_holdings.columns:
                # Handle specific columns that might be missing and have default values
                if col == 'shortName':
                    df_portfolio_holdings[col] = df_portfolio_holdings.get('name', 'Unknown')
                elif col == 'currencyCode':
                    df_portfolio_holdings[col] = 'EUR' # Default to EUR if not present
                elif col == 'type':
                    df_portfolio_holdings[col] = 'Unknown' # Default type
                elif col in ['averagePrice_eur', 'currentPrice_eur', 'value_eur', 'profit', 'fx_profit']:
                    df_portfolio_holdings[col] = 0.0 # Default numeric columns to 0.0
                elif col in ['quantity', 'value']:
                    df_portfolio_holdings[col] = 0 # Default quantity and value to 0
                else:
                    df_portfolio_holdings[col] = None # Default other missing columns to None
        print("Required columns check complete.")

        # Ensure numeric columns are correctly typed after ensuring they exist
        # These columns are critical for calculations and the prompt.
        numeric_cols_to_ensure = ['averagePrice', 'currentPrice', 'value', 'profit', 'fx_profit', 'averagePrice_eur', 'currentPrice_eur', 'value_eur']
        for col in numeric_cols_to_ensure:
            if col in df_portfolio_holdings.columns:
                df_portfolio_holdings[col] = pd.to_numeric(df_portfolio_holdings[col], errors='coerce').fillna(0)
        print("Numeric columns ensured.")

        # Create a df_merged-like structure for the prompt, now that all required columns are guaranteed to exist.
        df_merged_for_prompt = df_portfolio_holdings[required_cols].copy()
        
        # Convert DataFrame to JSON string for the prompt.
        # Using to_json with orient='records' to produce a JSON array of records.
        # This is a more standard JSON format than JSON Lines and should be parsable by Gemini.
        portfolio_data_json = df_merged_for_prompt.to_json(orient='records')
        print("DataFrame converted to JSON string for prompt.")

        # Gemini API Configuration
        client = genai.Client(api_key=GOOGLE_API_KEY)
        MODEL_ID = "gemini-2.5-flash"
        tools = [
            types.Tool(googleSearch=types.GoogleSearch(
            )),
        ]
        generate_content_config = types.GenerateContentConfig(
            thinking_config = types.ThinkingConfig(
                thinking_budget=-1,
            ),
            tools=tools,
        )
        print("Gemini API configuration set.")

        # Generate Detailed Portfolio Summary
        print("Generating Detailed Portfolio Summary...")
        # The previous error was due to the f-string interpreting characters within the embedded
        # portifolio_text or the prompt's JSON example as format specifiers.
        # To mitigate this, we will use str.format() and ensure the embedded JSON is properly escaped
        # or handled. Using a placeholder and then formatting is a safer approach.
        prompt_template = """
        You are a stock's financial analyst. Summarize my portfolio data, provided below in JSON format, using the following template:
        Portfolio Data (JSON):
        ```json
        {portfolio_data}
        ```

        Template:
        [Your Name/Portfolio Name] - Portfolio Summary as of [Date]
        1. Executive Summary:
        * Overall Performance: Briefly describe the portfolio's general performance (e.g., "Outperformed benchmark," "Steady growth," "Underperforming expectations").
        * Key Highlights: Mention 1-2 significant points (e.g., "Strong returns from tech sector," "Diversification proving effective," "Exposure to emerging markets").
        * Areas for Review: Note any immediate concerns or areas that might require attention (e.g., "Overweight in a specific sector," "Underperforming asset," "Cash drag").
        2. Portfolio Holdings Overview:
        * Total Market Value:
        * Total Number of Holdings:
        * Top 5 Holdings by Market Value: (List company name and percentage of total portfolio)
        1. [Company A] - [X]%
        2. [Company B] - [Y]%
        3. [Company C] - [Z]%
        4. [Company D] - [A]%
        5. [Company E] - [B]%
        3. Performance Analysis:
        * Total Return (Absolute): [e.g., +15%]
        * Total Return (Annualized, if applicable): [e.g., +10% over 1 year]
        * Benchmark Comparison: (e.g., S&P 500, MSCI World)
        * Portfolio Return: [X]%
        * Benchmark Return: [Y]%
        * Outperformance/Underperformance: [X-Y]%
        * Return Since Inception: [e.g., +50%]
        4. Asset Allocation:
        * By Asset Class:
        * Equities: [X]%
        * Fixed Income: [Y]%
        * Cash & Equivalents: [Z]%
        * Alternatives (e.g., Real Estate, Commodities): [A]%
        * Visual representation (e.g., pie chart) recommended here.
        * By Sector (for Equities):
        * Technology: [X]%
        * Healthcare: [Y]%
        * Financials: [Z]%
        * Visual representation (e.g., bar chart) recommended here.
        * By Geography:
        * North America: [X]%
        * Europe: [Y]%
        * Emerging Markets: [Z]%
        * Visual representation (e.g., pie chart) recommended here.
        5. Risk Analysis:
        * Diversification Level: (e.g., "Well-diversified," "Concentrated in certain sectors").
        * Volatility: (e.g., "Moderate," "Higher than benchmark").
        * Key Risks Identified: (e.g., "Market downturn," "Interest rate risk," "Concentration risk in a specific stock/sector").
        * Risk-Adjusted Returns (Optional, for advanced analysis):
        * Sharpe Ratio: [X]
        * Sortino Ratio: [Y]
        6. Future Outlook & Recommendations:
        * Current Market View: Briefly state your outlook (e.g., "Cautiously optimistic," "Anticipate continued volatility").
        * Key Opportunities: (e.g., "Potential for growth in renewable energy," "Undervalued sectors").
        * Key Threats: (e.g., "Inflationary pressures," "Geopolitical instability").
        * Proposed Actions/Adjustments (if any): (e.g., "Consider rebalancing towards fixed income," "Increase exposure to value stocks," "Monitor specific underperforming assets").


        Your response MUST be JSON formatted exactly as follows:
        {{
            "Executive Summary": {{}},
            "Portfolio Holdings Overview": {{}},
            "Performance Analysis": {{}},
            "Asset Allocation": {{}},
            "Risk Analysis": {{}},
            "Future Outlook & Recommendations": {{}}
        }}
        """
        # Use a placeholder for the JSON data to avoid issues with curly braces within the JSON itself.
        # The actual JSON data will be passed as a separate argument to format().
        prompt_detailed_protifolio = prompt_template.format(portfolio_data=portfolio_data_json)
        
        response_detailed_portfolio = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt_detailed_protifolio,
            config=generate_content_config
        )
        detailed_portfolio_text = response_detailed_portfolio.text
        print("Detailed Portfolio Summary generated.")
        print(f"Raw response: {detailed_portfolio_text[:500]}...") # Log first 500 chars of response

        # Attempt to parse the JSON response from Gemini
        try:
            # The Gemini API is expected to return a JSON string for the detailed summary.
            # We need to parse this string into a Python dictionary.
            detailed_portfolio_json = json.loads(detailed_portfolio_text)
            print("Detailed Portfolio Summary parsed successfully as JSON.")
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse Gemini detailed summary as JSON: {e}")
            # Return a JSON response with the raw text and the error
            return JsonResponse({
                'error': f'Failed to parse Gemini detailed summary as JSON. Raw response: {detailed_portfolio_text}',
                'raw_response': detailed_portfolio_text # Include raw response for frontend debugging
            }, status=500)

        # Generate Portfolio Summary (concise version)
        print("Generating Concise Portfolio Summary...")
        prompt_portfolio_summary = f"""
        You are a stock's financial analyst. Summarize the following detailed portfolio analysis into approximately 200 words:
        {detailed_portfolio_text}
        """
        response_portfolio_summary = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt_portfolio_summary,
            config=generate_content_config
        )
        portfolio_summary = response_portfolio_summary.text
        print("Concise Portfolio Summary generated.")

        # Generate News for each Topic
        news_list = []
        print("Generating News for each Topic...")
        for topic, details in Topics.items():
            print(f"Fetching news for topic: {topic}")
            prompt_news = f"""
            You are a stock's financial analyst helping with a Daily Stocks Newsletter.
            Using the portfolio summary: "{portfolio_summary}"
            For the following topic: "{topic}" with details: "{details}"
            Use the Google Search tool to find relevant news and insights.
            Provide the output in approximately 100 words.
            """
            try:
                response_news = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt_news,
                    config=generate_content_config
                )
                news_list.append({"topic": topic, "content": response_news.text})
                print(f"News generated for topic: {topic}")
            except Exception as e:
                print(f"ERROR generating news for topic {topic}: {e}")
                news_list.append({"topic": topic, "error": f"Error generating news for topic: {e}"})
        print("News generation complete.")

        # Prepare the final JSON response
        analysis_results = {
            "detailed_portfolio_summary": detailed_portfolio_json, # This is now guaranteed to be a dict
            "portfolio_summary": portfolio_summary,
            "news_by_topic": news_list
        }
        print("Analysis results prepared.")

        return JsonResponse(analysis_results)

    except Exception as e:
        # Catch any unexpected errors during the entire process
        print(f"FATAL ERROR in generate_analysis_view: {e}")
        # Log the traceback for more detailed debugging if needed
        traceback.print_exc()
        return JsonResponse({'error': f'An unexpected server error occurred: {e}'}, status=500)

# --- Existing portfolio_view function remains unchanged ---
# (The rest of the portfolio_view function code is omitted here for brevity but is present in the file)
# ... (rest of portfolio_view function) ...
