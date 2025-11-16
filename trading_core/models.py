from django.db import models
from django.utils import timezone
from decimal import Decimal

class ConfiguredStock(models.Model):
    """Stores the configuration for a single trading bot instance."""
    yf_ticker = models.CharField(max_length=20, unique=True, help_text="Yahoo Finance ticker symbol (e.g., AAPL, SAP.DE)")
    trading212_ticker = models.CharField(max_length=20, help_text="Trading 212 instrument ticker (e.g., AAPL_EQ, SAPd_EQ)")
    rsi_window = models.IntegerField(default=14, help_text="Window size for RSI calculation.")
    buy_threshold = models.IntegerField(default=35, help_text="RSI level below which a 'Buy' signal is generated.")
    sell_threshold = models.IntegerField(default=65, help_text="RSI level above which a 'Sell' signal is generated.")
    trade_quantity = models.IntegerField(default=1, help_text="Number of shares to buy/sell per trade.")
    is_active_for_trading = models.BooleanField(default=False, help_text="Set to True to enable automated trading for this stock via in-app scheduler.")
    last_run = models.DateTimeField(null=True, blank=True, help_text="Timestamp of the last time the automated trading logic ran for this stock.")
    
    total_realized_pnl = models.DecimalField(max_digits=15, decimal_places=2, default=0.00,
                                            help_text="Accumulated P&L from closed trades.")

    def __str__(self):
        return f"{self.yf_ticker} ({self.trading212_ticker})"

class Trade(models.Model):
    """Logs individual trade actions executed by the bot."""
    stock_config = models.ForeignKey(ConfiguredStock, on_delete=models.CASCADE, related_name='trades')
    timestamp = models.DateTimeField(default=timezone.now)
    signal_generated = models.CharField(max_length=10, choices=[('Buy', 'Buy'), ('Sell', 'Sell'), ('Hold', 'Hold')],
                                      help_text="Signal that triggered the trade (or 'Hold' if no trade).")
    quantity = models.IntegerField(help_text="Number of shares involved in the trade (absolute value).")
    price_at_execution = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True,
                                            help_text="Price at which the order was (or would be) executed.")
    order_type = models.CharField(max_length=20, default='Market', help_text="Type of order placed (e.g., Market).")
    trading212_order_id = models.CharField(max_length=100, blank=True, null=True,
                                            help_text="Order ID returned by Trading 212 API.")
    status = models.CharField(max_length=20, choices=[('Success', 'Success'), ('Failed', 'Failed'), ('Simulated', 'Simulated')],
                              default='Simulated', help_text="Status of the trade execution.")
    response_text = models.TextField(blank=True, null=True, help_text="Full response from the trading API.")
    
    realized_pnl_for_trade = models.DecimalField(max_digits=15, decimal_places=2, null=True, blank=True,
                                                help_text="Realized P&L for this specific trade if it closed a position.")

    def __str__(self):
        return f"{self.signal_generated} {self.quantity} of {self.stock_config.yf_ticker} at {self.price_at_execution} (Status: {self.status})"

class HistoricalData(models.Model):
    """Stores historical price, RSI, and signal data for each configured stock."""
    stock_config = models.ForeignKey(ConfiguredStock, on_delete=models.CASCADE, related_name='historical_data')
    timestamp = models.DateTimeField(db_index=True)
    open_price = models.FloatField(null=True, blank=True)
    high_price = models.FloatField(null=True, blank=True)
    low_price = models.FloatField(null=True, blank=True)
    close_price = models.FloatField(null=True, blank=True)
    volume = models.BigIntegerField(null=True, blank=True)
    rsi = models.FloatField(null=True, blank=True)
    signal = models.CharField(max_length=10, null=True, blank=True, choices=[('Buy', 'Buy'), ('Sell', 'Sell'), ('Hold', 'Hold')])

    class Meta:
        ordering = ['-timestamp']
        unique_together = ('stock_config', 'timestamp') # Ensure no duplicate entries for a given timestamp and stock

    def __str__(self):
        return f"{self.stock_config.yf_ticker} - {self.timestamp.strftime('%Y-%m-%d %H:%M')} - {self.signal}"

class Trading212Instrument(models.Model):
    """Stores instrument data from Trading 212."""
    ticker = models.CharField(max_length=50, unique=True)
    type = models.CharField(max_length=50)
    workingScheduleId = models.CharField(max_length=50)
    isin = models.CharField(max_length=50, blank=True, null=True)
    currencyCode = models.CharField(max_length=10)
    name = models.CharField(max_length=255)
    shortName = models.CharField(max_length=100)
    maxOpenQuantity = models.IntegerField(null=True, blank=True)
    addedOn = models.DateField(null=True, blank=True)

    def __str__(self):
        return f"{self.shortName} ({self.ticker})"
