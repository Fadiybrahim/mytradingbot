from django.db import models

class PortfolioHolding(models.Model):
    ticker = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=255, null=True, blank=True)
    quantity = models.FloatField()
    average_price = models.FloatField()
    current_price = models.FloatField()
    value = models.FloatField()
    profit = models.FloatField()
    fx_profit = models.FloatField(default=0)
    year_month = models.CharField(max_length=7, null=True, blank=True)
    currency_code = models.CharField(max_length=10)
    instrument_type = models.CharField(max_length=50, null=True, blank=True)
    
    # Fields for EUR conversion
    average_price_eur = models.FloatField(null=True, blank=True)
    current_price_eur = models.FloatField(null=True, blank=True)
    value_eur = models.FloatField(null=True, blank=True)
    allocated_funds = models.FloatField(default=0.0)

    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.ticker})"
