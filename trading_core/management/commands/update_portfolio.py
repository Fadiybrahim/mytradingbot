from django.core.management.base import BaseCommand
from trading_core.actions import fetch_and_update_portfolio_holdings

class Command(BaseCommand):
    help = 'Fetches and updates portfolio holdings from Trading212.'

    def handle(self, *args, **options):
        success, message = fetch_and_update_portfolio_holdings()
        if success:
            self.stdout.write(self.style.SUCCESS(message))
        else:
            self.stderr.write(self.style.ERROR(message))
