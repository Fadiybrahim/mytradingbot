import csv
from datetime import datetime
from django.core.management.base import BaseCommand
from trading_core.models import Trading212Instrument

class Command(BaseCommand):
    help = 'Imports instrument data from trading212_instro.csv into the Trading212Instrument model.'

    def handle(self, *args, **options):
        csv_file_path = 'trading212_instro.csv'
        
        try:
            with open(csv_file_path, mode='r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    # Prepare data for the model
                    ticker = row.get('ticker')
                    if not ticker:
                        self.stdout.write(self.style.WARNING(f"Skipping row due to missing ticker: {row}"))
                        continue

                    # Convert 'addedOn' to date if it exists and is not empty
                    added_on_str = row.get('addedOn')
                    added_on_date = None
                    if added_on_str:
                        try:
                            # Parse ISO 8601 format date string (e.g., '2023-11-02T16:28:13.000+02:00')
                            # We only need the date part.
                            added_on_date = datetime.fromisoformat(added_on_str).date()
                        except ValueError:
                            self.stdout.write(self.style.WARNING(f"Could not parse date '{added_on_str}' for ticker {ticker}. Expected ISO 8601 format (e.g., YYYY-MM-DDTHH:MM:SS.sss+ZZ:ZZ)."))
                            added_on_date = None # Set to None if parsing fails

                    # Process maxOpenQuantity, ensuring it's an integer or None
                    max_open_quantity_str = row.get('maxOpenQuantity')
                    max_open_quantity = None
                    if max_open_quantity_str:
                        try:
                            # Attempt to convert to float first to handle '10339.0' format, then to int
                            max_open_quantity = int(float(max_open_quantity_str))
                        except ValueError:
                            self.stdout.write(self.style.WARNING(f"Could not parse maxOpenQuantity '{max_open_quantity_str}' for ticker {ticker}. Expected a number."))
                            max_open_quantity = None # Set to None if parsing fails

                    # Get or create the instrument
                    instrument, created = Trading212Instrument.objects.update_or_create(
                        ticker=ticker,
                        defaults={
                            'type': row.get('type', ''),
                            'workingScheduleId': row.get('workingScheduleId', ''),
                            'isin': row.get('isin', ''),
                            'currencyCode': row.get('currencyCode', ''),
                            'name': row.get('name', ''),
                            'shortName': row.get('shortName', ''),
                            'maxOpenQuantity': max_open_quantity,
                            'addedOn': added_on_date,
                        }
                    )
                    
                    if created:
                        self.stdout.write(self.style.SUCCESS(f"Successfully created instrument: {instrument.ticker}"))
                    else:
                        self.stdout.write(f"Successfully updated instrument: {instrument.ticker}")

            self.stdout.write(self.style.SUCCESS("Data import from trading212_instro.csv completed successfully."))

        except FileNotFoundError:
            self.stderr.write(self.style.ERROR(f"Error: The file '{csv_file_path}' was not found."))
        except Exception as e:
            self.stderr.write(self.style.ERROR(f"An unexpected error occurred: {e}"))
