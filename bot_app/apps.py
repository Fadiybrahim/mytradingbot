# bot_app/apps.py
from django.apps import AppConfig
import threading
import time
import logging
import os
from django.conf import settings

logger = logging.getLogger(__name__)

class BotAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bot_app'

    _scheduler_active = False
    _scheduler_thread = None

    def ready(self):
        # IMPORTANT: This check ensures the scheduler only starts once in the main process
        if os.environ.get('RUN_MAIN', None) == 'true' and not BotAppConfig._scheduler_active:
            logger.info("Initializing automated trading scheduler...")
            
            from trading_core.tasks import execute_automated_trading_task

            interval = getattr(settings, 'AUTO_TRADING_INTERVAL_SECONDS', 60)

            def _run_scheduler_loop():
                while BotAppConfig._scheduler_active:
                    try:
                        execute_automated_trading_task()
                    except Exception as e:
                        logger.exception(f"CRITICAL: Error in automated trading task in scheduler: {e}")
                    
                    for _ in range(interval):
                        if not BotAppConfig._scheduler_active:
                            break
                        time.sleep(1)
                logger.info("Automated trading scheduler loop finished.")

            BotAppConfig._scheduler_active = True
            BotAppConfig._scheduler_thread = threading.Thread(target=_run_scheduler_loop, daemon=True)
            BotAppConfig._scheduler_thread.start()
            logger.info(f"Automated trading scheduler started. Will run every {interval} seconds.")
        elif os.environ.get('RUN_MAIN', None) != 'true':
            logger.debug("Skipping scheduler startup in non-main process.")
        else:
            logger.debug("Scheduler already active, skipping startup.")