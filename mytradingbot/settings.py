# mytradingbot/settings.py

import os
from pathlib import Path
import dj_database_url

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.getenv('SECRET_KEY', 'django-insecure-k%^9o$i591k7c(43x)j0w!g=v8(f*^&j=1s31@t_m3_p%^g8h') # Replace with a strong, secret key in production

# Set DEBUG to False if RENDER environment variable is present (indicating production)
DEBUG = os.getenv('DJANGO_DEBUG', 'True') == 'True' and not os.getenv('RENDER')

ALLOWED_HOSTS = []
if not DEBUG:
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",")
    # Ensure the Render domain is always included in production
    render_domain = 'mytradingbot-f3re.onrender.com'
    if render_domain not in ALLOWED_HOSTS:
        ALLOWED_HOSTS.append(render_domain)
else:
    # For local development, still allow from env or keep empty for default behavior
    ALLOWED_HOSTS = os.getenv("ALLOWED_HOSTS", "").split(",") if os.getenv("ALLOWED_HOSTS") else []

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'bot_app',
    'trading_core',
    'portfolio.apps.PortfolioConfig',
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware', # Add WhiteNoise for static files
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
]

ROOT_URLCONF = 'mytradingbot.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'mytradingbot.wsgi.application'

DATABASES = {
    'default': dj_database_url.config(
        default=f'sqlite:///{BASE_DIR / "db.sqlite3"}',
        conn_max_age=600
    )
}

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]

LANGUAGE_CODE = 'en-us'

# --- Timezone Settings ---
TIME_ZONE = 'Europe/Rome' # <--- Set your desired local timezone here
USE_I18N = True
USE_TZ = True # <--- Crucial for Django's timezone awareness

STATIC_URL = '/static/'
# STATICFILES_DIRS is typically used for project-wide static assets not tied to a specific app.
# Given the structure, relying on APP_DIRS for app-specific static files is more appropriate.
# STATICFILES_DIRS = [
#     os.path.join(BASE_DIR, 'static'),
# ]
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

STATICFILES_STORAGE = 'whitenoise.storage.CompressedManifestStaticFilesStorage' # Configure WhiteNoise storage

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'

# --- Automated Trading Scheduler Interval ---
AUTO_TRADING_INTERVAL_SECONDS = 60 # Run the automated trading task every 1 minute

# --- Logging Configuration ---
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'verbose': {
            'format': '{levelname} {asctime} {module} {process:d} {thread:d} {message}',
            'style': '{',
        },
        'simple': {
            'format': '{levelname} {asctime} {message}',
            'style': '{',
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple',
        },
        'file': {
            'class': 'logging.FileHandler',
            'filename': os.path.join(BASE_DIR, 'trading_bot.log'),
            'formatter': 'verbose',
        },
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console', 'file'],
            'level': os.getenv('DJANGO_LOG_LEVEL', 'INFO'),
            'propagate': False,
        },
        'trading_core': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        'bot_app': {
            'handlers': ['console', 'file'],
            'level': 'INFO',
            'propagate': False,
        },
        
        'portfolio': { 
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    },
}
