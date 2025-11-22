release: python manage.py collectstatic --noinput
web: gunicorn mytradingbot.wsgi --bind 0.0.0.0:$PORT

