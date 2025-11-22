release: python manage.py collectstatic --noinput
web: gunicorn mytradingbot.wsgi --log-file - --bind 0.0.0.0:$PORT
