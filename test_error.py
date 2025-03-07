import logging
import traceback

logging.basicConfig(level=logging.DEBUG)

try:
    context = None
    context.get('test')
except Exception as e:
    print(traceback.format_exc())
