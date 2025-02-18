import os
import logging
from flask import Flask

# Configure logging (from original code)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app (from edited code)
app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello World!"

if __name__ == '__main__':
    try:
        port = 3000
        logger.info(f"Starting minimal Flask server on port {port}")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(f"Error type: {type(e)}")
        raise