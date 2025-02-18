import os
import logging
from flask import Flask, render_template, send_from_directory
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Ensure required directories exist
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, 'static')
templates_dir = os.path.join(current_dir, 'templates')

for directory in [static_dir, templates_dir]:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory at {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise

# Initialize Flask app
app = Flask(__name__, 
            static_folder=static_dir,
            template_folder=templates_dir)

@app.route('/')
def index():
    try:
        logger.info("Attempting to render index page")
        return render_template('index.html')
    except Exception as e:
        error_msg = f"Error rendering index: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg, 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    try:
        logger.info(f"Attempting to serve static file: {filename}")
        return send_from_directory(static_dir, filename)
    except Exception as e:
        error_msg = f"Error serving static file {filename}: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg, 404

if __name__ == '__main__':
    try:
        port = 3000
        logger.info(f"Starting Flask server on port {port}")
        app.run(
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        raise