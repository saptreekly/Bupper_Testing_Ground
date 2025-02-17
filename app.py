import os
import sys
import logging
import socket
import time

# Configure logging to show detailed error messages
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG for maximum verbosity
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    # Check if Flask is importable
    logger.info("Attempting to import Flask...")
    from flask import Flask, render_template, request, jsonify
    logger.info("Flask import successful")

    # Create basic Flask app
    logger.info("Initializing Flask application...")
    app = Flask(__name__)

    @app.route('/')
    def index():
        return render_template('index.html')

    if __name__ == '__main__':
        try:
            # Get port from environment or use default
            port = int(os.environ.get('PORT', 50000))
            logger.info(f"Starting Flask server on port {port}...")

            # Test if port is available
            test_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            test_socket.settimeout(1)  # Set timeout for socket operations

            try:
                test_socket.bind(('0.0.0.0', port))
                test_socket.close()
                logger.info(f"Port {port} is available")
            except socket.error as e:
                logger.error(f"Port {port} is not available: {e}")
                raise

            # Log system information
            logger.info(f"Python version: {sys.version}")
            logger.info(f"Working directory: {os.getcwd()}")
            logger.info(f"Environment: {os.environ.get('FLASK_ENV', 'not set')}")

            logger.info("Starting Flask application server...")
            app.run(
                host='0.0.0.0',
                port=port,
                debug=False,
                use_reloader=False
            )

        except Exception as e:
            logger.error(f"Failed to start Flask server: {str(e)}", exc_info=True)
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            sys.exit(1)

except Exception as e:
    logger.error(f"Critical error during app initialization: {str(e)}", exc_info=True)
    sys.exit(1)