import os
import logging
import socket
import psutil
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
import traceback

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Changed to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_port_in_use(port):
    """Check if a port is in use"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(('0.0.0.0', port))
            return False
        except socket.error:
            return True

def cleanup_port(port):
    """Attempt to clean up a port that might be in use"""
    try:
        processes_terminated = 0
        for proc in psutil.process_iter(['pid', 'name']):
            try:
                # Get process connections without specifying 'connections' in process_iter
                for conn in proc.net_connections('inet'):  # Updated to use net_connections
                    if conn.laddr.port == port:
                        logger.info(f"Found process {proc.pid} using port {port}")
                        proc.terminate()
                        processes_terminated += 1
                        logger.info(f"Terminated process {proc.pid}")
                        proc.wait(timeout=2)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except Exception as e:
                logger.error(f"Error checking process: {str(e)}")
                continue

        if processes_terminated > 0:
            logger.info(f"Terminated {processes_terminated} processes using port {port}")
            time.sleep(2)  # Give OS time to release the port
            return not is_port_in_use(port)
        return True
    except Exception as e:
        logger.error(f"Error cleaning up port {port}: {str(e)}")
        return False

# Ensure required directories exist
current_dir = os.path.dirname(os.path.abspath(__file__))
static_dir = os.path.join(current_dir, 'static')
templates_dir = os.path.join(current_dir, 'templates')

for directory in [static_dir, templates_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory at {directory}")

try:
    app = Flask(__name__, 
                static_folder=static_dir,
                template_folder=templates_dir)
    app.config['SECRET_KEY'] = 'secret!'
    logger.info("Flask app initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Flask app: {str(e)}")
    logger.error(traceback.format_exc())
    raise

@app.route('/health')
def health_check():
    """Simple health check endpoint"""
    logger.info("Health check endpoint called")
    return "OK", 200

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
        port = 8080
        max_retries = 3
        retry_count = 0

        # Initial port cleanup
        if is_port_in_use(port):
            logger.info(f"Port {port} is in use, attempting cleanup before start")
            if not cleanup_port(port):
                logger.error(f"Failed to clean up port {port}")
                raise RuntimeError(f"Could not free up port {port}")

        logger.info(f"Starting Flask server on port {port}")
        logger.info(f"Static folder: {static_dir}")
        logger.info(f"Templates folder: {templates_dir}")
        app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)  # Disabled reloader
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        raise