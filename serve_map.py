import http.server
import socketserver
import logging
import os

PORT = 8080
Handler = http.server.SimpleHTTPRequestHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serve_map():
    logger.warning("This server is deprecated. Please use the Flask app (app.py) instead.")
    return

if __name__ == "__main__":
    logger.warning("This script is deprecated. Please use 'python app.py' instead.")