import http.server
import socketserver
import logging
import os

PORT = 8080
Handler = http.server.SimpleHTTPRequestHandler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def serve_map():
    try:
        with socketserver.TCPServer(("0.0.0.0", PORT), Handler) as httpd:
            logger.info(f"Serving map at port {PORT}")
            httpd.serve_forever()
    except Exception as e:
        logger.error(f"Error serving map: {str(e)}")
        raise

if __name__ == "__main__":
    serve_map()