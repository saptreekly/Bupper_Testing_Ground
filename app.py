import os
import sys
import psutil
import socket
import logging
import traceback
import time
from flask import Flask, render_template, request, jsonify
from example import benchmark_optimization
import threading

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def is_port_in_use(port):
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
        for proc in psutil.process_iter():
            try:
                # Get process connections
                connections = proc.connections()
                for conn in connections:
                    if hasattr(conn, 'laddr') and conn.laddr.port == port:
                        logger.info(f"Found process {proc.pid} using port {port}")
                        proc.terminate()
                        processes_terminated += 1
                        logger.info(f"Terminated process {proc.pid}")
                        # Wait for the process to actually terminate
                        proc.wait(timeout=2)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                continue
            except psutil.TimeoutExpired:
                logger.warning(f"Timeout waiting for process {proc.pid} to terminate")
                continue

        if processes_terminated > 0:
            # Give the OS some time to release the port
            logger.info(f"Waiting for port {port} to be released after terminating {processes_terminated} processes")
            time.sleep(2)

            # Verify the port is now free
            if not is_port_in_use(port):
                logger.info(f"Successfully freed port {port}")
            else:
                logger.warning(f"Port {port} is still in use after cleanup")
    except Exception as e:
        logger.error(f"Error cleaning up port {port}: {str(e)}")

# Create static directory if it doesn't exist
static_dir = os.path.join(os.path.dirname(__file__), 'static')
if not os.path.exists(static_dir):
    os.makedirs(static_dir)
    logger.info(f"Created static directory at {static_dir}")

app = Flask(__name__, static_folder='static')

# Store active optimization tasks
active_tasks = {}

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()
        n_cities = data.get('n_cities', 4)
        n_vehicles = data.get('n_vehicles', 1)
        location = data.get('location', 'San Francisco, California, USA')
        backend = data.get('backend', 'qiskit')
        hybrid = data.get('hybrid', False)

        # Cancel flag for the optimization
        cancel_flag = {'cancelled': False}

        def check_cancelled():
            return cancel_flag['cancelled']

        try:
            metrics = benchmark_optimization(
                n_cities=n_cities,
                n_vehicles=n_vehicles,
                place_name=location,
                backend=backend,
                hybrid=hybrid,
                check_cancelled=check_cancelled
            )

            # Generate map files
            if metrics and 'network' in metrics and 'nodes' in metrics:
                routes = metrics.get('routes', [])
                if routes:
                    node_routes = [[metrics['nodes'][i] for i in route] for route in routes]
                    map_filename = f"route_map_{backend}_{'hybrid' if hybrid else 'pure'}_{n_cities}cities"

                    # Create both HTML and PNG versions
                    metrics['network'].create_folium_map(
                        node_routes,
                        save_path=f"static/{map_filename}.html"
                    )
                    metrics['network'].create_static_map(
                        node_routes,
                        save_path=f"static/{map_filename}.png"
                    )

                    # Create a new dictionary with only serializable data
                    serializable_metrics = {
                        'total_time': float(metrics.get('total_time', 0)),
                        'solution_length': float(metrics.get('solution_length', 0)),
                        'quantum_classical_gap': float(metrics.get('quantum_classical_gap', 0)),
                        'n_routes': int(metrics.get('n_routes', 0)),
                        'optimization_time': float(metrics.get('optimization_time', 0))
                    }

                    return jsonify({
                        'success': True,
                        'metrics': serializable_metrics,
                        'map_path': f"/static/{map_filename}.html",
                        'png_path': f"/static/{map_filename}.png"
                    })

        except Exception as opt_error:
            logger.error(f"Optimization error: {str(opt_error)}")
            return jsonify({
                'success': False,
                'error': str(opt_error)
            }), 500

    except Exception as e:
        logger.error(f"Error processing optimization request: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    try:
        port = 3000  # Set fixed port
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            # Try to clean up the port first
            if is_port_in_use(port):
                logger.info(f"Port {port} is in use, attempting to clean up...")
                cleanup_port(port)

            # Double check if the port is now available
            if not is_port_in_use(port):
                logger.info(f"Starting Flask server on port {port}")
                app.run(
                    host='0.0.0.0',
                    port=port,
                    debug=False,
                    use_reloader=False  # Disable reloader to avoid duplicate processes
                )
                break

            retry_count += 1
            logger.warning(f"Port {port} still in use after cleanup attempt {retry_count}")
            time.sleep(2)  # Wait before retrying
        else:
            raise RuntimeError(f"Could not free up port {port} after {max_retries} attempts")

    except Exception as e:
        logger.error("Error starting Flask server:")
        logger.error(traceback.format_exc())
        raise