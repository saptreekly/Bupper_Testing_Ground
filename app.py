import os
import sys
import psutil
import socket
import logging
import traceback
import time
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
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
app.config['SECRET_KEY'] = 'secret!'

# Initialize SocketIO with explicit configuration
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    logger=True,
    engineio_logger=True,
    ping_timeout=60,
    ping_interval=25,
    async_mode='threading'
)

@socketio.on_error()
def error_handler(e):
    logger.error(f"SocketIO error: {str(e)}")

# Store active optimization tasks
active_tasks = {}

@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected to WebSocket from {request.sid}")
    try:
        emit('connection_established', {'status': 'connected'})
    except Exception as e:
        logger.error(f"Error in connect handler: {str(e)}")

@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f"Client disconnected from WebSocket: {request.sid}")

def emit_progress(task_id, message, data=None):
    """Emit progress update via WebSocket"""
    try:
        update = {
            'message': message,
            'timestamp': time.strftime('%H:%M:%S'),
            'data': data
        }
        logger.info(f"Emitting progress for task {task_id}: {message}")
        socketio.emit(f'optimization_progress_{task_id}', update)
        logger.info(f"Successfully emitted progress for task {task_id}")
    except Exception as e:
        logger.error(f"Error emitting progress: {str(e)}")

def optimization_progress_callback(task_id):
    """Create a callback function for the optimization process"""
    def callback(step, data):
        try:
            if isinstance(data, dict):
                status = f"Step {step}/{data.get('total_steps', '?')}"
                if data.get('cost') is not None:
                    status += f" - Cost: {data['cost']:.4f}"
                    if data.get('best_cost') is not None:
                        status += f" (Best: {data['best_cost']:.4f})"
            else:
                # Fallback for simple cost value
                status = f"Step {step} - Cost: {float(data):.4f}"
                data = {
                    'step': step,
                    'cost': float(data),
                    'progress': step / 100,  # Assuming 100 steps total
                    'status': 'Optimizing'
                }
            logger.info(f"Progress callback for task {task_id}: {status}")
            emit_progress(task_id, status, data)
        except Exception as e:
            logger.error(f"Error in optimization callback: {str(e)}")
    return callback

@app.route('/')
def index():
    try:
        return render_template('index.html')
    except Exception as e:
        logger.error(f"Error rendering index: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Explicitly serve files from static directory"""
    try:
        logger.info(f"Serving static file: {filename}")
        return send_from_directory('static', filename)
    except Exception as e:
        logger.error(f"Error serving static file {filename}: {str(e)}")
        return f"Error: {str(e)}", 404

@app.route('/dashboard')
def dashboard():
    """Serve the interactive visualization dashboard."""
    try:
        return render_template('dashboard.html')
    except Exception as e:
        logger.error(f"Error rendering dashboard: {str(e)}")
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
        task_id = str(time.time())

        logger.info(f"Starting optimization with {n_cities} cities, {n_vehicles} vehicles in {location}")

        # Limit maximum steps based on problem size to prevent long running optimizations
        max_steps = min(50, 100 // n_cities)  # Reduce steps for larger problems
        timeout = 60  # Set 60 second timeout

        try:
            metrics = benchmark_optimization(
                n_cities=n_cities,
                n_vehicles=n_vehicles,
                place_name=location,
                backend=backend,
                hybrid=hybrid,
                max_steps=max_steps,
                timeout=timeout,
                progress_callback=optimization_progress_callback(task_id)
            )

            # Generate map files
            if metrics and 'network' in metrics and 'nodes' in metrics:
                routes = metrics.get('routes', [])
                if routes:
                    logger.info(f"Raw routes received: {routes}")
                    node_routes = [[metrics['nodes'][i] for i in route] for route in routes]
                    logger.info(f"Node routes generated: {node_routes}")

                    # Log coordinates for verification
                    coords = metrics['network'].get_node_coordinates([node for route in node_routes for node in route])
                    logger.info(f"Route coordinates: {coords}")

                    map_filename = f"route_map_{backend}_{'hybrid' if hybrid else 'pure'}_{n_cities}cities"

                    # Ensure static directory exists and is writable
                    os.makedirs('static', exist_ok=True)

                    # Create both HTML and PNG versions
                    map_path = os.path.join('static', f"{map_filename}.html")
                    png_path = os.path.join('static', f"{map_filename}.png")

                    logger.info(f"Generating map files at: {map_path} and {png_path}")

                    metrics['network'].create_folium_map(
                        node_routes,
                        save_path=map_path
                    )
                    metrics['network'].create_static_map(
                        node_routes,
                        save_path=png_path
                    )

                    # Verify files were created
                    if os.path.exists(map_path) and os.path.exists(png_path):
                        logger.info(f"Map files generated successfully")
                        logger.info(f"HTML file size: {os.path.getsize(map_path)} bytes")
                        logger.info(f"PNG file size: {os.path.getsize(png_path)} bytes")
                    else:
                        logger.error(f"Failed to generate map files")
                        return jsonify({
                            'success': False,
                            'error': 'Failed to generate map files'
                        }), 500

                    # Enhanced metrics for visualization with reduced cost history
                    serializable_metrics = {
                        'total_time': float(metrics.get('total_time', 0)),
                        'solution_length': float(metrics.get('solution_length', 0)),
                        'quantum_classical_gap': float(metrics.get('quantum_classical_gap', 0)),
                        'n_routes': int(metrics.get('n_routes', 0)),
                        'initial_depth': int(metrics.get('initial_circuit_depth', 0)),
                        'optimized_depth': int(metrics.get('optimized_circuit_depth', 0)),
                        'cost_history': metrics.get('cost_history', [])[-20:],  # Only last 20 points for visualization
                        'backend_times': {
                            'qiskit': float(metrics.get('qiskit_time', 0)),
                            'pennylane': float(metrics.get('pennylane_time', 0))
                        }
                    }

                    return jsonify({
                        'success': True,
                        'task_id': task_id,  # Include task_id in response
                        'metrics': serializable_metrics,
                        'map_path': f"/static/{map_filename}.html",
                        'png_path': f"/static/{map_filename}.png"
                    })
                else:
                    logger.error("No routes found in metrics")
                    return jsonify({
                        'success': False,
                        'error': 'No routes generated'
                    }), 500
            else:
                logger.error("Missing required data in metrics")
                return jsonify({
                    'success': False,
                    'error': 'Missing required data for map generation'
                }), 500

        except Exception as opt_error:
            logger.error(f"Optimization error: {str(opt_error)}")
            emit_progress(task_id, "Error", {"error": str(opt_error)})
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
        port = 3000
        max_retries = 3
        retry_count = 0

        while retry_count < max_retries:
            if is_port_in_use(port):
                logger.info(f"Port {port} is in use, attempting to clean up...")
                cleanup_port(port)

            if not is_port_in_use(port):
                logger.info(f"Starting Flask server with SocketIO on port {port}")
                socketio.run(
                    app,
                    host='0.0.0.0',
                    port=port,
                    debug=True,
                    use_reloader=False,  # Disable reloader to prevent conflicts
                    allow_unsafe_werkzeug=True,
                    log_output=True
                )
                break

            retry_count += 1
            logger.warning(f"Port {port} still in use after cleanup attempt {retry_count}")
            time.sleep(2)
        else:
            raise RuntimeError(f"Could not free up port {port} after {max_retries} attempts")

    except Exception as e:
        logger.error("Error starting Flask server:")
        logger.error(traceback.format_exc())
        raise