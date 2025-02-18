import os
import logging
from flask import Flask, render_template, send_from_directory, request, jsonify
import traceback
from example import benchmark_optimization
import uuid
from flask_socketio import SocketIO, emit

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
route_maps_dir = os.path.join(static_dir, 'route_maps')

for directory in [static_dir, templates_dir, route_maps_dir]:
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
            logger.info(f"Created directory at {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {str(e)}")
            raise

# Initialize Flask app and SocketIO
app = Flask(__name__, 
            static_folder=static_dir,
            template_folder=templates_dir)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

@app.route('/')
def index():
    try:
        logger.info("Attempting to render index page")
        return render_template('dashboard.html')
    except Exception as e:
        error_msg = f"Error rendering index: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return error_msg, 500

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        logger.info("Received optimization request")
        data = request.get_json()

        if not data:
            logger.error("No data provided in optimization request")
            return jsonify({"success": False, "error": "No data provided"}), 400

        # Generate unique filenames for this optimization
        run_id = str(uuid.uuid4())[:8]
        map_filename = f'route_maps/route_map_{run_id}.html'
        png_filename = f'route_maps/route_map_{run_id}.png'

        logger.info(f"Generated filenames: map={map_filename}, png={png_filename}")

        def progress_callback(step, update_data):
            logger.info(f"Progress update - Step {step}: {update_data}")
            try:
                socketio.emit(
                    f'optimization_progress_{run_id}',
                    {
                        'step': step,
                        'data': update_data,
                        'message': update_data.get('status', 'Processing...'),
                        'timestamp': str(uuid.uuid4())[:8]
                    }
                )
            except Exception as e:
                logger.error(f"Error in progress callback: {str(e)}")

        try:
            metrics = benchmark_optimization(
                n_cities=data.get('n_cities', 4),
                n_vehicles=data.get('n_vehicles', 1),
                place_name=data.get('location', 'San Francisco, California, USA'),
                backend=data.get('backend', 'pennylane'),
                hybrid=data.get('hybrid', False),
                progress_callback=progress_callback
            )

            # Save map files
            map_path = os.path.join(static_dir, map_filename)
            png_path = os.path.join(static_dir, png_filename)

            logger.info(f"Attempting to save maps to: {map_path} and {png_path}")

            if 'network' in metrics and 'nodes' in metrics:
                routes = metrics.get('routes', [])
                if routes:
                    node_routes = [[metrics['nodes'][i] for i in route] for route in routes]
                    metrics['network'].create_folium_map(node_routes, save_path=map_path)
                    metrics['network'].create_static_map(node_routes, save_path=png_path)
                    logger.info(f"Successfully generated route maps: {map_filename}")
                else:
                    logger.error("No routes found in metrics")
            else:
                logger.error("Missing network or nodes in metrics")

            response_metrics = {
                'total_time': metrics.get('total_time', 0),
                'solution_length': metrics.get('solution_length', 0),
                'quantum_classical_gap': metrics.get('quantum_classical_gap', 0),
                'n_routes': len(metrics.get('routes', [])),
                'cost_terms': len(metrics.get('cost_terms', [])),
                'qubo_sparsity': metrics.get('qubo_sparsity', 0)
            }

            return jsonify({
                'success': True,
                'metrics': response_metrics,
                'map_path': f'/static/{map_filename}',
                'png_path': f'/static/{png_filename}',
                'task_id': run_id
            })

        except Exception as opt_error:
            logger.error(f"Optimization error: {str(opt_error)}")
            logger.error(traceback.format_exc())
            return jsonify({
                'success': False,
                'error': f"Optimization failed: {str(opt_error)}"
            }), 500

    except Exception as e:
        logger.error(f"Error in optimize endpoint: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': f"Server error: {str(e)}"
        }), 500

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
        port = 5001
        logger.info(f"Starting Flask server with SocketIO on port {port}")
        socketio.run(
            app,
            host='0.0.0.0',
            port=port,
            debug=False,
            use_reloader=False
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(traceback.format_exc())
        raise