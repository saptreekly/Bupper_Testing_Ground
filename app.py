import os
import logging
from flask import Flask, render_template, send_from_directory, request, jsonify
import traceback
from example import benchmark_optimization
import uuid

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

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        logger.info("Received optimization request")
        data = request.get_json()

        if not data:
            logger.error("No data provided in optimization request")
            return jsonify({"success": False, "error": "No data provided"}), 400

        n_cities = data.get('n_cities', 4)
        n_vehicles = data.get('n_vehicles', 1)
        location = data.get('location', 'San Francisco, California, USA')
        backend = data.get('backend', 'pennylane')
        hybrid = data.get('hybrid', False)

        logger.info(f"Optimization parameters: cities={n_cities}, vehicles={n_vehicles}, "
                   f"location={location}, backend={backend}, hybrid={hybrid}")

        # Generate unique filenames for this optimization
        run_id = str(uuid.uuid4())[:8]
        map_filename = f'route_map_{run_id}.html'
        png_filename = f'route_map_{run_id}.png'

        def progress_callback(step, data):
            logger.info(f"Optimization progress - Step {step}: {data}")

        try:
            metrics = benchmark_optimization(
                n_cities=n_cities,
                n_vehicles=n_vehicles,
                place_name=location,
                backend=backend,
                hybrid=hybrid,
                progress_callback=progress_callback
            )

            # Save map files
            map_path = os.path.join(static_dir, map_filename)
            png_path = os.path.join(static_dir, png_filename)

            if 'network' in metrics and 'nodes' in metrics:
                routes = metrics.get('routes', [])
                if routes:
                    node_routes = [[metrics['nodes'][i] for i in route] for route in routes]
                    metrics['network'].create_folium_map(node_routes, save_path=map_path)
                    metrics['network'].create_static_map(node_routes, save_path=png_path)
                    logger.info(f"Successfully generated route maps: {map_filename}")

            # Clean up metrics for JSON response
            response_metrics = {
                'total_time': metrics.get('total_time', 0),
                'solution_length': metrics.get('solution_length', 0),
                'quantum_classical_gap': metrics.get('quantum_classical_gap', 0),
                'n_routes': len(metrics.get('routes', []))
            }

            return jsonify({
                'success': True,
                'metrics': response_metrics,
                'map_path': f'/static/{map_filename}',
                'png_path': f'/static/{png_filename}'
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