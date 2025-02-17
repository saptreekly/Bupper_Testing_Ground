import os
from flask import Flask, render_template, jsonify, request
import logging
from example import benchmark_optimization
import time

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure static directory exists
os.makedirs('static/route_maps', exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        data = request.get_json()
        n_cities = int(data.get('n_cities', 4))
        n_vehicles = int(data.get('n_vehicles', 1))
        location = data.get('location', 'San Francisco, California, USA')
        backend = data.get('backend', 'qiskit')
        hybrid = data.get('hybrid', False)

        # Add timeout protection
        if n_cities > 6:
            return jsonify({
                'success': False,
                'error': 'Maximum number of cities is 6 to ensure reasonable computation time'
            }), 400

        start_time = time.time()
        timeout = 300  # 5 minutes timeout

        logger.info(f"Starting optimization with {n_cities} cities using {backend} backend")

        def check_cancelled():
            # Check if client disconnected
            if request.environ.get('werkzeug.server.shutdown') or \
               'werkzeug.socket' in request.environ or \
               not request or \
               request.is_disconnected:
                logger.info("Client disconnected, stopping optimization")
                return True
            # Check timeout
            if time.time() - start_time > timeout:
                logger.info("Optimization timed out")
                return True
            return False

        try:
            metrics = benchmark_optimization(
                n_cities=n_cities,
                n_vehicles=n_vehicles,
                place_name=location,
                backend=backend,
                hybrid=hybrid,
                check_cancelled=check_cancelled  # Pass the cancellation check
            )
        except RuntimeError as e:
            if "cancelled by user" in str(e):
                logger.info("Optimization cancelled by user")
                return jsonify({
                    'success': False,
                    'error': 'Optimization cancelled'
                }), 499
            logger.error(f"Runtime error during optimization: {str(e)}")
            return jsonify({
                'success': False,
                'error': 'Optimization failed. Please try again with different parameters or backend.'
            }), 500
        except Exception as e:
            logger.error(f"Error during optimization: {str(e)}")
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

        # Check if process is taking too long
        if time.time() - start_time > timeout:
            return jsonify({
                'success': False,
                'error': 'Optimization timed out. Please try with fewer cities or a different backend.'
            }), 408

        # Extract route information
        routes = metrics.get('routes', [])
        nodes = metrics.get('nodes', [])
        network = metrics.get('network')

        if network and nodes and routes:
            # Generate both HTML and PNG maps
            node_routes = [[nodes[i] for i in route] for route in routes]
            map_path = f"route_maps/route_{backend}_{'hybrid' if hybrid else 'pure'}.html"
            png_path = f"route_maps/route_{backend}_{'hybrid' if hybrid else 'pure'}.png"

            # Save maps in static directory
            full_map_path = os.path.join('static', map_path)
            full_png_path = os.path.join('static', png_path)

            os.makedirs(os.path.dirname(full_map_path), exist_ok=True)
            network.create_folium_map(node_routes, save_path=full_map_path)
            network.create_static_map(node_routes, save_path=full_png_path)

            return jsonify({
                'success': True,
                'map_path': '/' + full_map_path,  # Add leading slash for absolute path
                'png_path': '/' + full_png_path,
                'metrics': {
                    'total_time': metrics.get('total_time'),
                    'solution_length': metrics.get('solution_length'),
                    'quantum_classical_gap': metrics.get('quantum_classical_gap'),
                    'n_routes': metrics.get('n_routes')
                }
            })

        return jsonify({
            'success': False,
            'error': 'Failed to generate routes'
        }), 400

    except Exception as e:
        logger.error(f"Error in optimization: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000, debug=True)