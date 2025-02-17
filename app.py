import os
import sys
from flask import Flask, render_template, jsonify, request, send_from_directory
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    app = Flask(__name__)
    app.static_folder = 'static'  # Explicitly set static folder

    # Ensure static directory exists with proper permissions
    static_folder = os.path.join(os.getcwd(), 'static')
    route_maps_folder = os.path.join(static_folder, 'route_maps')

    logger.info(f"Creating static directories...")
    os.makedirs(static_folder, exist_ok=True)
    os.makedirs(route_maps_folder, exist_ok=True)

    logger.info(f"Static folder: {static_folder}")
    logger.info(f"Route maps folder: {route_maps_folder}")

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
            logger.info(f"Static directory path: {os.path.abspath('static')}")

            def check_cancelled():
                if request.environ.get('werkzeug.server.shutdown'):
                    logger.info("Server shutdown requested")
                    return True
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
                    check_cancelled=check_cancelled
                )
                logger.info(f"Optimization completed successfully. Metrics: {metrics}")
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

            logger.info(f"Routes generated: {routes}")
            logger.info(f"Nodes: {nodes}")

            if not all([routes, nodes, network]):
                logger.error("Missing required route data - routes: %s, nodes: %s, network: %s",
                            bool(routes), bool(nodes), bool(network))
                return jsonify({
                    'success': False,
                    'error': 'Failed to generate route data'
                }), 500

            # Generate both HTML and PNG maps
            node_routes = [[nodes[i] for i in route] for route in routes]
            logger.info(f"Node routes for visualization: {node_routes}")

            timestamp = int(time.time())
            map_path = f"route_maps/route_{backend}_{'hybrid' if hybrid else 'pure'}_{timestamp}.html"
            png_path = f"route_maps/route_{backend}_{'hybrid' if hybrid else 'pure'}_{timestamp}.png"

            # Save maps in static directory
            full_map_path = os.path.join('static', map_path)
            full_png_path = os.path.join('static', png_path)

            logger.info(f"Generating maps: HTML={full_map_path}, PNG={full_png_path}")
            os.makedirs(os.path.dirname(full_map_path), exist_ok=True)

            try:
                # Get coordinates for validation
                for route in node_routes:
                    coords = network.get_node_coordinates(route)
                    logger.info(f"Route coordinates: {coords}")

                network.create_folium_map(node_routes, save_path=full_map_path)
                network.create_static_map(node_routes, save_path=full_png_path)
                logger.info("Successfully generated maps")

                # Verify file existence
                if not os.path.exists(full_map_path) or not os.path.exists(full_png_path):
                    logger.error("Map files not created even though no exception was raised")
                    return jsonify({
                        'success': False,
                        'error': 'Map files were not created properly'
                    }), 500

            except Exception as e:
                logger.error(f"Error generating maps: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': 'Failed to generate route visualization'
                }), 500

            return jsonify({
                'success': True,
                'map_path': '/static/' + map_path,
                'png_path': '/static/' + png_path,
                'metrics': {
                    'total_time': metrics.get('total_time'),
                    'solution_length': metrics.get('solution_length'),
                    'quantum_classical_gap': metrics.get('quantum_classical_gap'),
                    'n_routes': metrics.get('n_routes')
                }
            })

        except Exception as e:
            logger.error(f"Error in optimization: {str(e)}", exc_info=True)
            return jsonify({
                'success': False,
                'error': str(e)
            }), 500

    @app.route('/static/<path:path>')
    def serve_static(path):
        logger.info(f"Serving static file: {path}")
        return send_from_directory('static', path)

    if __name__ == '__main__':
        try:
            port = 50000
            logger.info(f"Starting Flask server on port {port}...")
            logger.info("Static folder path: %s", os.path.abspath('static'))
            app.run(host='0.0.0.0', port=port, debug=True, use_reloader=False)
        except OSError as e:
            if "Address already in use" in str(e):
                logger.error(f"Port {port} is in use by another program. Error: {str(e)}")
                raise
            else:
                logger.error(f"Failed to start server: {str(e)}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error starting server: {str(e)}")
            raise

except Exception as e:
    logger.error(f"Error during Flask app initialization: {str(e)}", exc_info=True)
    sys.exit(1)