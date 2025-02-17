import osmnx as ox
import networkx as nx
import folium
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
import time
from shapely.geometry import box

logger = logging.getLogger(__name__)

class StreetNetwork:
    """Handle real street network data from OpenStreetMap."""

    def __init__(self, place_name: str = "San Francisco, California, USA"):
        """Initialize street network for a given location."""
        try:
            logger.info(f"Fetching street network for {place_name}")
            start_time = time.time()

            # Configure osmnx
            ox.config(use_cache=True, log_console=True)

            # Set up bounding box for San Francisco to limit the area
            if "San Francisco" in place_name:
                north, south = 37.8120, 37.7067
                east, west = -122.3555, -122.5185
                logger.info("Using San Francisco bounding box to limit network size")
                try:
                    self.G = ox.graph_from_bbox(north, south, east, west,
                                          network_type='drive', simplify=True)
                except Exception as bbox_error:
                    logger.error(f"Failed to fetch with bounding box: {str(bbox_error)}")
                    logger.info("Falling back to place name query")
                    self.G = ox.graph_from_place(place_name,
                                            network_type='drive', simplify=True)
            else:
                logger.info("Fetching network by place name")
                self.G = ox.graph_from_place(place_name,
                                        network_type='drive', simplify=True)

            logger.info(f"Network fetched in {time.time() - start_time:.1f}s")

            # Convert to non-directional graph using NetworkX's method
            logger.info("Converting to undirected graph")
            self.G = self.G.to_undirected()

            # Project the graph to use meters for distance calculations
            logger.info("Projecting graph to metric coordinates")
            self.G = ox.project_graph(self.G)

            # Get node positions
            logger.info("Extracting node positions")
            self.node_positions = ox.graph_to_gdfs(self.G, edges=False)

            logger.info(f"Successfully loaded network with {len(self.G.nodes)} nodes and {len(self.G.edges)} edges")
            logger.info(f"Total initialization time: {time.time() - start_time:.1f}s")
        except Exception as e:
            logger.error(f"Error initializing street network: {str(e)}")
            raise RuntimeError(f"Failed to initialize street network: {str(e)}")

    def get_random_nodes(self, n: int, min_distance: float = 500) -> List[int]:
        """Get n random nodes that are at least min_distance meters apart."""
        try:
            start_time = time.time()
            logger.info(f"Selecting {n} random nodes with minimum distance of {min_distance}m")

            all_nodes = list(self.G.nodes())
            selected_nodes = []
            attempts = 0
            max_attempts = 1000

            # Always include a node near the center for depot
            center_point = self.node_positions.unary_union.centroid
            depot_node = ox.distance.nearest_nodes(self.G, center_point.x, center_point.y)
            selected_nodes.append(depot_node)
            logger.info(f"Selected depot node at ({center_point.x:.4f}, {center_point.y:.4f})")

            # Gradually reduce min_distance if we can't find enough nodes
            while len(selected_nodes) < n and min_distance > 100:
                for _ in range(max_attempts):
                    if len(selected_nodes) >= n:
                        break

                    node = np.random.choice(all_nodes)
                    valid = True

                    for selected in selected_nodes:
                        try:
                            distance = nx.shortest_path_length(self.G, node, selected, weight='length')
                            if distance < min_distance:
                                valid = False
                                break
                        except nx.NetworkXNoPath:
                            valid = False
                            break

                    if valid:
                        selected_nodes.append(node)
                        logger.debug(f"Selected node {len(selected_nodes)}/{n}")

                min_distance *= 0.8  # Reduce min_distance by 20% if we can't find enough nodes

            if len(selected_nodes) < n:
                logger.warning(f"Could only find {len(selected_nodes)} nodes meeting distance criteria")
                # Fill remaining positions with closest valid nodes
                remaining = n - len(selected_nodes)
                for _ in range(remaining):
                    for node in all_nodes:
                        if node not in selected_nodes:
                            selected_nodes.append(node)
                            break

            logger.info(f"Node selection completed in {time.time() - start_time:.1f}s")
            return selected_nodes[:n]  # Ensure we return exactly n nodes
        except Exception as e:
            logger.error(f"Error selecting random nodes: {str(e)}")
            raise

    def get_shortest_path(self, start_node: int, end_node: int) -> Tuple[float, List[Tuple[float, float]]]:
        """Get shortest path and distance between two nodes."""
        try:
            path = nx.shortest_path(self.G, start_node, end_node, weight='length')
            distance = nx.shortest_path_length(self.G, start_node, end_node, weight='length')

            # Get coordinates for the path
            path_coords = []
            for node in path:
                coords = (self.node_positions.loc[node, 'geometry'].y,
                         self.node_positions.loc[node, 'geometry'].x)
                path_coords.append(coords)

            return distance, path_coords
        except Exception as e:
            logger.error(f"Error calculating shortest path: {str(e)}")
            raise

    def create_folium_map(self, routes: List[List[int]], save_path: str = "street_map.html"):
        """Create an interactive map visualization of the routes."""
        try:
            start_time = time.time()
            logger.info("Creating interactive map visualization")

            # Get center point of the network
            center_point = self.node_positions.unary_union.centroid
            m = folium.Map(location=[center_point.y, center_point.x], 
                         zoom_start=13,
                         tiles='cartodbpositron')

            # Create a color for each route
            colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen']

            total_paths = sum(len(route)-1 for route in routes)
            paths_plotted = 0

            # Plot each route
            for route_idx, route in enumerate(routes):
                color = colors[route_idx % len(colors)]
                logger.info(f"Plotting route {route_idx+1}/{len(routes)}")

                # Plot path between each consecutive pair of nodes
                for i in range(len(route)-1):
                    start = route[i]
                    end = route[i+1]

                    try:
                        _, path_coords = self.get_shortest_path(start, end)

                        # Create a line for the path
                        folium.PolyLine(
                            locations=path_coords,
                            weight=4,
                            color=color,
                            opacity=0.8
                        ).add_to(m)

                        # Add markers for start and end points
                        start_coords = (self.node_positions.loc[start, 'geometry'].y,
                                      self.node_positions.loc[start, 'geometry'].x)
                        end_coords = (self.node_positions.loc[end, 'geometry'].y,
                                    self.node_positions.loc[end, 'geometry'].x)

                        # Special marker for depot (first node of first route)
                        if route_idx == 0 and i == 0:
                            folium.Marker(
                                start_coords,
                                popup='Depot',
                                icon=folium.Icon(color='red', icon='info-sign')
                            ).add_to(m)
                        else:
                            folium.CircleMarker(
                                start_coords,
                                radius=6,
                                color=color,
                                fill=True
                            ).add_to(m)

                        folium.CircleMarker(
                            end_coords,
                            radius=6,
                            color=color,
                            fill=True
                        ).add_to(m)

                        paths_plotted += 1
                        logger.debug(f"Plotted path {paths_plotted}/{total_paths}")

                    except Exception as e:
                        logger.warning(f"Could not plot path in route {route_idx}: {str(e)}")
                        continue

            # Save the map
            m.save(save_path)
            logger.info(f"Interactive map saved to {save_path} (time: {time.time() - start_time:.1f}s)")

        except Exception as e:
            logger.error(f"Error creating map visualization: {str(e)}")
            raise

    def get_distance_matrix(self, nodes: List[int]) -> np.ndarray:
        """Create distance matrix for the selected nodes."""
        try:
            start_time = time.time()
            logger.info(f"Creating distance matrix for {len(nodes)} nodes")

            n = len(nodes)
            distance_matrix = np.zeros((n, n))
            paths_calculated = 0
            total_paths = n * (n-1)

            for i in range(n):
                for j in range(n):
                    if i != j:
                        try:
                            distance_matrix[i,j] = nx.shortest_path_length(
                                self.G, nodes[i], nodes[j], weight='length')
                            paths_calculated += 1
                            if paths_calculated % 10 == 0:
                                logger.debug(f"Calculated {paths_calculated}/{total_paths} paths")
                        except nx.NetworkXNoPath:
                            # Use great circle distance as fallback
                            coord1 = (self.node_positions.loc[nodes[i], 'geometry'].y,
                                    self.node_positions.loc[nodes[i], 'geometry'].x)
                            coord2 = (self.node_positions.loc[nodes[j], 'geometry'].y,
                                    self.node_positions.loc[nodes[j], 'geometry'].x)
                            distance_matrix[i,j] = ox.distance.great_circle_vec(coord1[0], coord1[1],
                                                                          coord2[0], coord2[1])
                            logger.warning(f"No path between nodes {nodes[i]} and {nodes[j]}, using great circle distance")

            logger.info(f"Distance matrix created in {time.time() - start_time:.1f}s")
            return distance_matrix
        except Exception as e:
            logger.error(f"Error creating distance matrix: {str(e)}")
            raise