import osmnx as ox
import networkx as nx
import folium
import numpy as np
from typing import List, Tuple, Dict, Any
import logging
import time
from shapely.geometry import box
import random
import math

logger = logging.getLogger(__name__)

class StreetNetwork:
    """Handle real street network data from OpenStreetMap."""

    def __init__(self, place_name: str = "San Francisco, California, USA"):
        """Initialize street network for a given location."""
        try:
            logger.info(f"Fetching street network for {place_name}")
            start_time = time.time()

            # Try different methods to fetch the network
            try:
                # First attempt: Try using a specific address in San Francisco
                if "San Francisco" in place_name:
                    address = "Union Square, San Francisco, California, USA"
                    logger.info(f"Attempting to fetch network using address: {address}")
                    self.G = ox.graph_from_address(address, dist=1000, network_type='drive')
                    logger.info("Successfully fetched network using address")
                else:
                    # For other locations, try place name directly
                    logger.info(f"Attempting to fetch network using place name: {place_name}")
                    self.G = ox.graph_from_place(place_name, network_type='drive')
                    logger.info("Successfully fetched network using place name")
            except Exception as e:
                logger.warning(f"Initial fetch attempt failed: {str(e)}")
                try:
                    # Final fallback: Use a bounding box for a small area
                    logger.info("Falling back to bounding box method")
                    # For San Francisco downtown area
                    center_lat, center_lon = 37.7749, -122.4194
                    dist = 1000  # meters
                    north, south, east, west = ox.utils_geo.bbox_from_point((center_lat, center_lon), dist=dist)
                    logger.info(f"Using bounding box: N={north}, S={south}, E={east}, W={west}")

                    self.G = ox.graph_from_bbox(north, south, east, west, network_type='drive')
                    logger.info("Successfully fetched network using bounding box")
                except Exception as bbox_error:
                    logger.error(f"All fetch attempts failed. Last error: {str(bbox_error)}")
                    raise RuntimeError("Could not fetch street network using any method")

            # Convert to non-directional graph
            self.G = self.G.to_undirected()
            logger.info("Converted to undirected graph")

            # Project the graph to use meters for distance calculations
            self.G = ox.project_graph(self.G)
            logger.info("Projected graph to use meters for distance calculations")

            # Get node positions and convert to lat/long
            self.node_positions = ox.graph_to_gdfs(self.G, edges=False)
            # Convert back to lat/long coordinates (EPSG:4326)
            self.node_positions = self.node_positions.to_crs(epsg=4326)
            logger.info("Retrieved and converted node positions to lat/long format")

            logger.info(f"Network initialization completed in {time.time() - start_time:.1f}s")

        except Exception as e:
            logger.error(f"Error initializing street network: {str(e)}")
            raise RuntimeError(f"Failed to initialize street network: {str(e)}")

    def calculate_haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance between two points in meters."""
        R = 6371000  # Earth's radius in meters

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def get_random_nodes(self, n: int, min_distance: float = 200) -> List[int]:
        """Get n random nodes that are at least min_distance meters apart."""
        try:
            start_time = time.time()
            logger.info(f"Selecting {n} random nodes with minimum distance of {min_distance}m")

            # Get center node for depot
            center_point = self.node_positions.unary_union.centroid
            depot_node = ox.nearest_nodes(self.G, center_point.x, center_point.y)
            selected_nodes = [depot_node]
            logger.info(f"Selected depot node: {depot_node}")

            # Get all nodes from the graph
            all_nodes = list(self.G.nodes())
            logger.info(f"Found {len(all_nodes)} nodes in graph")

            # Create a set of used nodes for faster lookup
            used_nodes = {depot_node}

            attempts = 0
            max_attempts = 1000
            min_distance_current = min_distance

            while len(selected_nodes) < n and attempts < max_attempts:
                remaining_nodes = [node for node in all_nodes if node not in used_nodes]
                if not remaining_nodes:
                    break

                node = random.choice(remaining_nodes)
                valid = True

                # Check distance to all selected nodes
                for selected in selected_nodes:
                    try:
                        # Try network distance first
                        try:
                            path_length = nx.shortest_path_length(
                                self.G, node, selected, weight='length')
                            if path_length < min_distance_current:
                                valid = False
                                break
                        except nx.NetworkXNoPath:
                            # If no path exists, use haversine distance as fallback
                            node_coord = self.get_node_coordinates([node])[0]
                            selected_coord = self.get_node_coordinates([selected])[0]
                            dist = self.calculate_haversine_distance(
                                node_coord[0], node_coord[1],
                                selected_coord[0], selected_coord[1]
                            )
                            if dist < min_distance_current:
                                valid = False
                                break
                    except Exception as e:
                        logger.warning(f"Error calculating distance: {str(e)}")
                        valid = False
                        break

                if valid:
                    selected_nodes.append(node)
                    used_nodes.add(node)
                    logger.info(f"Selected node {len(selected_nodes)}/{n} at distance {min_distance_current:.1f}m")
                    min_distance_current = min_distance  # Reset distance for next node

                attempts += 1
                if attempts % 100 == 0:
                    min_distance_current *= 0.8
                    logger.info(f"Reducing minimum distance to {min_distance_current:.1f}m")

            if len(selected_nodes) < n:
                logger.warning(f"Could only find {len(selected_nodes)} nodes at desired distances")
                # Fill remaining slots with closest available nodes
                remaining_needed = n - len(selected_nodes)
                available_nodes = [node for node in all_nodes if node not in used_nodes]

                if available_nodes:
                    # Sort by distance to center_point
                    center_coord = (center_point.y, center_point.x)
                    available_nodes.sort(key=lambda node:
                        self.calculate_haversine_distance(
                            self.node_positions.loc[node, 'geometry'].y,
                            self.node_positions.loc[node, 'geometry'].x,
                            center_coord[0], center_coord[1]
                        )
                    )

                    selected_nodes.extend(available_nodes[:remaining_needed])
                    logger.info(f"Added {remaining_needed} additional nodes to complete selection")

            logger.info(f"Node selection completed in {time.time() - start_time:.1f}s")
            return selected_nodes[:n]

        except Exception as e:
            logger.error(f"Error selecting random nodes: {str(e)}")
            raise

    def get_node_coordinates(self, nodes: List[int]) -> List[Tuple[float, float]]:
        """Get latitude/longitude coordinates for a list of nodes."""
        try:
            coordinates = []
            for node in nodes:
                # Get coordinates in lat/long format (already in EPSG:4326)
                coords = (
                    self.node_positions.loc[node, 'geometry'].y,  # latitude
                    self.node_positions.loc[node, 'geometry'].x   # longitude
                )
                logger.debug(f"Node {node} coordinates (lat, long): {coords}")
                coordinates.append(coords)
            return coordinates
        except Exception as e:
            logger.error(f"Error getting node coordinates: {str(e)}")
            raise

    def get_distance_matrix(self, nodes: List[int]) -> np.ndarray:
        """Create distance matrix for the selected nodes."""
        try:
            start_time = time.time()
            logger.info(f"Creating distance matrix for {len(nodes)} nodes")

            n = len(nodes)
            distance_matrix = np.zeros((n, n))

            for i in range(n):
                for j in range(n):
                    if i != j:
                        try:
                            distance_matrix[i,j] = nx.shortest_path_length(
                                self.G, nodes[i], nodes[j], weight='length')
                        except nx.NetworkXNoPath:
                            # Use haversine distance as fallback
                            coord1 = self.get_node_coordinates([nodes[i]])[0]
                            coord2 = self.get_node_coordinates([nodes[j]])[0]
                            distance_matrix[i,j] = self.calculate_haversine_distance(
                                coord1[0], coord1[1], coord2[0], coord2[1]
                            )
                            logger.warning(f"No path between nodes {nodes[i]} and {nodes[j]}, using haversine distance")

            logger.info(f"Distance matrix created in {time.time() - start_time:.1f}s")
            return distance_matrix
        except Exception as e:
            logger.error(f"Error creating distance matrix: {str(e)}")
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

            # Plot each route
            for route_idx, route in enumerate(routes):
                color = colors[route_idx % len(colors)]
                logger.info(f"Plotting route {route_idx+1}/{len(routes)}")

                # Plot path between each consecutive pair of nodes
                for i in range(len(route)-1):
                    start = route[i]
                    end = route[i+1]

                    try:
                        path = nx.shortest_path(self.G, start, end, weight='length')
                        path_coords = self.get_node_coordinates(path)

                        # Create a line for the path
                        folium.PolyLine(
                            locations=path_coords,
                            weight=4,
                            color=color,
                            opacity=0.8
                        ).add_to(m)

                        # Add markers for start and end points
                        start_coords = path_coords[0]
                        end_coords = path_coords[-1]

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

                    except Exception as e:
                        logger.warning(f"Could not plot path in route {route_idx}: {str(e)}")
                        continue

            # Save the map
            m.save(save_path)
            logger.info(f"Interactive map saved to {save_path} (time: {time.time() - start_time:.1f}s)")

        except Exception as e:
            logger.error(f"Error creating map visualization: {str(e)}")
            raise