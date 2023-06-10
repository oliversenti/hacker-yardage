import osmnx as ox
import folium


# Retrieve the street network for a given location
G = ox.graph_from_place('Piedmont, California', network_type='drive')

# Get the coordinates for the start and end points of a road
start_point = (37.8245, -122.2326)
end_point = (37.8154, -122.2354)

# Find the nearest nodes on the street network to the start and end points
start_node = ox.distance.nearest_nodes(G, start_point[1], start_point[0])
end_node = ox.distance.nearest_nodes(G, end_point[1], end_point[0])

# Find the shortest path between the start and end nodes
route = ox.shortest_path(G, start_node, end_node)

# Create a folium map centered at the start point
m = folium.Map(location=start_point, zoom_start=14)

# Add the road route to the map as a polyline
route_coords = [(G.nodes[node]['y'], G.nodes[node]['x']) for node in route]
folium.PolyLine(locations=route_coords, color='red', weight=5).add_to(m)

# Save the map to an HTML file
m.save('map.html')

import osmnx as ox
import folium

# Define the location of interest
address = "San Francisco, California, USA"

# Obtain the latitude and longitude of the location using geocode
location = ox.geocode(address)

# Download the OSM data for the location
G = ox.graph_from_point(location, network_type="all")

# Create a folium map centered at the location
m = folium.Map(location=[location[0], location[1]], zoom_start=15)

# Plot the street network on the map
ox.plot_graph_folium(G, graph_map=m, edge_color="red", edge_width=2)

# Save the map to an HTML file
m.save("fairwaymap.html")

