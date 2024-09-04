import folium
from IPython.display import display

# Define the latitude and longitude
latitude = 37.7749
longitude = -122.4194

# Create a map centered around the latitude and longitude
world_map = folium.Map(location=[latitude, longitude], zoom_start=10)

# Add a marker to the map
folium.Marker([latitude, longitude], popup='Marker', tooltip='Click for more').add_to(world_map)

# Display the map
display(world_map)
