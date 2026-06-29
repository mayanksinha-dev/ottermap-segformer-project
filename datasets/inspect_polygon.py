import geopandas as gpd

gdf = gpd.read_file("data/raw/labels/1.geojson")

print(gdf.geometry.iloc[0])
print("Has interiors:", len(gdf.geometry.iloc[0].interiors))