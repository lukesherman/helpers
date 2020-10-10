#!/usr/bin/env python
# coding: utf-8

# **Process**
#    - Get OSM road data for country
#    - Get sampled landmass grid.
#    - Limit grid to points contained in country shapefile
#    - Calculate total road length in each grid cell
#    
# **Questions**
#    - Eventually: Do we want a function that plots the tiles and roads? I know John mentioned that last time....
# 

# In[1]:


#%env OMP_NUM_THREADS=8


# In[2]:


import numpy as np
import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString, Point, MultiPolygon, Polygon, MultiLineString
from shapely.ops import prep
import time

import requests
import json

from haversine import haversine, haversine_vector

import matplotlib.pyplot as plt


# # Step 0: Create function to log job progress

# In[3]:


def log_text(text, mode="a", print_text=True):
    """
    Function that writes text to a log file. Takes text and mode as inputs. Default mode is "a" for append. Can also accept "w" which will clear the log.
    """
    if print_text:
        print(text)
    f = open("log.txt", mode)
    f.write(("\n" + text))
    f.close()


# # Step 1: Create functions to download and process Open Street Map (OSM) data
# 

# In[4]:


def get_osm_road_data_in_tile(grid_lat,grid_lon, delta=.01):
    """
    Overpass API query written to download all road data for a specific tile. It takes the tile centroid as an input. 
    Note that country specifications are often in foreign typecase.
    Returns a json file.
    
    """
    
    bottom_bound = grid_lat-delta/2
    left_bound = grid_lon-delta/2
    top_bound = grid_lat+delta/2
    right_bound = grid_lon + delta/2
    
    bounding_box = str((bottom_bound,left_bound,top_bound,right_bound))

    
    overpass_url = "https://overpass.kumi.systems/api/interpreter" # The kumi interpreter is not supposed to rate limit
    overpass_query = """
    /*
    This shows the roads in the country selected country code. Try Qatar with the input "QA".
    */

    [out:json][timeout:1600];
    way<box_string>["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|service|pedestrian|track|path|road"];
    out ids geom;

    """
    bottom_bound = grid_lat - delta/2
    left_bound = grid_lon - delta/2
    top_bound = grid_lat + delta/2
    right_bound = grid_lon + delta/2 
    overpass_query = overpass_query.replace("<box_string>", bounding_box)

    response = requests.get(overpass_url, 
                            params={'data': overpass_query})
    
    
    
    try:
        data = response.json()
    except:
        log_text(response.text)
        return "API_request_fail"
    
    if data.get("remark"): #If the data download has a remark, usually it means runtime error. stop and make note in the log
        log_text("API request encountered runtime error.")
        log_text(data.get("remark"))
        return "API_request_fail"
    
    return data


# In[5]:


#data = get_osm_road_data_in_tile(37.865,-122.275)


# The query above works.
# 
#    - we may want to refine the seach in the future so as to further limit the query to certain types of road. See [OSM highway documentation](https://wiki.openstreetmap.org/wiki/Key:highway). 
# 

# In[6]:


def way_to_line(way):
    """
    Takes a way from OSM with lats and lons as an input.
    Returns a shapely line object.
    """
    coords = []
    for i in range(len(way["geometry"])):
        coord = (way.get("geometry")[i]["lon"], way.get("geometry")[i]["lat"])
        coords.append(coord)
    line = LineString(coords)
    return line


# In[7]:


def overpass_query_to_df(data):
    """
    Takes a raw query from OSM as an input.
    Returns a geopandas style dataframe.
    """
    lines = []
    ids = []
    for i in range((len(data.get("elements")))):
        way = data.get("elements")[i]
        road_id = way["id"]
        line = way_to_line(way)
        lines.append(line)
        ids.append(road_id)
    df = pd.DataFrame([ids,lines]).T.rename(columns = {0 : "way_id", 1: "geometry"})
    return df


# In[8]:


# roads = overpass_query_to_df(data)
# roads = gpd.GeoDataFrame(roads)
# roads.head(10)


# We need to divide by area also, but that's probably best to do in a vectorized way at the end.

# # Step 2: Retrieve grid and country shape files

# We use the country shape files to further filter the grid so that we only look at grid points in the country of interest.

# In[9]:


# country_code = "QA"


# In[10]:


# grid_path = "/shares/maps100/data/output/grid/LandmassIntermediateResSparseGrid10.csv"
# grid = pd.read_csv(grid_path)


# In[11]:


# country_path = "/shares/maps100/data/raw/country_bounds/ne_10m_admin_0_countries.shp"
# countries = gpd.read_file(country_path)


# In[12]:


#country_gpd = countries[countries["ISO_A2"] == country_code] # Country code (2 character) is set at the beginning.


# In[13]:


# Limit grid based on bounding box 
# This will get rid of most of the points and make the function call below much faster

# left_bound, bottom_bound, right_bound, top_bound = country_gpd.total_bounds
# indices = (grid["lon"] >= left_bound) & (grid["lon"] <= right_bound)  & (grid["lat"] <= top_bound) & (grid["lat"] >= bottom_bound)
# box_grid = grid.loc[indices]


# In[14]:


def country_shape_grid(df, gpdFile):
    """
    Takes a grid df with lats and lons. Checks for point intersections with the attached geopandas file.
    
    Returns a new pandas df.
    
    Needs shapely, geopandas, and numpy.
    """
    pd.options.mode.chained_assignment = None  # default='warn' 
    #Hiding an annoying warning. Should probably fix
    
    gpdFile["preped"] = gpdFile["geometry"].apply(prep) # prepare the geometry to improve speed
    
    lats = df["lat"].values  #Making two arrays that together correspond to all of the grid points
    lons = df["lon"].values
    
    points = [Point((lons[i], lats[i])) for i in range(len(lats))] # turn each point into a Shapely object
    
    for i in gpdFile.index:
        #print(i)
        prepared_polygon = gpdFile["preped"].loc[i]

        intersect_points = list(filter(prepared_polygon.contains, points))

        if i == gpdFile.index[0]:
            hits = intersect_points
        else:
            hits = hits + intersect_points

    output_lons = []
    output_lats = []

    for i in range(len(hits)):
        output_lons.append(hits[i].x)
        output_lats.append(hits[i].y)

    outputGrid = {    #Note that this output will be the full length 'flat' grid as json file. 
        "lat" : output_lats,
        "lon" : output_lons,
        }
    
    return pd.DataFrame(outputGrid).sort_values(["lat","lon"]) #currently the output is not ordered. This improved runtime


# In[15]:


#country_grid = country_shape_grid(box_grid,country_gpd)


# # Step 3: Calculate total length of road in cell

# Turning lines into km lengths

# In[16]:


def linestring_to_km_length(linestring):
    """
    Takes a linestring object and returns the length of that object in km. Depends on the haversine package which applies the haversine formula.
    """
    points = np.array(linestring.xy).T # Turn linestring into an np array of points
    
    points_1 = points[:-1] #First set of points drops the last value
    points_2 = points[1:] # Second set of points drops the first value. Now the array is one shorter and points are offset
    
    return np.sum(haversine_vector(points_1, points_2))


# Now build a function that efficiently checks for intersections for a given tile:

# In[17]:


def length_of_road_in_tile(roads_gpd, tile):
    """
    For a given tile and roads dataset, this function calculates the length of road in that cell. 
    The output is currently in degres, but probably should be in kilometers.
    """
    length = 0
    prepped_tile = prep(tile)
    
    for i in range(len(roads_gpd)):
        if prepped_tile.intersects(roads_gpd["geometry"][i]):
            line_intersection = tile.intersection(roads_gpd["geometry"][i])
            
            if isinstance(line_intersection,LineString):
                length += linestring_to_km_length(line_intersection)
                
            elif isinstance(line_intersection,MultiLineString): # Sometimes cutting up a road actually returns more than one line. This deals with that scenario
                for j in range(len(line_intersection)):
                    #print("Multi line at=",j)
                    length += linestring_to_km_length(line_intersection[j])
            else:
                raise Exception
            
    return length


# In[18]:


def calculate_road_length_for_subgrid(grid, delta = .01):
    """
    Takes in grid specifications and creates tiles. For each tile, this function calls "length_of_road_in_cell."
    This function stores the length of road in the cell as well as the corresponding lat and lon of the tile centroid.
    """
    
    grid["road_length_km"] = np.nan
    log_text("Subgrid has {} tiles".format(len(grid)))
    fail_counter = 0
    
    for i in grid.index:
        if i % 100 == 0:
            log_text("Processing grid tile " + str(i) + " out of " + str(len(grid)))

        data = get_osm_road_data_in_tile(grid.loc[i,"lat"],grid.loc[i,"lon"], delta=delta)  #Download roads data for each coordinate point
            
        if data == "API_request_fail":
            fail_counter +=1
            log_text("Network request failed. Total fails=".format(fail_counter))
            grid.loc[i, "road_length_km"] = data
                
        else:
            roads = overpass_query_to_df(data)
            roads_gpd = gpd.GeoDataFrame(roads)
        
            tile = (Point(grid["lon"][i],grid["lat"][i]).buffer(delta/2, cap_style=3))
        
            grid.loc[i, "road_length_km"] = length_of_road_in_tile(roads_gpd, tile)
    
    return grid


# In[19]:


#df = calculate_road_length_for_subgrid(country_grid)


# In[20]:


# df.head(5) 


# In[21]:


# output_path = "/shares/maps100/data/output/applications/road_length/country:{}_road_length.csv".format(country_code)

# df.to_csv(output_path)


# # Step 4: Create single function that takes only a country as an input

# In[1]:


def create_subgrid_for_country_and_write_road_length(country_code, delta = .01):
    
    """ 
    Takes a two character country code as an input and creates a subgrid for the country we want.
    """
    
    grid_path = "/shares/maps100/data/output/grid/LandmassIntermediateResSparseGrid10.csv"
    grid = pd.read_csv(grid_path)
    
    country_path = "/shares/maps100/data/raw/country_bounds/ne_10m_admin_0_countries.shp"
    countries = gpd.read_file(country_path)
    country_gpd = countries[countries["ISO_A2"] == country_code] # Get the shape file for the country we're looking at
    
    # This will get rid of most of the points in the landmass grid and make the country_shape_grid function call below much faster
    left_bound, bottom_bound, right_bound, top_bound = country_gpd.total_bounds

    indices = (grid["lon"] >= left_bound) & (grid["lon"] <= right_bound)  & (grid["lat"] <= top_bound) & (grid["lat"] >= bottom_bound)
    box_grid = grid.loc[indices]
    
    country_grid = country_shape_grid(box_grid,country_gpd)
    
    df = calculate_road_length_for_subgrid(country_grid, delta=delta)
    
    df["country_code"] = country_code 
    
    output_path = "/shares/maps100/data/output/applications/road_length/country:{}_road_length.csv".format(country_code)

    df.to_csv(output_path, index=False)
    
    log_text("Processing road length for country_code={} ({}) is complete!".format(country_code,country_gpd["NAME_EN"].values[0]))
    
    fail_count = sum(df["road_length_km"] == "API_request_fail")
    
    log_text("{} coordinate points out of {} were not calculated due to failed API requests".format(fail_count,len(country_grid)))


# In[2]:


create_subgrid_for_country_and_write_road_length("AR")


# In[ ]:


#pd.read_csv("/shares/maps100/data/output/applications/road_length/country:QA_road_length.csv")


# ## Step 5: Run function on a loop for all countries

# In[ ]:


# country_path = "/shares/maps100/data/raw/country_bounds/ne_10m_admin_0_countries.shp"
# countries = gpd.read_file(country_path)
# countries = countries[countries["ISO_A2"] != "-99"] # Drop the null countries

# country_code_list = list(countries["ISO_A2"])


# log_text("START","w")
# for code in country_code_list:
#     log_text("Process beginning for country_code={}.".format(code))
#     create_subgrid_for_country_and_write_road_length(code)


# ***

# **Now just a little bit of visualization to check on my work in the above....**
# 
# These visualizations were built to look at Qatar, but can be adjusted to look at any country.
# ***
# 

# In[ ]:


# fig, ax = plt.subplots(figsize=(8,8))
# ax.scatter(box_grid["lon"], box_grid["lat"], label = "Tiles removed by the shape filter")

# country_gpd["geometry"].exterior.plot(ax = ax, label = "Country Boundary")

# ax.scatter(box_grid["lon"], box_grid["lat"], label = "Tiles removed by the shape filter")

# ax.scatter(country_grid["lon"], country_grid["lat"], label =  "Final tile centroids")

# ax.set_xlim((50,52))
# ax.legend()


# In[ ]:


# indices = (grid["lon"] >= 50) & (grid["lon"] <= 52)  & (grid["lat"] <= 26.2) & (grid["lat"] >= 24.4)
# box_grid = grid.loc[indices]

# fig, ax = plt.subplots(figsize=(8,8))
# ax.scatter(box_grid["lon"], box_grid["lat"], label = "Tiles removed by the shape filter")

# country_gpd["geometry"].exterior.plot(ax = ax, label = "Country Boundary")

# ax.scatter(box_grid["lon"], box_grid["lat"], label = "Landmass tiles removed by the shape filter")

# ax.scatter(country_grid["lon"], country_grid["lat"], label =  "Final tile centroids")

# ax.set_xlim((50,52))
# ax.legend()


# In[ ]:


# roads.plot()


# In[ ]:





# In[ ]:


#simple plot of roads and tile

# fig, ax = plt.subplots(1, figsize = (8,8))
# plt.plot(np.array(list(zip(tile.exterior.xy[0],tile.exterior.xy[1])))[:,0], np.array(list(zip(tile.exterior.xy[0],tile.exterior.xy[1])))[:,1], c="r")
# roads.plot(ax=ax)



# ## Unused functions:

# Unfortunately, the function below always times out when attempting to download roads for large countries. Instead, we switched to getting the roads on a tile by tile basis.
def get_osm_road_data(iso2c):
    """
    Overpass API query written to download all road data for a specific country. 
    Note that country specifications are often in foreign typecase.
    Returns a json file.
    """

    overpass_url = "https://overpass.kumi.systems/api/interpreter" # The kumi interpreter is not supposed to rate limit
    overpass_query = """
    /*
    This shows the roads in the country selected country code. Try Qatar with the input "QA".
    */

    [out:json][timeout:1600];
    area["ISO3166-1"="<iso_code>"];
     (way["highway"~"motorway|trunk|primary|secondary|tertiary|unclassified|residential|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link|living_street|service|pedestrian|track|path|road"](area);
    );
    out ids geom;

    """

    overpass_query = overpass_query.replace("<iso_code>", iso2c)
    
    response = requests.get(overpass_url, 
                            params={'data': overpass_query})
    
    
    try:
        data = response.json()
    except:
        log_text(response.text)
        return False
    
    return data# country_code = "QA" #Takes a two character country code as an input
# data = get_osm_road_data(country_code