import numpy          as np
import geopy.distance as geodist
PI  = np.pi
NaN = np.nan


###############################################################################################################
def geocircle(lon0, lat0, radius, number_of_points = 64): 
    # lon0  : longitude of the circle origin
    # lat0  : latitude  of the circle origin
    # radius: radius in km
   
    # Output
    lon_out = np.empty([ number_of_points + 1 ])
    lat_out = np.empty([ number_of_points + 1 ])


    # Parameters    
    dr0 = 0.1      # Initial value for tyhe incremantation
    dx  = 0.001  # When |Radius - distance| < dx, the solution is "found"

    coord0 = (lat0, lon0)

    angle_increment = 2*PI / number_of_points
    
    r = 0
    i = 0
    for angle in np.arange(0, 2*PI, angle_increment):

        dr        = dr0
        direction = +1;
        distance  =  0

        while np.abs(radius - distance) > dx:
            lon      = lon0 + r * np.cos(angle)
            lat      = lat0 + r * np.sin(angle)
            coord    = (lat, lon)
            if(lat>90):
                #latx = 180-lat
                #lonx = lon+180
                #coordx    = (latx, lonx)
                #distance = geodist.distance(coord0, coordx).km 
                lat = NaN
                lon = NaN
                distance = radius
            else:
                distance = geodist.distance(coord0, coord).km

            if distance > radius and direction > 0:
                direction = -1 
                dr = dr / 10
            elif distance < radius and direction < 0:
                direction = +1
                dr = dr /10
            #elif distance == radius:
            #    dr = drmin / 10
             
            r = r + direction * dr 

        lon_out[i] = lon    
        lat_out[i] = lat
        i = i + 1
    lon_out[i] = lon_out[0]
    lat_out[i] = lat_out[0]
    return lon_out, lat_out



###############################################################################################################
def check_if_location_is_inside_radius(lon1, lat1, lon2, lat2, R):
    N = len(lon1)
    location_inside_radius = np.zeros(N)
    for n in range(N):
        distance = geodist.distance( (lat1[n],lon1[n]), (lat2[n],lon2[n]) ).km
        if distance < R: location_inside_radius[n] = 1

    return location_inside_radius



############################################################################################
from   shapely.geometry         import Point
from   shapely.geometry.polygon import Polygon
def point_inside_polygon(points, polygon):
    polygon = Polygon(polygon)
    N       = len(points)
    flag    = np.zeros(N)
    for n in range(N):
        point   = Point(points[n])
        if polygon.contains(point):
            flag[n] = 1
    return flag


###########################################################################################
def construct_coord(x,y):
    coord = []
    N     = len(x)
    for n in range(N):
        coord.append((x[n],y[n]))
    return coord

