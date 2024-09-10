import numpy                    as     np 
from   shapely.geometry         import Point
from   shapely.geometry.polygon import Polygon


############################################################################################
def nearest_point_on_a_line(p1, p2, p3, type='fullline'):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    dx = x2 - x1
    dy = y2 - y1
    d2 = dx*dx + dy*dy
    nx = ((x3-x1)*dx + (y3-y1)*dy) / d2
    if type == 'segment': 
        nx = min(1, max(0, nx))
    return (dx*nx + x1, dy*nx + y1)




############################################################################################
def point_inside_polygon(points, polygon):
    polygon = Polygon(polygon)
    N       = len(points)
    flag    = np.zeros(N)
    for n in range(N):
        point   = Point(points[n])
        if polygon.contains(point):
            flag[n] = 1
    return flag

