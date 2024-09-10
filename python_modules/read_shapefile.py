def rshapefile(shapefile_path):
    import numpy as np
    import shapefile
    import collections
    NaN=np.nan


    shapefile_directory="/chinook/poitras/shapefile";

    if shapefile_path == "LakeOntario": shapefile_path = shapefile_directory+"/Great_lakes/LakeOntario.shp"


    myshp  = open(shapefile_path, "rb")
   #mydbf  = open("/chinook/poitras/shapefile/LakeOntario/LakeOntario.dbf", "rb")
   #sf     = shapefile.Reader(shp=myshp, dbf=mydbf)
    sf     = shapefile.Reader(shp=myshp)
    shapes = sf.shapes()
    
    
    lon=[ NaN ];
    lat=[ NaN ];
    for i in range(len(shapes)):
        points = shapes[i].points
        points = operation_on_list_with_duplicated_items(ilist=points,  operation="add_after", starting_from=2, item_to_add=(NaN,NaN))
        lon    = [point[0] for point in points] + [ NaN ] + lon 
        lat    = [point[1] for point in points] + [ NaN ] + lat


    return(lon, lat)





##################################################################################################
def list_duplicates(seq):
    from collections import defaultdict
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)    
    return ((key,locs) for key,locs in tally.items() if len(locs)>1)




##################################################################################################
def operation_on_list_with_duplicated_items(ilist, operation, starting_from, item_to_add):
    olist=ilist.copy()
    if   operation == "add_after" : dindex =  1
    elif operation == "add_before": dindex =  0
    elif operation == "replace"   : dindex = -1
    elif operation == "remove"    : dindex = -2
    else                          : print("ERROR: operation = %s. Accepted entries for operation are add_after, add_before, replace, remove" % operation); return NaN; 
    dup_indices_to_keep = []
    for dup in sorted(list_duplicates(olist)):
       #dup_item            = dup[0]
        dup_indices         = dup[1]
        dup_indices_to_keep = dup_indices_to_keep + dup_indices[starting_from-1:len(dup_indices)] 
    dup_indices_to_keep.sort(reverse=True)


    for index in dup_indices_to_keep:
        if    dindex >=  0: olist.insert(index+dindex,item_to_add)
        elif  dindex == -1: olist[index] = item_to_add
        elif  dindex == -2: olist.pop(index)
    return olist
