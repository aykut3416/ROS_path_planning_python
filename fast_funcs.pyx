cimport numpy as np
import numpy as npy
cdef extern from "math.h":
    cpdef double sin(double x)
    cpdef double cos(double x)
    cpdef double atan2(double y, double x)
    cpdef double tan(double x)
    cpdef double sqrt(double x)
DEF PI = 3.14159265
cimport cython



def create_ranges(list p1,list p2,int N):
    cdef float dx,dy,dz
    cdef list result
    cdef int dim
    result=[]
    dim=len(p1)
    if dim==1:
        dx=(p2[0]-p1[0])/<double>N
        for i in range(0,N+1):
            result.append(p1[0]+i*dx)
        return result
    elif dim==2:
        dx=(p2[0]-p1[0])/<double>N
        dy=(p2[1]-p1[1])/<double>N
        for i in range(1,N+1):
            result.append((p1[0]+i*dx,p1[1]+i*dy))
    elif dim==3:
        dx=(p2[0]-p1[0])/<double>N
        dy=(p2[1]-p1[1])/<double>N
        dz=(p2[2]-p1[2])/<double>N
        for i in range(1,N+1):
            result.append((p1[0]+i*dx,p1[1]+i*dy,p1[2]+i*dz))
    return result

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def radius_neighbors(list a1, list target_a, float search_radi):
    cdef float dist
    cdef int target_size = len(target_a)
    cdef int i
    cdef list index_list=[]
    for i in range(target_size):
        dist=sqrt((target_a[i][0]-a1[0])**2+(target_a[i][1]-a1[1])**2)
        if dist<search_radi and dist>0.01:
            index_list.append((i,dist))
    return index_list

def radius_neighbors_inlist(list a1, list target_a, float search_radi, list search_list):
    cdef float dist
    cdef int i
    cdef list index_list=[]
    for i in search_list:
        dist=sqrt((target_a[i][0]-a1[0])**2+(target_a[i][1]-a1[1])**2)
        if dist<search_radi and dist>0.02:
            index_list.append(i)
    return index_list

def knn_neighbours(list a1, np.ndarray[np.float64_t, ndim=2] targeta, int k):
    cdef np.ndarray[np.float64_t, ndim=1] cx, cy, dists
    cx = targeta[:, 0] - a1[0]
    cy = targeta[:, 1] - a1[1]
    dists = npy.hypot(cx, cy)
    return dists.argsort()[1:k+1]

def nearest_neighbour(list a1, list a2):
    cdef float min_dist,dist
    cdef int i,min_ind,l_size
    l_size=len(a2)
    min_dist=1000.0
    for i in range(l_size):
        dist=sqrt((a1[0] - a2[i][0])**2+(a1[1] - a2[i][1])**2)
        if dist<min_dist and dist>0.01:
            min_dist=dist
            min_ind=i
    return min_ind,min_dist

