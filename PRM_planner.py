# Copyright (c) 2023 Aykut ÖZDEMİR
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped
import random
import math as m
from nav_msgs.msg import OccupancyGrid,Odometry
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point,PoseStamped
import time
import pyximport
import heapq
from tf.transformations import euler_from_quaternion
pyximport.install(setup_args={"script_args":["--compiler=unix"],
                              "include_dirs":np.get_include()},
                  reload_support=True)
import fast_funcs # Import cython library
# Global variables
pose=[0,0,0] # Robot pose global variable
tree=None # Graph variable
path=None # Path variable 

#Simulation parameters
n_samples=1000 
n_neighbors=8
map_topic='/move_base/global_costmap/costmap'
odom_topic= "/vesc/odom" # Should be published in Odometry type

def create_marker(m_type,frame,id_m,rm,gm,bm,sx,sy,sz): # Marker initialization procedure for visualization purposes
    mark = Marker()
    mark.header.frame_id=frame
    mark.header.stamp = rospy.Time.now()
    mark.ns = "markers"
    mark.id=id_m
    mark.type=m_type
    mark.action = Marker.ADD
    mark.color.r=rm
    mark.color.g=gm
    mark.color.b=bm
    mark.color.a=1.
    mark.scale.x= sx
    mark.scale.y = sy
    mark.scale.z = sz
    return mark


class CollisionChecker:
    # This class subscribes the map topic
    def __init__(self):
        self.map = None
        self.origin = None
        self.resolution = None
        self.map_sub = rospy.Subscriber(map_topic, OccupancyGrid, self.map_callback)
        rospy.loginfo('Collision checker initialized')

    def map_callback(self, msg):
        self.map = np.array(msg.data).reshape(msg.info.height, msg.info.width)
        self.origin = msg.info.origin
        self.resolution = msg.info.resolution

    def check_collision(self, pose): # This function returns whether pose is in collision with obstacles or not

        x = pose[0]
        y = pose[1]
        row = int((y - self.origin.position.y) / self.resolution)
        col = int((x - self.origin.position.x) / self.resolution)

        if row < 0 or row >= self.map.shape[0] or col < 0 or col >= self.map.shape[1]: # Map limit check
            return True

        if self.map[row, col] >= 50 or self.map[row, col]==-1: # If area is unexplored or occupied 
            return True

        return False

class prm_planner:
    def __init__(self,start_p,goal_p,map_,n_samples,n_neighbors):
        # Initialize PRM variables
        self.n_samples=n_samples
        self.start=start_p
        self.goal=goal_p
        self.grid=map_
        self.n_neighbors=n_neighbors
        self.samples=[]
        self.neighbors = {}
    def plan(self):
        # Try to create n random samples in collision free areas 
        n_nodes=0
        while n_nodes<self.n_samples:
            node_x=self.grid.origin.position.x+random.random()*(self.grid.map.shape[0]*self.grid.resolution)
            node_y=self.grid.origin.position.y+random.random()*(self.grid.map.shape[1]*self.grid.resolution)
            if not self.grid.check_collision([node_x,node_y]):
                self.samples.append([node_x,node_y])
                n_nodes+=1
        
        sample_arr=np.array(self.samples).reshape(-1,2) #Array form of sample list

        for i in range(len(self.samples)): 
            self.neighbors[i]=[] # Each node will have many neighbor nodes initialize as an empty list
            neighs=fast_funcs.knn_neighbours(self.samples[i],sample_arr,self.n_neighbors) # Find k neighbour nodes of ith sample
            for neigh in neighs:
                dist=m.hypot(self.samples[i][0]-self.samples[neigh][0],self.samples[i][1]-self.samples[neigh][1]) 
                n_segments=max(2,int(dist/self.grid.resolution)) # 0.05 is map resolution
                all_segs=fast_funcs.create_ranges(self.samples[i],self.samples[neigh],n_segments) # Draw a line between the node and its neighbor in map resolution
                coll=False
                for pt in all_segs: # Check all segments whether obstacle free or not
                    if self.grid.check_collision(pt): 
                        coll=True
                        break
                if coll==False: # If all segments are safe append this valid edge to the neighbors dictionary
                    self.neighbors[i].append((neigh,dist))
        
        # Find the nearest node in the graph to the robot position
        dists=[m.hypot(sample[0]-self.start[0],sample[1]-self.start[1]) for sample in self.samples]
        start_ind=dists.index(min(dists))
        # Find the nearest node in the graph to the goal position
        dists=[m.hypot(sample[0]-self.goal[0],sample[1]-self.goal[1]) for sample in self.samples]
        goal_ind=dists.index(min(dists))

        # Perform an astar search on this graph to find a route
        path=astar(start_ind,goal_ind,self.neighbors,self.samples)
        line_list=[]
        for i in range(len(self.samples)): 
            for (n_ind, c_) in self.neighbors[i]:
                line_list.append(self.samples[n_ind]+self.samples[i])

        if type(path)!=type(None):
            route=[self.samples[path[i]] for i in range(len(path))]
            route.insert(0,self.start)   
            route.append(self.goal)      
            return line_list,route
        else:
            return [],[]

def astar(initial,goal,tree,data):
    # Initialization of A* variables
    openHeap=[]
    closeSet=set()
    openSet=set()
    g=[0]*len(data)
    h=[0]*len(data)
    parent=[-1]*len(data)
    path=[]
    parent[initial]= -1
    g[initial]=0
    h[initial]=m.sqrt((data[initial][0]-data[goal][0])**2+(data[initial][1]-data[goal][1])**2)
    openSet.add(initial)
    heapq.heappush(openHeap,(g[initial]+h[initial],initial))

    def retracePath(c): #Retrace function
        path.append(c)
        if parent[c] == -1:
            return
        retracePath(parent[c])


    while openSet:
        curr_l,curr_ind=heapq.heappop(openHeap) # Priority queue get first element 

        if curr_ind==goal: # If goal node on the graph equals to current index
            retracePath(curr_ind) # path variable created from end to root
            path.reverse() 
            return path

        openSet.discard(curr_ind) 
        closeSet.add(curr_ind) # Mark as visited

        for (adj,cost) in tree[curr_ind]: # Visit all neighbors of the current node
            g[adj]=cost+g[curr_ind] 
            h[adj]=m.sqrt((data[adj][0]-data[goal][0])**2+(data[adj][1]-data[goal][1])**2)
            f=g[adj]+h[adj] # Calculate heuristic function
            if (adj not in openSet) and (adj not in closeSet): # If it is not visited before add to openset and priority queue
                openSet.add(adj)
                heapq.heappush(openHeap,(f,adj))
                parent[adj]=curr_ind
            elif (adj in openSet): # If it is previously added to the queue, update its parent if shorter connection is found
                for (k,j) in openHeap:
                    if(j==adj):
                        if f<k:
                            parent[adj]=curr_ind
                            openHeap[openHeap.index((k,j))]=(f,j)
        
def goal_sub(msg): # Goal message subscriber
    global tree,path
    goalP=[msg.pose.position.x,msg.pose.position.y]
    print("Goal came")
    a=time.time()
    planner=prm_planner(pose,goalP,coll,n_samples=1000,n_neighbors=8) # Initialize the planner
    graph,plan=planner.plan() # Call the planning procedure
    if plan!=[]: # If plan is found update global variables for visualization 
        tree=graph
        path=plan
    else:
        print("Path not found!")
    print(f"Time elapsed : {time.time()-a}")
    


def pose_sub(msg): # Robot pose subscriber
    global pose
    quat = msg.pose.pose.orientation
    roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w]) # In order to convert quaternion to euler angles (not necessary)
    pose=[msg.pose.pose.position.x,msg.pose.pose.position.y,yaw]

if __name__ == "__main__":
    rospy.init_node('prm_planner', anonymous=True)
    coll=CollisionChecker() # Initialize map

    # Initialize markers to visualize in RViz
    pts1 = create_marker(Marker.LINE_LIST, "map", 2, 1., 1.0, 0., 0.05, 0.05, 0.05) 
    pts2 = create_marker(Marker.LINE_LIST, "map", 1, 0.5, 0.5, 0.5, 0.05, 0.05, 0.05)
    pub1 = rospy.Publisher('/visualization_marker', Marker, queue_size=1)

    # Subscribe goal and robot pose topics
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, goal_sub)
    rospy.Subscriber(odom_topic, Odometry, pose_sub)

    rate=rospy.Rate(100)
    while type(coll.origin)==type(None): # Wait until map is initialized
            pass
    
    while not rospy.is_shutdown():
        if type(path)!=type(None): # If plan is created 

            # Visualize graph
            pts2.points=[]
            for i in range(len(tree)):
                a=Point()
                a.x = tree[i][0]
                a.y = tree[i][1]
                a.z = 0.0
                pts2.points.append(a)
                a=Point()
                a.x = tree[i][2]
                a.y = tree[i][3]
                a.z = 0.0
                pts2.points.append(a)
            pub1.publish(pts2)
            
            # Visualize global plan
            pts1.points=[]
            for i in range(len(path)-1):
                a=Point()
                a.x = path[i][0]
                a.y = path[i][1]
                a.z = 0.1
                pts1.points.append(a)
                a=Point()
                a.x = path[i+1][0]
                a.y = path[i+1][1]
                a.z = 0.1
                pts1.points.append(a)
            pub1.publish(pts1)
        rate.sleep()
