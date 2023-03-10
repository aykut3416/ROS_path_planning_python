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
from tf.transformations import euler_from_quaternion
import heapq
from pqdict import pqdict

pyximport.install(setup_args={"script_args":["--compiler=unix"],
                              "include_dirs":np.get_include()},
                  reload_support=True)
import fast_funcs
# Global variables
pose=[0,0,0]
tree_a=None
tree_b=None
path=None

# Simulation parameters
extension_r = 1.0 # In meters maximum extension distance of a branch
n_samples=3000 # Maximum number of samples 
map_topic='/move_base/global_costmap/costmap'
odom_topic= "/vesc/odom" # Should be published in Odometry type
xy_goal_tolerance = 0.2 # Goal region radius


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
    mark.pose.orientation.x = 0
    mark.pose.orientation.y = 0
    mark.pose.orientation.z = 0
    mark.pose.orientation.w = 1.0
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

class bidirectional_FMT_planner():
    def __init__(self, start,goal,map_):
        # Initialize variables
        self.samples=[]
        self.grid=map_
        self.start=start[:2]
        self.goal=goal
        self.search_r = extension_r
        self.parents1 = {}
        self.parents2 = {}
        self.n_samples = n_samples

        # Create obstacle-free n samples
        n_random=0
        while(n_random<n_samples):
            node_x=self.grid.origin.position.x+random.random()*(self.grid.map.shape[0]*self.grid.resolution)
            node_y=self.grid.origin.position.y+random.random()*(self.grid.map.shape[1]*self.grid.resolution)
            if not self.grid.check_collision([node_x,node_y]):
                n_random+=1
                self.samples.append([node_x,node_y])
        
    def plan(self):
        # Add start point, goal point and their parents 
        self.samples.append(self.start)
        self.samples.append(self.goal)
        self.parents1[self.n_samples] = -1 # First parent list (root is start)
        self.parents2[self.n_samples+1] = -1 # Second parent list (root is goal)

        z = self.n_samples # First assign z to start node index

        # Initialize sets and queues for first tree
        V_open1 = pqdict({z: 0.})
        V_closed1 = []
        V_unvisited1 = list(range(len(self.samples)))
        V_unvisited1.remove(z)

        # Initialize sets and queues for second tree
        V_open2 = pqdict({z+1: 0.})
        V_closed2 = []
        V_unvisited2 = list(range(len(self.samples)))
        V_unvisited2.remove(z+1)

        for i in range(self.n_samples):
            if i%2==0: # Turn based expansion 
                # Tree 1 expansion -----
                if not V_open1: # If tree 1 queue empty and solution is not found return with failure
                    return [],[],[]
                z = V_open1.top() # Get the most advantegous node in open queue of Tree 1

                unv_neigh = fast_funcs.radius_neighbors_inlist(self.samples[z],self.samples,self.search_r,list(set(V_unvisited1))) # Find radius neighbors of z in unvisited set
                
                for u_n in unv_neigh:
                    open_neigh=fast_funcs.radius_neighbors_inlist(self.samples[u_n], self.samples, self.search_r,list(set(V_open1))) # Find radius neighbors of unvisited sample in open set
                    
                    if len(open_neigh)==0: # If open neighbor list is empty skip loop
                        continue

                    #Calculate costs to reach this unvisited node u_n from open nodes in neighborhood
                    costs = [V_open1[neigh] + m.hypot(self.samples[neigh][0]-self.samples[u_n][0], self.samples[neigh][1]-self.samples[u_n][1]) for neigh in open_neigh]
                    y_min = open_neigh[costs.index(min(costs))] # Find the connection which has the minimum cost.
            
                    # Subsample the connection and check for collision
                    parent_p = self.samples[y_min]
                    new_p = self.samples[u_n] 
                    n_points=max(int(m.hypot(parent_p[1]-new_p[1],parent_p[0]-new_p[0])/self.grid.resolution),2)
                    traj=fast_funcs.create_ranges(parent_p,new_p,n_points)
                    coll=False
                    for pt in traj: 
                        if self.grid.check_collision(pt): 
                            coll=True
                            break
                    if coll==False and u_n not in V_open1: # If connection is collision free and unvisited node not added to the tree
                        V_open1.additem(u_n, V_open1[y_min] + m.hypot(parent_p[1]-new_p[1],parent_p[0]-new_p[0]))
                        self.parents1[u_n] = y_min
                        if u_n in list(set(V_unvisited1)):
                            V_unvisited1.remove(u_n)

                V_open1.pop(z)
                V_closed1.append(z) # Remove z from open list and add to closed list, because we already check all possibilities

                if z in V_closed2 or z in list(set(V_open2)): # If node z also added to the other tree before (we have a match)
                    path1 = self.get_path(self.parents1,z) # Get the first tree path
                    path2 = self.get_path(self.parents2,z)[::-1] # Get the second tree path
                    path=[self.samples[p] for p in path1]+[self.samples[p] for p in path2[1:]] # Concatenate coordinates

                    #For visualization purposes
                    line_list1=[self.samples[pind]+self.samples[self.parents1[pind]] for pind in self.parents1.keys() if self.parents1[pind]!=-1]
                    line_list2=[self.samples[pind]+self.samples[self.parents2[pind]] for pind in self.parents2.keys() if self.parents2[pind]!=-1]
                    return line_list1,line_list2,path
                 # Tree 1 expansion ----- END
            else:
                 # Tree 2 expansion -----
                if not V_open2: # If tree 2 queue empty and solution is not found return with failure
                    return [],[],[]
                z = V_open2.top() # Get the most advantegous node in open queue of Tree 2

                unv_neigh = fast_funcs.radius_neighbors_inlist(self.samples[z],self.samples,self.search_r,list(set(V_unvisited2))) # Find radius neighbors of z in unvisited set
                
                for u_n in unv_neigh:
                    open_neigh=fast_funcs.radius_neighbors_inlist(self.samples[u_n], self.samples, self.search_r,list(set(V_open2))) # Find radius neighbors of unvisited sample in open set
                    
                    if len(open_neigh)==0: # If open neighbor list is empty skip loop
                        continue

                    #Calculate costs to reach this unvisited node u_n from open nodes in neighborhood
                    costs = [V_open2[neigh] + m.hypot(self.samples[neigh][0]-self.samples[u_n][0], self.samples[neigh][1]-self.samples[u_n][1]) for neigh in open_neigh]
                    y_min = open_neigh[costs.index(min(costs))]
            
                    # Subsample the connection and check for collision
                    parent_p = self.samples[y_min]
                    new_p = self.samples[u_n] 
                    n_points=max(int(m.hypot(parent_p[1]-new_p[1],parent_p[0]-new_p[0])/self.grid.resolution),2)
                    traj=fast_funcs.create_ranges(parent_p,new_p,n_points)
                    coll=False
                    for pt in traj: 
                        if self.grid.check_collision(pt): 
                            coll=True
                            break
                    if coll==False and u_n not in V_open2: # If connection is collision free and unvisited node not added to the tree
                        V_open2.additem(u_n, V_open2[y_min] + m.hypot(parent_p[1]-new_p[1],parent_p[0]-new_p[0]))
                        self.parents2[u_n] = y_min
                        if u_n in list(set(V_unvisited2)):
                            V_unvisited2.remove(u_n)

                V_open2.pop(z)
                V_closed2.append(z) # Remove z from open list and add to closed list, because we already check all possibilities

                if z in V_closed1 or z in list(set(V_open1)):  # If node z also added to the other tree before (we have a match)
                    path1 = self.get_path(self.parents1,z) # Get the first tree path
                    path2 = self.get_path(self.parents2,z)[::-1] # Get the second tree path
                    path=[self.samples[p] for p in path1]+[self.samples[p] for p in path2[1:]] # Concatenate coordinates

                    #For visualization purposes
                    line_list1=[self.samples[pind]+self.samples[self.parents1[pind]] for pind in self.parents1.keys() if self.parents1[pind]!=-1] # Tree 1 all connections
                    line_list2=[self.samples[pind]+self.samples[self.parents2[pind]] for pind in self.parents2.keys() if self.parents2[pind]!=-1] # Tree 2 all connections
                    return line_list1,line_list2,path
                 # Tree 2 expansion ----- END

           
    def get_path(self,parent_dict, child_id):
        path = [child_id]
        root = parent_dict[child_id]
        while root != -1:
            path.append(root)
            root = parent_dict[root]
        path.reverse()
        return path

def goal_sub(msg): # Goal point subscriber
    global tree_a,tree_b,path
    goalP=[msg.pose.position.x,msg.pose.position.y]
    print("Goal came")
    a=time.time()

    planner=bidirectional_FMT_planner(pose,goalP,coll) # Initialize planner
    graph1,graph2,plan=planner.plan() # Perform planning

    if len(plan)>2: # If a plan is found
        tree_a=graph1
        tree_b=graph2
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
    rospy.init_node('planner1', anonymous=True)
    coll=CollisionChecker() # Initialize map

    # Initialize markers to visualize in RViz
    pts1 = create_marker(Marker.LINE_LIST, "map", 1, 1., 1.0, 0., 0.05, 0.05, 0.05)
    pts2 = create_marker(Marker.LINE_LIST, "map", 2, 0, 1.0, 0.5, 0.02, 0.02, 0.02)
    pts3 = create_marker(Marker.LINE_LIST, "map", 3, 1.0, 0.5, 0, 0.02, 0.02, 0.02)
    pub1 = rospy.Publisher('/visualization_marker', Marker, queue_size=1)

    # Subscribe goal and robot pose topics
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, goal_sub)
    rospy.Subscriber(odom_topic, Odometry, pose_sub)

    rate=rospy.Rate(100)
    counter=0
    while type(coll.origin)==type(None):  # Wait until map is initialized
            pass
    
    while not rospy.is_shutdown():
        if type(path)!=type(None): # If path is found

            # Visualize planned trajectory
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

            # Visualize tree 1
            pts2.points=[]
            for i in range(len(tree_a)):
                a=Point()
                a.x = tree_a[i][0]
                a.y = tree_a[i][1]
                a.z = 0.0
                pts2.points.append(a)
                a=Point()
                a.x = tree_a[i][2]
                a.y = tree_a[i][3]
                a.z = 0.0
                pts2.points.append(a)
            pub1.publish(pts2)
            
            # Visualize tree 2
            pts3.points=[]
            for i in range(len(tree_b)):
                a=Point()
                a.x = tree_b[i][0]
                a.y = tree_b[i][1]
                a.z = 0.0
                pts3.points.append(a)
                a=Point()
                a.x = tree_b[i][2]
                a.y = tree_b[i][3]
                a.z = 0.0
                pts3.points.append(a)
            pub1.publish(pts3)
        rate.sleep()

