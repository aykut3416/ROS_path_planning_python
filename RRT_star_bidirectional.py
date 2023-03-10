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
pyximport.install(setup_args={"script_args":["--compiler=unix"],
                              "include_dirs":np.get_include()},
                  reload_support=True)
import fast_funcs
# Global variables
pose=[0,0,0]
tree_a=None
tree_b=None
path=None

#Simulation parameters
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


class bi_rrt_star_planner:
    def __init__(self,start_p,goal_p,map_):
        
        #Initialize parameters
        self.n_samples=n_samples
        self.start=start_p
        self.goal=goal_p
        self.grid=map_

        # Initialize the first tree
        self.samples_a=[start_p[:2]]
        self.costs_a=[0]
        self.parents_a={0:-1}

        # Initialize the second tree
        self.samples_b=[goal_p]
        self.costs_b=[0]
        self.parents_b={0:-1}

        self.max_extension = extension_r # In meters
        self.xy_tolerance = xy_goal_tolerance
        
        self.path=[]
    
    def plan(self):
        n_nodes_a=0 # Number of nodes in tree 1
        n_nodes_b=0 # Number of nodes in tree 2
        min_cost=1000 # Default cost value
        while (n_nodes_a+n_nodes_b)<self.n_samples:
            # Create a random position
            node_x=self.grid.origin.position.x+random.random()*(self.grid.map.shape[0]*self.grid.resolution)
            node_y=self.grid.origin.position.y+random.random()*(self.grid.map.shape[1]*self.grid.resolution)

            nearest_ind_a,dist_a=fast_funcs.nearest_neighbour([node_x,node_y],self.samples_a) # nearest node in tree 1
            nearest_ind_b,dist_b=fast_funcs.nearest_neighbour([node_x,node_y],self.samples_b) # nearest node in tree 2

            # First tree random target determination
            if dist_a>self.max_extension:
                angle = m.atan2(node_y-self.samples_a[nearest_ind_a][1],node_x-self.samples_a[nearest_ind_a][0]) 
                node_x_a = self.samples_a[nearest_ind_a][0] + m.cos(angle)*self.max_extension
                node_y_a = self.samples_a[nearest_ind_a][1] + m.sin(angle)*self.max_extension
                dist_a = self.max_extension
            else:
                node_x_a=node_x
                node_y_a=node_y

            # Second tree random target determination
            if dist_b>self.max_extension:
                angle = m.atan2(node_y-self.samples_b[nearest_ind_b][1],node_x-self.samples_b[nearest_ind_b][0]) 
                node_x_b = self.samples_b[nearest_ind_b][0] + m.cos(angle)*self.max_extension
                node_y_b = self.samples_b[nearest_ind_b][1] + m.sin(angle)*self.max_extension
                dist_b = self.max_extension
            else:
                node_x_b=node_x
                node_y_b=node_y
            
            # First tree expansion
            n_segments=max(2,int(dist_a/self.grid.resolution))  # Calculate subsample number
            all_segs=fast_funcs.create_ranges(self.samples_a[nearest_ind_a],[node_x_a,node_y_a],n_segments) # Draw a line between the node and discretize with the map resolution
            # Collision check for all segments
            coll=False
            for pt in all_segs: 
                if self.grid.check_collision(pt): 
                    coll=True
                    break
            if coll==False: # If all segments are safe connect this point to the tree
                n_nodes_a+=1
                self.samples_a.append([node_x_a,node_y_a])
                self.costs_a.append(dist_a+self.costs_a[nearest_ind_a])
                self.parents_a[n_nodes_a]=nearest_ind_a
                
                neighs=fast_funcs.radius_neighbors([node_x_a,node_y_a],self.samples_a,self.max_extension) # Find neighbors of new node a in tree 1 (radius : self.max_extension)
                for (neigh_id,cost) in neighs:
                    new_con_cost = m.hypot(self.samples_a[neigh_id][0]-node_x_a,self.samples_a[neigh_id][1]-node_y_a) # Distance calculation
                    n_segments=max(2,int(new_con_cost/0.05))
                    all_segs=fast_funcs.create_ranges(self.samples_a[neigh_id],[node_x_a,node_y_a],n_segments) # Draw a line between the node and discretize with the map resolution
                    # Collision check for all segments
                    coll=False
                    for pt in all_segs: 
                        if self.grid.check_collision(pt): 
                            coll=True
                            break
                    if coll==False: # If connection is safe
                        if (self.costs_a[neigh_id]+new_con_cost)<self.costs_a[n_nodes_a]: # Parent update process for new node (if shorter connection is found)
                            self.costs_a[n_nodes_a]=(self.costs_a[neigh_id]+new_con_cost)
                            self.parents_a[n_nodes_a]=neigh_id
                        elif (self.costs_a[n_nodes_a]+new_con_cost)<self.costs_a[neigh_id]: # Rewire operation (Parent update for neighbor nodes) (if shorter connection is found)
                            self.costs_a[neigh_id]=(self.costs_a[n_nodes_a]+new_con_cost)
                            self.parents_a[neigh_id]=n_nodes_a
                
                # Checking if a connection is available with the second tree
                neighs_b=fast_funcs.radius_neighbors([node_x_a,node_y_a],self.samples_b,self.max_extension) # Find neighbors of new point in other tree
                for (neigh_id,cost) in neighs_b:
                    new_con_cost = m.hypot(self.samples_b[neigh_id][0]-node_x_a,self.samples_b[neigh_id][1]-node_y_a)
                    n_segments=max(2,int(new_con_cost/0.05))
                    all_segs=fast_funcs.create_ranges(self.samples_b[neigh_id],[node_x_a,node_y_a],n_segments) # Draw a line between the node and discretize with the map resolution
                    # Check for collision for all segments
                    coll=False
                    for pt in all_segs: 
                        if self.grid.check_collision(pt): 
                            coll=True
                            break
                    if coll==False: 
                        con_cost=self.costs_b[neigh_id]+new_con_cost+self.costs_a[n_nodes_a] # Calculate new path cost
                        if con_cost<min_cost: # If connection is safe and lower cost connection is found 
                            min_cost=con_cost # Update minimum cost
                            path_a=self.retrace_path(n_nodes_a,self.parents_a) # Find first tree path
                            path_b=self.retrace_path(neigh_id,self.parents_b) # Find second tree path
                            route=[self.samples_a[i] for i in path_a[::-1]]
                            route+=[self.samples_b[i] for i in path_b] # Concatenate trajectories
            # First tree expansion --- END
                
            # Second tree expansion
            n_segments=max(2,int(dist_b/self.grid.resolution))
            all_segs=fast_funcs.create_ranges(self.samples_b[nearest_ind_b],[node_x_b,node_y_b],n_segments) # Draw a line between the node and discretize with the map resolution
            coll=False
            for pt in all_segs: 
                if self.grid.check_collision(pt): 
                    coll=True
                    break
            if coll==False: # If all segments are safe append this valid edge to the neighbors dictionary
                n_nodes_b+=1
                self.samples_b.append([node_x_b,node_y_b])
                self.costs_b.append(dist_b+self.costs_b[nearest_ind_b])
                self.parents_b[n_nodes_b]=nearest_ind_b
                
                neighs=fast_funcs.radius_neighbors([node_x_b,node_y_b],self.samples_b,self.max_extension)
                for (neigh_id,cost) in neighs:
                    new_con_cost = m.hypot(self.samples_b[neigh_id][0]-node_x_b,self.samples_b[neigh_id][1]-node_y_b)
                    n_segments=max(2,int(new_con_cost/self.grid.resolution))
                    all_segs=fast_funcs.create_ranges(self.samples_b[neigh_id],[node_x_b,node_y_b],n_segments) # Draw a line between the node and discretize with the map resolution
                    coll=False
                    for pt in all_segs: 
                        if self.grid.check_collision(pt): 
                            coll=True
                            break
                    if coll==False:
                        if (self.costs_b[neigh_id]+new_con_cost)<self.costs_b[n_nodes_b]: # Parent update process for new node (if shorter connection is found)
                            self.costs_b[n_nodes_b]=(self.costs_b[neigh_id]+new_con_cost)
                            self.parents_b[n_nodes_b]=neigh_id
                        elif (self.costs_b[n_nodes_b]+new_con_cost)<self.costs_b[neigh_id]: # Rewire operation (Parent update for neighbor nodes) (if shorter connection is found)
                            self.costs_b[neigh_id]=(self.costs_b[n_nodes_b]+new_con_cost)
                            self.parents_b[neigh_id]=n_nodes_b

                # Checking if a connection is available with the first tree
                neighs_a=fast_funcs.radius_neighbors([node_x_b,node_y_b],self.samples_a,self.max_extension) # Find neighbors of new point in other tree 
                for (neigh_id,cost) in neighs_a:
                    new_con_cost = m.hypot(self.samples_a[neigh_id][0]-node_x_b,self.samples_a[neigh_id][1]-node_y_b)
                    n_segments=max(2,int(new_con_cost/self.grid.resolution))
                    all_segs=fast_funcs.create_ranges(self.samples_a[neigh_id],[node_x_b,node_y_b],n_segments) # Draw a line between the node and discretize with the map resolution
                    coll=False
                    for pt in all_segs: 
                        if self.grid.check_collision(pt): 
                            coll=True
                            break
                    if coll==False:
                        con_cost=self.costs_a[neigh_id]+new_con_cost+self.costs_b[n_nodes_b]
                        if con_cost<min_cost: # If connection is safe and lower cost connection is found 
                            min_cost=con_cost # Update minimum cost
                            path_a=self.retrace_path(neigh_id,self.parents_a) # Find first tree path
                            path_b=self.retrace_path(n_nodes_b,self.parents_b) # Find second tree path
                            route=[self.samples_a[i] for i in path_a[::-1]]
                            route+=[self.samples_b[i] for i in path_b] # Concatenate trajectories
            # Second tree expansion -- END

        line_list_a=[self.samples_a[pind]+self.samples_a[self.parents_a[pind]] for pind in self.parents_a.keys() if pind!=0] # Store all edge lines in tree 1
        line_list_b=[self.samples_b[pind]+self.samples_b[self.parents_b[pind]] for pind in self.parents_b.keys() if pind!=0] # Store all edge lines in tree 2
        
        if min_cost<1000:   # If a path is found
            return line_list_a,line_list_b,route #Return with tree1,tree2 and plan
        else:
            return line_list_a,line_list_b,[]
        
    def retrace_path(self,last_ind,parents):
        traj=[]
        while(last_ind!= -1):
            traj.append(last_ind)
            last_ind = parents[last_ind]
        return traj
        
            
    
        
def goal_sub(msg): # Goal point subscriber
    global tree_a,tree_b,path
    goalP=[msg.pose.position.x,msg.pose.position.y]
    print("Goal came")
    a=time.time()
    planner=bi_rrt_star_planner(pose,goalP,coll) # Initialize the planner
    graph1,graph2,plan=planner.plan() # Perform planning
    if len(plan)>2: # If path is found update globals
        tree_a=graph1
        tree_b=graph2
        path=plan
    else:
        print("Path not found!")
    print(f"Time elapsed : {time.time()-a}")
    


def pose_sub(msg): # Position subscriber
    global pose
    quat = msg.pose.pose.orientation
    roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])
    pose=[msg.pose.pose.position.x,msg.pose.pose.position.y,yaw]

if __name__ == "__main__":
    rospy.init_node('bi_rrt_star_planner', anonymous=True)
    coll=CollisionChecker() # Initialize the map

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

    while type(coll.origin)==type(None): # Wait until map is initialized
            pass
    
    while not rospy.is_shutdown():
        if type(path)!=type(None): # If path is created
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

            # Visualize first tree
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
            
            # Visualize second tree
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
