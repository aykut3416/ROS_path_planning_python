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
tree=None
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

class rrt_planner:
    def __init__(self,start_p,goal_p,map_):
        # Initialize parameters
        self.n_samples=n_samples
        self.start=start_p
        self.goal=goal_p
        self.grid=map_
        self.samples=[start_p[:2]]
        self.edges=[]
        self.costs=[0]
        self.max_extension = extension_r # In meters
        self.parents={0:-1}
        self.path=[]
    def plan(self): 
        n_nodes=0
        while n_nodes<self.n_samples:
            # Try to create a random sample 
            node_x=self.grid.origin.position.x+random.random()*(self.grid.map.shape[0]*self.grid.resolution)
            node_y=self.grid.origin.position.y+random.random()*(self.grid.map.shape[1]*self.grid.resolution)
            nearest_ind,dist=fast_funcs.nearest_neighbour([node_x,node_y],self.samples)
            if dist>self.max_extension: # Limit sample with extension radius if random point is not close the nearest node on tree
                angle = m.atan2(node_y-self.samples[nearest_ind][1],node_x-self.samples[nearest_ind][0]) 
                node_x = self.samples[nearest_ind][0] + m.cos(angle)*self.max_extension
                node_y = self.samples[nearest_ind][1] + m.sin(angle)*self.max_extension
                dist = self.max_extension
            n_segments=max(2,int(dist/self.grid.resolution)) # Calculate subsample number
            all_segs=fast_funcs.create_ranges(self.samples[nearest_ind],[node_x,node_y],n_segments) # Draw a line between the node and discretize with the map resolution
            coll=False
            for pt in all_segs: 
                if self.grid.check_collision(pt): 
                    coll=True
                    break
            if coll==False: # If all segments are safe connect this point to the tree
                self.samples.append([node_x,node_y])
                self.parents[len(self.samples)-1]=nearest_ind
                n_nodes+=1
                if m.hypot(node_x-self.goal[0],node_y-self.goal[1])<xy_goal_tolerance: # If the node in goal region
                    self.retrace_path(n_nodes) # Backward search
                    break
        line_list=[self.samples[pind]+self.samples[self.parents[pind]] for pind in self.parents.keys() if pind!=0] # Store all edge lines in tree
        
        if type(self.path)!=type(None): # If path is found
            self.path = self.path[::-1] # Reverse sample index order of path
            route=[self.samples[self.path[i]] for i in range(len(self.path))] 
            route.append(self.goal) # Add goal point to end of the route       
            return line_list,route
        else:
            return line_list,[]

    def retrace_path(self,last_ind): # Backward search function
        self.path.append(last_ind)
        if self.parents[last_ind] == -1:
            return
        self.retrace_path(self.parents[last_ind])
            
    


        
def goal_sub(msg): # Goal message subscriber
    global tree,path
    goalP=[msg.pose.position.x,msg.pose.position.y]
    print("Goal came")
    a=time.time()
    planner=rrt_planner(pose,goalP,coll) # Initialize the planner
    graph,plan=planner.plan() # Perform planning process
    if plan!=[]: # If a path is found
        tree=graph
        path=plan
    else:
        print("Path not found!")
    print(f"Time elapsed : {time.time()-a}")
    


def pose_sub(msg): # Position topic subscriber
    global pose
    quat = msg.pose.pose.orientation
    roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w]) # In order to convert quaternion to euler angles (not necessary)
    pose=[msg.pose.pose.position.x,msg.pose.pose.position.y,yaw]

if __name__ == "__main__":
    rospy.init_node('rrt_planner', anonymous=True)
    coll=CollisionChecker() # Initialize the map

    # Initialize markers to visualize in RViz
    pts1 = create_marker(Marker.LINE_LIST, "map", 2, 1., 1.0, 0., 0.05, 0.05, 0.05)
    pts2 = create_marker(Marker.LINE_LIST, "map", 1, 0.5, 0.5, 0.5, 0.02, 0.02, 0.02)
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
