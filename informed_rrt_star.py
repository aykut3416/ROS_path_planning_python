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


class informed_rrt_star:
    def __init__(self,start_p,goal_p,map_):
        self.n_samples=n_samples
        self.start=start_p
        self.goal=goal_p
        self.grid=map_
        self.samples=[start_p[:2]]
        self.edges=[]
        self.costs=[0]
        self.max_extension = extension_r # In meters
        self.xy_tolerance = xy_goal_tolerance
        self.parents={0:-1}
        self.path=[]
        self.c = None
        self.m = None

    def sample_setup(self,p_start,p_end,d_best): # Setup sampling parameters (m and c) when new best path is found
        dmin=m.hypot(p_start[1]-p_end[1],p_start[0]-p_end[0])
        m1=np.array([[(p_end[0]-p_start[0])/dmin,-(p_end[1]-p_start[1])/dmin],[(p_end[1]-p_start[1])/dmin,(p_end[0]-p_start[0])/dmin]])
        m2=np.array([[d_best/2.0,0],[0,m.sqrt(d_best**2-dmin**2)/2.0]])
        self.m=np.array([[(p_end[0]+p_start[0])/2.0],[(p_end[1]+p_start[1])/2.0]])
        self.c=np.matmul(m1,m2)

    def sample(self): # Sample new point by using ellipse parameters (m and c)
        x=random.random()*2.0-1.0
        y=random.random()*2.0-1.0
        while (m.sqrt(y**2+x**2)>1.0):
            y=random.random()*2.0-1.0
        res=np.matmul(self.c,np.array([[x],[y]]))+self.m
        return list(res.reshape(-1))
    
    def plan(self):
        
        c_min=1000 # Default path length
        n_nodes=0 # Number of nodes in tree
        route=[]
        while n_nodes<self.n_samples:
            if c_min == 1000: # Random sampling ensure the sample is obstacle free
                node_x=self.grid.origin.position.x+random.random()*(self.grid.map.shape[0]*self.grid.resolution)
                node_y=self.grid.origin.position.y+random.random()*(self.grid.map.shape[1]*self.grid.resolution)
                while(self.grid.check_collision([node_x,node_y])):
                    node_x=self.grid.origin.position.x+random.random()*(self.grid.map.shape[0]*self.grid.resolution)
                    node_y=self.grid.origin.position.y+random.random()*(self.grid.map.shape[1]*self.grid.resolution)
            else: # Informed (ellipsoidal) sampling ensure the sample is obstacle free
                res = self.sample()
                node_x = res[0]
                node_y = res[1]
                while(self.grid.check_collision([node_x,node_y])):
                    res = self.sample()
                    node_x = res[0]
                    node_y = res[1]
            
            nearest_ind,dist=fast_funcs.nearest_neighbour([node_x,node_y],self.samples)
            if dist>self.max_extension:
                angle = m.atan2(node_y-self.samples[nearest_ind][1],node_x-self.samples[nearest_ind][0]) 
                node_x = self.samples[nearest_ind][0] + m.cos(angle)*self.max_extension
                node_y = self.samples[nearest_ind][1] + m.sin(angle)*self.max_extension
                dist = self.max_extension
            n_segments=max(2,int(dist/self.grid.resolution))  # Calculate subsample number
            all_segs=fast_funcs.create_ranges(self.samples[nearest_ind],[node_x,node_y],n_segments) # Draw a line between the node and discretize with the map resolution
            # Collision check for all segments
            coll=False
            for pt in all_segs: 
                if self.grid.check_collision(pt): 
                    coll=True
                    break
            if coll==False: # If all segments are safe connect this point to the tree
                n_nodes+=1
                self.samples.append([node_x,node_y])
                self.costs.append(dist+self.costs[nearest_ind])
                self.parents[n_nodes]=nearest_ind
                
                if m.hypot(node_x-self.goal[0],node_y-self.goal[1])<self.xy_tolerance: # Goal region is reached? 
                    path=self.retrace_path(n_nodes,self.parents) # Backward search
                    if type(path)!=type(None): # If path is found
                        temp_path=[self.samples[ind] for ind in path[::-1]]+[self.goal] # Calculate new trajectory
                        c_sol=sum([m.hypot(temp_path[ind][0]-temp_path[ind+1][0],temp_path[ind][1]-temp_path[ind+1][1]) for ind in range(len(temp_path)-1)]) # Calculate path cost
                        if c_sol<c_min: # If new cost less than current min cost
                            c_min=c_sol # Update min cost
                            route= temp_path # Update path
                            self.sample_setup(self.start,self.goal,c_min) # Update ellipse parameters

                neighs=fast_funcs.radius_neighbors([node_x,node_y],self.samples,self.max_extension) # Find neighbors of new node in tree (radius : self.max_extension)
                for (neigh_id,cost) in neighs:
                    new_con_cost = m.hypot(self.samples[neigh_id][0]-node_x,self.samples[neigh_id][1]-node_y)
                    n_segments=max(2,int(new_con_cost/self.grid.resolution)) # Calculate subsample number
                    all_segs=fast_funcs.create_ranges(self.samples[neigh_id],[node_x,node_y],n_segments) # Draw a line between the node and its nearest neighbor in map resolution
                    # Collision check for all segments
                    coll=False
                    for pt in all_segs: 
                        if self.grid.check_collision(pt): 
                            coll=True
                            break
                    if coll==False: # If connection is safe
                        if (self.costs[neigh_id]+new_con_cost)<self.costs[n_nodes]: # Parent update process for new node (if shorter connection is found)
                            self.costs[n_nodes]=(self.costs[neigh_id]+new_con_cost)
                            self.parents[n_nodes]=neigh_id
                        elif (self.costs[n_nodes]+new_con_cost)<self.costs[neigh_id]: # Rewire operation (Parent update for neighbor nodes) (if shorter connection is found)
                            self.costs[neigh_id]=(self.costs[n_nodes]+new_con_cost)
                            self.parents[neigh_id]=len(self.samples)-1
                

        line_list=[self.samples[pind]+self.samples[self.parents[pind]] for pind in self.parents.keys() if pind!=0] # Store all edge lines in tree
        return line_list,route
        
    
    def retrace_path(self,last_ind,parents): # Backward search function
        traj=[]
        while(last_ind!= -1):
            traj.append(last_ind)
            last_ind = parents[last_ind]
        return traj
    
        
def goal_sub(msg): # Goal topic subscriber
    global tree,path
    goalP=[msg.pose.position.x,msg.pose.position.y]
    print("Goal came")
    a=time.time()
    planner=informed_rrt_star(pose,goalP,coll) # Initialize planner
    graph,plan=planner.plan() # Run planning process
    if len(plan)>2: # If a path is found
        tree=graph
        path=plan
    else:
        print("Path not found!")
    print(f"Time elapsed : {time.time()-a}")
    


def pose_sub(msg): # Pose topic publisher
    global pose
    quat = msg.pose.pose.orientation
    roll, pitch, yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w]) # In order to convert quaternion to euler angles (not necessary)
    pose=[msg.pose.pose.position.x,msg.pose.pose.position.y,yaw]

if __name__ == "__main__":
    rospy.init_node('informed_rrt_star', anonymous=True)
    coll=CollisionChecker() # Initialize the map

    # Initialize markers to visualize in RViz
    pts1 = create_marker(Marker.LINE_LIST, "map", 2, 1., 1.0, 0., 0.05, 0.05, 0.05)
    pts2 = create_marker(Marker.LINE_LIST, "map", 1, 0.5, 0.5, 0.5, 0.02, 0.02, 0.02)
    pub1 = rospy.Publisher('/visualization_marker', Marker, queue_size=1)

    # Subscribe goal and robot pose topics
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, goal_sub)
    rospy.Subscriber(odom_topic, Odometry, pose_sub)

    rate=rospy.Rate(100)
    counter=0
    while type(coll.origin)==type(None): # Wait until map is initialized
            pass
    
    while not rospy.is_shutdown():
        if type(path)!=type(None): # If path is found

            # Visualize tree
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
            
            # Visualize planned path
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
