Revised from GitLab repo used for Autonomous Vehicles class at JMU Fall 2019.
Some intermediate development/testing files used during the project were not included in this repository for clarity. The functionality remains the same. 

# Pose Tracking Documentation
**Installation**

The following is a list of requirements and dependencies for the JMU Autonomous Vehicle prototype pose tracking system.
1.	OpenPose – Documentation and installation instructions can be found at https://github.com/CMU-Perceptual-Computing-Lab/openpose. Make sure to compile OpenPose for Python. Instructions also found at the link above. 
2.	Scikit-Learn – Documentation and installation instructions can be found at https://scikit-learn.org/stable/install.html. Should be able to use ‘pip install scikit-learn’. 
3.	OpenCV – Documentation and installation instructions can be found at https://pypi.org/project/opencv-python/. ‘pip install opencv-python’
The above requirements may also have some dependencies of their own (numpy, matlib etc). Instructions on installing these components can be found at the links above or during the installation process of the above software. 
4. ROS - Documentation and installation instructions can be found at https://www.ros.org/install/ . 


‘pose_tracking.py’ is the entry point to the system. This file runs as a ROS node and should only be run from Auto Cart’s ROS launch file. Documentation for that can be found from the ROS Navigation team.
 
