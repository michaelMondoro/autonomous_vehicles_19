Revised from GitLab repo used for Autonomous Vehicles class at JMU Fall 2019.

# Pose Tracking Documentation
**Installation**

The following is a list of requirements and dependencies for the JMU Autonomous Vehicle prototype pose tracking system.
1.	OpenPose – Documentation and installation instructions can be found at https://github.com/CMU-Perceptual-Computing-Lab/openpose. Make sure to compile OpenPose for Python. Instructions also found at the link above. 
2.	Scikit-Learn – Documentation and installation instructions can be found at https://scikit-learn.org/stable/install.html. Should be able to use ‘pip install scikit-learn’. 
3.	OpenCV – Documentation and installation instructions can be found at https://pypi.org/project/opencv-python/. ‘pip install opencv-python’
The above requirements may also have some dependencies of their own (numpy, matlib etc). Instructions on installing these components can be found at the links above or during the installation process of the above software. 


**Execution**

Before running test_live_analysis.py or test_live_analysis_v2.py, you should check to make sure the path to the trained model is correct within the file. To run either program, simply run ‘python3.5 test_live_analysis.py’. (Depending on how you compiled OpenPose, OpenCV and Scikit-Learn you may run in a different python version.)

‘pose_tracking.py’ runs as a ROS node and should only be run from Auto Cart’s  ROS launch file. Documentation for that can be found in the ROS section of this report. 
 
