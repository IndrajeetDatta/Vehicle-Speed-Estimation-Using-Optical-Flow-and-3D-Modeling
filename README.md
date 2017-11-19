# Vehicle-Speed-Estimation-Using-Optical-Flow-and-3D-Modeling
Final Year Capstone Project for Bachelor of Science in Engineering in Information and Communication Technology at Asian Institute of Technology, Thailand.

One of the biggest challenges in implementing accurate vision-based vehicle speed measurement systems is that the cameras have to be placed very high above the road in order for simple planar speed models to be accurate, making them expensive or even impossible depending on the  nearby structures. 

Cameras placed in lower elevations cannot provide accurate results due to the lack of knowledge of the height of the vehicles, as pixels corresponding to the top of taller vehicles  can seem to move at higher speeds than pixels corresponding to the tops of vehicles with lower height.

The main goal of my thesis is to develop an efficient and inexpensive vehicle speed measurement system that can provide results with high accuracy across a variety of imaging conditions from a relatively low height of about 3-4 m by 3D modeling of a cuboid around the vehicle using optical flow data.

As a result, I have been to able to use optical flow data and used it to estimate dimensions of cuboids around vehicles using non-linear regression analysis. The speed was measured by measuring the motion of the centroid of the cuboids in world coordinates which keeps in account of the height of the vehicles which eliminates the elevation problem.
