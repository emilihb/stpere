St. Pere Project
================
This a project created related with the work done for my PhD. 
There's currently an implementatin of an 8-state EKF that uses
constant velocity model with acceleration noise to estimate the
trajectory of an imaging sonar.


The dataset can be found in http://eia.udg.edu/~dribas/files/StPereDataset.zip

Instructions on how to interpret data in http://eia.udg.edu/~dribas/files/description.pdf

@article{ribas08,
	Author = {D. Ribas and P. Ridao and J.D. Tard{\'o}s and J. Neira},
	Date-Added = {2008-04-17 10:55:49 +0200},
	Date-Modified = {2008-12-09 13:03:48 +0100},
	Journal = {Journal of Field Robotics},
	Month = {November - December},
	Number = {11-12},
	Pages = {898-921},
	Title = {Underwater {SLAM} in Man Made Structured Environments},
	Volume = {25},
	Year = {2008}}


Instructions
------------
	python odometry.py
	

TODO
----
	- 12-state EKF
	- Scan matching algorithms (icp, probialisitic ICP, pIC)
	

