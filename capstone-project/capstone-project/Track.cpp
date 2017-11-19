#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"

#include "Track.h"
#include "Blob.h"
#include "Cuboid.h"
#include "LMFunctor.h"
#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>

using namespace std;
using namespace cv;



//////////////////////////////////----- TRACK CONSTRUCTOR -----/////////////////////////////////////

Track::Track(Blob &blob, int trackNumber)
{
	// Initializing member functions.
	this->trackNumber = trackNumber;
	this->beingTracked = true; 
	this->trackUpdated = true;  
	this->trackColor = Scalar(rand() % 256, rand() % 256, rand() % 256);
	this->matchCount = 1;
	this->noMatchCount = 0;

	// Find the feature points and flow tails of the first blob passed in.
	blob.findFeatures();
	blob.findFlows();
	this->v_blobs.push_back(blob);


	// Intializing initial cuboid dimensions for optimization.
	this->initialCuboidLength = 5.0; //5.0
	this->initialCuboidWidth = 2.0; //2.0
	this->initialCuboidHeight = 1.5; //1.5

	// Obtaining the optical flow information of the blob.
	vector<Point2f> flowTails, flowHeads;
	flowTails = blob.getFlowTails();
	flowHeads = blob.getFlowHeads();


	// Finding the average optical flow motion in the ground plane.
	vector<float> v_flowLengthX, v_flowLengthY;
	float sum_flowLengthX = 0, sum_flowLengthY = 0;
	for (int i = 0; i < flowTails.size(); i++)
	{
		Point3f gp_ft = findWorldPoint(flowTails[i], 0.0, cameraMatrix, rotationMatrix, translationVector);
		Point3f gp_fh = findWorldPoint(flowHeads[i], 0.0, cameraMatrix, rotationMatrix, translationVector);

		float flowLengthX = gp_fh.x - gp_ft.x;
		float flowLengthY = gp_fh.y - gp_ft.y;

		v_flowLengthX.push_back(flowLengthX);
		v_flowLengthY.push_back(flowLengthY);

		sum_flowLengthX += flowLengthX;
		sum_flowLengthY += flowLengthY;
	}

	// Initializing the cuboids initial motion in X and Y axis to the average ground plane flow obtained.
	this->initialMotionX = sum_flowLengthX / flowTails.size();
	this->initialMotionY = sum_flowLengthY / flowTails.size();

	// Sorting the flow vectors to estimate the upper and lower bounds of the motion.
	sort(v_flowLengthX.begin(), v_flowLengthX.end());
	sort(v_flowLengthY.begin(), v_flowLengthY.end());


	// Initializing the upper and lower bounds for the parameters to be optimized.
	float minLength = 1.0; //1.3
	float maxLength = 7.0;
	float minWidth = 1.0;
	float maxWidth = 3.0;
	float minHeight = 1.0;
	float maxHeight = 3.0; //2.0
	float minMotionX = v_flowLengthX[0];
	float maxMotionX = v_flowLengthX.back();
	float minMotionY = v_flowLengthY[0];
	float maxMotionY = v_flowLengthY.back();

	// Printing out the frame number of the track number of the cuboid to be optimized (So, that I can know the printed results during optimization belongs to which cuboid).
	/*cout << "Frame Number: " << currentFrameCount << endl;
	cout << "Track Number: " << this->trackNumber << endl;
	cout << endl;*/

	// Setting the parameters to be optimized equal to the member functions of initial parameters stored in the track.
	int n = 5; // No. of parameters.
	VectorXf parameters(n);
	parameters(0) = this->initialCuboidLength;
	parameters(1) = this->initialCuboidWidth;
	parameters(2) = this->initialCuboidHeight;
	parameters(3) = this->initialMotionX;
	parameters(4) = this->initialMotionY;

	// The ground plane point of the center of the blob's bounding box is estimated to be the ground plane point of the centroid of the cuboid.
	Point3f point = findWorldPoint(blob.getCenter(), 0.0, cameraMatrix, rotationMatrix, translationVector);
	
	//// Initializing the variables of the functor that will passed in the Eigen's Levenberg-Marquardt constructor.
	LMFunctor functor;
	functor.point = point;
	functor.flowTails = flowTails;
	functor.flowHeads = flowHeads;
	functor.m = flowTails.size();
	functor.n = n;
	functor.minLength = minLength;
	functor.maxLength = maxLength;
	functor.minWidth = minWidth;
	functor.maxWidth = maxWidth;
	functor.minHeight = minHeight;
	functor.maxHeight = maxHeight;
	functor.minMotionX = minMotionX;
	functor.maxMotionX = maxMotionX;
	functor.minMotionY = minMotionY;
	functor.maxMotionY = maxMotionY;

	LevenbergMarquardt<LMFunctor, float> lm(functor);
	int status = lm.minimize(parameters); // Minimizes the parameters.

	/*cout << "Length: " << parameters(0) << endl;
	cout << "Width: " << parameters(1) << endl;
	cout << "Height: " << parameters(2) << endl;
	cout << "Motion in X: " << parameters(3) << endl;
	cout << "Motion in Y: " << parameters(4) << endl;
	cout << endl;
	cout << "Checking for upper and lower bounds of  parameters." << endl;
	cout << "Max. Length: " << maxLength << " Min. Length: " << minLength << endl;
	cout << "Max. Width: " << maxWidth << " Min. Width: " << minWidth << endl;
	cout << "Max. Height: " << maxHeight << " Min. Height: " << minHeight << endl;
	cout << "Max. Motion in X: " << maxMotionX << " Min. Motion in X: " << minMotionX << endl;
	cout << "Max. Motion in Y: " << maxMotionY << " Min. Motion in Y: " << minMotionY << endl;
	cout << endl;*/

	// Checking the upper and lower bounds for the returned paramters.
	if (parameters(0) < minLength) parameters(0) = minLength;
	if (parameters(0) > maxLength) parameters(0) = maxLength;
	if (parameters(1) < minWidth) parameters(1) = minWidth;
	if (parameters(1) > maxWidth) parameters(1) = maxWidth;
	if (parameters(2) < minHeight) parameters(2) = minHeight;
	if (parameters(2) > maxHeight) parameters(2) = maxHeight;
	if (parameters(3) < minMotionX) parameters(3) = minMotionX;
	if (parameters(3) > maxMotionX) parameters(3) = maxMotionX;
	if (parameters(4) < minMotionY) parameters(4) = minMotionY;
	if (parameters(4) > maxMotionY) parameters(4) = maxMotionY;

	// Setting the member functions to the parameters.
	this->optimizedCuboidLength = parameters(0);
	this->optimizedCuboidWidth = parameters(1);
	this->optimizedCuboidHeight = parameters(2);
	this->optimizedMotionX = parameters(3);
	this->optimizedMotionY = parameters(4);

	// Printing out the optimized results.
	/*cout << "Final Optimized Parameters." << endl;
	cout << "Optimized Length: " << this->optimizedCuboidLength << endl;
	cout << "Optimized Width: " << this->optimizedCuboidWidth << endl;
	cout << "Optimized Height: " << this->optimizedCuboidHeight << endl;
	cout << "Optimized Motion in X: " << this->optimizedMotionX << endl;
	cout << "Optimized Motion in Y: " << this->optimizedMotionY << endl;
	cout << "------------------------------------------------------------------------------" << endl;
	cout << endl;*/

	// Constructing a new cuboid with the optimized parameters.
	Cuboid cuboid(point, this->optimizedCuboidLength, this->optimizedCuboidWidth, this->optimizedCuboidHeight, atan(this->optimizedMotionX / this->optimizedMotionY));

	// Pushing the cuboid to the vector of cuboids stored in the track.
	this->v_cuboids.push_back(cuboid);

	// Predicting the ground plane point of the centroid of the next cuboid from optimied parameters of motion. 
	Point3f point2(point.x + this->optimizedMotionX, point.y + this->optimizedMotionY, 0.0);
	this->nextPoint = point2;

	// Since it is the first cuboid the instantaneous speed is set to 0.
	this->instantaneousSpeeds.push_back(0.0);
}

/////////////////////////////----- METHOD TO ADD BLOBS TO TRACK -----////////////////////////////////

void Track::add(Blob &blob)
{
	// Setting track updated to be true, incrementing match counter and setting the no match counter to 0.
	this->trackUpdated = true;
	this->matchCount++;
	this->noMatchCount = 0;

	// Setting the optical flow tails of the blob that has been passed in a parameter same as the optical flow heads of the previous blob stored in the track. Then, finding the optical flow tails.
	blob.setFlowTails(v_blobs.back().getFlowHeads());
	blob.findFlows();

	// Pushing back the blob to the vector of blobs stored in the track.
	this->v_blobs.push_back(blob);

	// Setting the initial dimensions of the new cuboid to be added to the optimized dimensions obtained from the previous cuboid in the track (Since it is supposed to be the same vehicle).
	this->initialCuboidLength = this->optimizedCuboidLength;
	this->initialCuboidWidth = this->optimizedCuboidWidth;
	this->initialCuboidHeight = this->optimizedCuboidHeight;

	// Storing the optical flow data of the blob in variables.
	vector<Point2f> flowTails, flowHeads;
	flowTails = blob.getFlowTails();
	flowHeads = blob.getFlowHeads();

	// Finding the average optical flow motion in the ground plane in the X and Y axis.
	vector<float> v_flowLengthX, v_flowLengthY;
	float sum_flowLengthX = 0, sum_flowLengthY = 0;
	for (int i = 0; i < flowTails.size(); i++)
	{
		Point3f gp_ft = findWorldPoint(flowTails[i], 0.0, cameraMatrix, rotationMatrix, translationVector);
		Point3f gp_fh = findWorldPoint(flowHeads[i], 0.0, cameraMatrix, rotationMatrix, translationVector);

		float flowLengthX = gp_fh.x - gp_ft.x;
		float flowLengthY = gp_fh.y - gp_ft.y;

		v_flowLengthX.push_back(flowLengthX);
		v_flowLengthY.push_back(flowLengthY);

		sum_flowLengthX += flowLengthX;
		sum_flowLengthY += flowLengthY;
	}
	
	// Setting the initial motion of the cuboid in the X and Y axis to the average optical flow motion found in the respective X and Y axis.
	this->initialMotionX = sum_flowLengthX / flowTails.size();
	this->initialMotionY = sum_flowLengthY / flowTails.size();

	// Sorting the flow vectors to estimate the upper and lower bounds of the motion.
	sort(v_flowLengthX.begin(), v_flowLengthX.end());
	sort(v_flowLengthY.begin(), v_flowLengthY.end());

	// Initializing the upper and lower bounds for the parameters to be optimized.
	float minLength = 1.3;
	float maxLength = 7.0;
	float minWidth = 1.0;
	float maxWidth = 3.0;
	float minHeight = 1.0;
	float maxHeight = 2.0;
	float minMotionX = v_flowLengthX[0];
	float maxMotionX = v_flowLengthX.back();
	float minMotionY = v_flowLengthY[0];
	float maxMotionY = v_flowLengthY.back();

	// Printing out the frame number of the track number of the cuboid to be optimized (So, that I can know the printed results during optimization belongs to which cuboid).
	/*cout << "Frame Number: " << currentFrameCount << endl;
	cout << "Track Number: " << this->trackNumber << endl;
	cout << endl;*/

	// Setting the parameters to be optimized equal to the member functions of initial parameters stored in the track.
	int n = 5; // No. of paramters.
	VectorXf parameters(n);
	parameters(0) = this->initialCuboidLength;
	parameters(1) = this->initialCuboidWidth;
	parameters(2) = this->initialCuboidHeight;
	parameters(3) = this->initialMotionX;
	parameters(4) = this->initialMotionY;

	// The ground plane point of the centroid of the cuboid to be optimized is set to be the point predicted by adding the optimized motion to the ground plane point of the centroid of the previous cuboid.
	Point3f point = this->nextPoint;

	// Initializing the variables of the functor that will passed in the Eigen's Levenberg-Marquardt constructor.
	LMFunctor functor;
	functor.point = point;
	functor.flowTails = flowTails;
	functor.flowHeads = flowHeads;
	functor.m = flowTails.size();
	functor.n = n;
	functor.minLength = minLength;
	functor.maxLength = maxLength;
	functor.minWidth = minWidth;
	functor.maxWidth = maxWidth;
	functor.minHeight = minHeight;
	functor.maxHeight = maxHeight;
	functor.minMotionX = minMotionX;
	functor.maxMotionX = maxMotionX;
	functor.minMotionY = minMotionY;
	functor.maxMotionY = maxMotionY;

	// Contructing Eigen's Levenberg-Marquardt optimizer.
	LevenbergMarquardt<LMFunctor, float> lm(functor);

	// Minimizing the parameters using Eigen's inbuilt minimize function.
	int status = lm.minimize(parameters);

	/*
	// Printing out the optimized paramters.
	cout << "Length: " << parameters(0) << endl;
	cout << "Width: " << parameters(1) << endl;
	cout << "Height: " << parameters(2) << endl;
	cout << "Motion in X: " << parameters(3) << endl;
	cout << "Motion in Y: " << parameters(4) << endl;
	cout << endl;

	// Printing the upper and lower bounds of the parameters and checking the results.
	cout << "Checking for upper and lower bounds of parameters." << endl;
	cout << "Max. Length: " << maxLength << " Min. Length: " << minLength << endl;
	cout << "Max. Width: " << maxWidth << " Min. Width: " << minWidth << endl;
	cout << "Max. Height: " << maxHeight << " Min. Height: " << minHeight << endl;
	cout << "Max. Motion in X: " << maxMotionX << " Min. Motion in X: " << minMotionX << endl;
	cout << "Max. Motion in Y: " << maxMotionY << " Min. Motion in Y: " << minMotionY << endl;
	cout << endl;
	*/

	// If the parameter is greater than the upper bound or lesser than the lower bound then it is set to the upper/lower bound respectively.
	if (parameters(0) < minLength) parameters(0) = minLength;
	if (parameters(0) > maxLength) parameters(0) = maxLength;
	if (parameters(1) < minWidth) parameters(1) = minWidth;
	if (parameters(1) > maxWidth) parameters(1) = maxWidth;
	if (parameters(2) < minHeight) parameters(2) = minHeight;
	if (parameters(2) > maxHeight) parameters(2) = maxHeight;
	if (parameters(3) < minMotionX) parameters(3) = minMotionX;
	if (parameters(3) > maxMotionX) parameters(3) = maxMotionX;
	if (parameters(4) < minMotionY) parameters(4) = minMotionY;
	if (parameters(4) > maxMotionY) parameters(4) = maxMotionY;

	// Setting the member functions that store the optimized parameters to the final optimized parameters.
	this->optimizedCuboidLength = parameters(0);
	this->optimizedCuboidWidth = parameters(1);
	this->optimizedCuboidHeight = parameters(2);
	this->optimizedMotionX = parameters(3);
	this->optimizedMotionY = parameters(4);

	/*
	// Printing out the final optimized parameters that will be used to construct the cuboid.
	cout << "Final Optimized Parameters." << endl;
	cout << "Optimized Length: " << this->optimizedCuboidLength << endl;
	cout << "Optimized Width: " << this->optimizedCuboidWidth << endl;
	cout << "Optimized Height: " << this->optimizedCuboidHeight << endl;
	cout << "Optimized Motion in X: " << this->optimizedMotionX << endl;
	cout << "Optimized Motion in Y: " << this->optimizedMotionY << endl;
	cout << "------------------------------------------------------------------------------" << endl;
	cout << endl;
	*/

	// Constructing a new cuboid with the final optimized parameters.
	Cuboid cuboid(point, this->optimizedCuboidLength, this->optimizedCuboidWidth, this->optimizedCuboidHeight, atan(this->optimizedMotionX / this->optimizedMotionY));

	// Pushing back the cuboid to the vector of cuboids.
	this->v_cuboids.push_back(cuboid);

	// Predicting the ground plane point of the centroid of the next cuboid in the track from the optimized parameters of motion in X and Y axis.
	Point3f point2(point.x + this->optimizedMotionX, point.y + this->optimizedMotionY, 0.0);
	this->nextPoint = point2;

	// Finding the instantaneous speed of the cuboid by diving the distance of the current and the previous cuboid in the track in the world coordinate axis and then dividing the by the time elapsed between two frame (multiplied by 3.6 to find the unit in KMPH).
	Point3f centroid1 = v_cuboids.rbegin()[0].getCentroid();
	Point3f centroid2 = v_cuboids.rbegin()[1].getCentroid();
	float speed = sqrt(pow((centroid1.x - centroid2.x), 2) + pow((centroid1.y - centroid2.y), 2)) / (1 / frameRate) * 3.6;

	// Pushing the instantaneous speed found to the vector that stores the instantaneous speeds of the cuboids in order.
	this->instantaneousSpeeds.push_back(speed);
}
////////////////////////----- METHOD TO FIND AVERAGE SPEED OF THE TRACK -----////////////////////////

float Track::findAverageSpeed() // This method finds the average speed of the track until a specific frame by dividing the sum of all the instantaneous speeds of each cuboid in the track and by diving that sum by total number of cuboids stored in the track.
{
	double total = 0;

	for (int i = 0; i < instantaneousSpeeds.size(); i++)
	{
		total += instantaneousSpeeds[i];
	}
	return total / (instantaneousSpeeds.size() - 1);
}

////////////////////////----- METHOD TO DRAW BLOB's BOUNDING BOXES -----/////////////////////////////

void Track::drawBlob(Mat &outputFrame) // This method draws the bounding box of the last blob stored int the track and also the contour of the blob. It also shows the track number.
{
	Blob blob = this->v_blobs.back();
	rectangle(outputFrame, blob.getBoundingRect(), this->trackColor, 2, CV_AA);
	circle(outputFrame, blob.getCenter(), 1, this->trackColor, -1, CV_AA);

	vector<vector<Point>> contours;
	contours.push_back(blob.getContour());
	drawContours(outputFrame, contours, -1, BLUE, 1, CV_AA);

	rectangle(outputFrame, Point(blob.getBoundingRect().x - 1, blob.getBoundingRect().y - 15), Point(blob.getBoundingRect().x + blob.getWidth(), blob.getBoundingRect().y), this->trackColor, -1, CV_AA);
	putText(outputFrame, "Track: " + to_string(this->trackNumber), Point(blob.getBoundingRect().x + 3, blob.getBoundingRect().y - 3), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);
}
/////////////////////////----- METHOD FOR DRAWING OPTICAL FLOWS ------//////////////////////////

void Track::drawFlows(Mat &outputFrame) // This method draws the optical flow vectors on to an output frame.
{
	Blob blob = this->v_blobs.back();
	vector<Point2f> flowTails, flowHeads;
	flowTails = blob.getFlowTails();
	flowHeads = blob.getFlowHeads();
	for (int i = 0; i < flowTails.size(); i++)
	{
		circle(outputFrame, flowTails[i], 1, RED, -1, CV_AA);
		arrowedLine(outputFrame, flowTails[i], flowHeads[i], this->trackColor, 1, CV_AA);
	}

}

//////////////////////////////----- METHOD FOR DRAWING CUBOIDS -----/////////////////////////////////

void Track::drawCuboid(Mat &outputFrame) // This method draws the cuboid projected on to the image plane. It also shows the track number and the optimized paramters of the cuboid and also the instantaneous and the average speed.
{
	Cuboid cuboid = v_cuboids.back();
	vector<Point2f> ipVertices = cuboid.getIPVertices();

	line(outputFrame, ipVertices[0], ipVertices[1], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[1], ipVertices[2], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[2], ipVertices[3], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[3], ipVertices[0], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[4], ipVertices[5], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[5], ipVertices[6], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[6], ipVertices[7], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[7], ipVertices[4], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[0], ipVertices[4], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[1], ipVertices[5], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[2], ipVertices[6], this->trackColor, 2, CV_AA);
	line(outputFrame, ipVertices[3], ipVertices[7], this->trackColor, 2, CV_AA);

	rectangle(outputFrame, Point((ipVertices[0].x + ipVertices[3].x) / 2 - 10, (ipVertices[0].y + ipVertices[4].y) / 2 - 5), Point((ipVertices[0].x + ipVertices[3].x) / 2 + 10, (ipVertices[0].y + ipVertices[4].y) / 2 + 5), this->trackColor, -1, CV_AA);


	putText(outputFrame, to_string(this->trackNumber), Point2f((ipVertices[0].x + ipVertices[3].x) / 2 - 8, (ipVertices[0].y + ipVertices[4].y) / 2 + 3), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "T. No.: " + to_string(this->trackNumber), Point2f(ipVertices[0].x, ipVertices[0].y + 10), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "L: " + to_string(this->optimizedCuboidLength), Point2f(ipVertices[0].x, ipVertices[0].y + 30), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "W: " + to_string(this->optimizedCuboidWidth), Point2f(ipVertices[0].x, ipVertices[0].y + 40), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "H: " + to_string(this->optimizedCuboidHeight), Point2f(ipVertices[0].x, ipVertices[0].y + 50), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "M.X.: " + to_string(this->optimizedMotionX), Point2f(ipVertices[0].x, ipVertices[0].y + 60), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "M.Y.: " + to_string(this->optimizedMotionY), Point2f(ipVertices[0].x, ipVertices[0].y + 70), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "I.S.: " + to_string(this->instantaneousSpeeds.back()), Point2f(ipVertices[0].x, ipVertices[0].y + 80), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "A.S.: " + to_string(this->findAverageSpeed()), Point2f(ipVertices[0].x, ipVertices[0].y + 90), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

}


////////////////////////////////////////////////////////////////////////////////////////////////

Track::~Track() {};

////////////////////////////////////////////////////////////////////////////////////////////////
