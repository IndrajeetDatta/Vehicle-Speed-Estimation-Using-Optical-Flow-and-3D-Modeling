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



////////////////////////////////////////////////////////////////////////////////////////////////

Track::Track(Blob &blob, int trackNumber)
{
	this->trackNumber = trackNumber;
	this->beingTracked = true;
	this->trackUpdated = false;
	this->trackColor = Scalar(rand() % 256, rand() % 256, rand() % 256);
	this->matchCount = 1;
	this->noMatchCount = 0;

	blob.findFeatures();
	blob.findFlows();
	this->v_blobs.push_back(blob);

	

	this->initialCuboidLength = 5.0; //5.0;
	this->initialCuboidWidth = 2.0; // 2.0;
	this->initialCuboidHeight = 1.5; // 1.5;

	vector<Point2f> flowTails, flowHeads;
	flowTails = blob.getFlowTails();
	flowHeads = blob.getFlowHeads();

	

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
	this->initialMotionX = sum_flowLengthX / flowTails.size();
	this->initialMotionY = sum_flowLengthY / flowTails.size();

	sort(v_flowLengthX.begin(), v_flowLengthX.end());
	sort(v_flowLengthY.begin(), v_flowLengthY.end());



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

	this->cuboidCount++;
	optimizationData << "frame-number" << currentFrameCount;
	optimizationData << "track-number" << this->trackNumber;
	optimizationData << "cuboid-number" << cuboidCount;
	optimizationData << "flow-tails" << flowTails;
	optimizationData << "flow-heads" << flowHeads;

	int n = 5;
	VectorXf parameters(n);
	parameters(0) = this->initialCuboidLength;
	parameters(1) = this->initialCuboidWidth;
	parameters(2) = this->initialCuboidHeight;
	parameters(3) = this->initialMotionX;
	parameters(4) = this->initialMotionY;

	Point3f point = findWorldPoint(blob.getCenter(), 0.0, cameraMatrix, rotationMatrix, translationVector);

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
	int status = lm.minimize(parameters);

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

	this->optimizedCuboidLength = parameters(0);
	this->optimizedCuboidWidth = parameters(1);
	this->optimizedCuboidHeight = parameters(2);
	this->optimizedMotionX = parameters(3);
	this->optimizedMotionY = parameters(4);


	float sumErrors = 0;
	for (int i = 0; i < lm.fvec.size(); i++)
	{
		sumErrors += pow(lm.fvec(i), 2);
	}

	optimizationData << "optimization-status" << status;

	optimizationData << "initial-length" << this->initialCuboidLength;

	optimizationData << "optimized-length" << this->optimizedCuboidLength;

	optimizationData << "initial-width" << this->initialCuboidWidth;

	optimizationData << "optimized-width" << this->optimizedCuboidWidth;

	optimizationData << "initial-height" << this->initialCuboidHeight;

	optimizationData << "optimized-height" << this->optimizedCuboidHeight;

	optimizationData << "initial-motion-x" << this->initialMotionX;

	optimizationData << "optimized-motion-x" << this->optimizedMotionX;

	optimizationData << "initial-motion-y" << this->initialMotionY;

	optimizationData << "optimized_motion-y" << this->optimizedMotionY;

	optimizationData << "sum-of-squared-errors" << sumErrors;





	Cuboid cuboid(point, this->optimizedCuboidLength, this->optimizedCuboidWidth, this->optimizedCuboidHeight, atan(this->optimizedMotionX / this->optimizedMotionY));
	cuboid.findFlowsOnPlanes(flowTails, flowHeads);
	cuboid.setCuboidNumber(matchCount);
	
	this->v_cuboids.push_back(cuboid);
	

	Point3f point2(point.x + this->optimizedMotionX, point.y + this->optimizedMotionY, 0.0);
	this->nextPoint = point2;

	this->instantaneousSpeeds.push_back(0.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////

void Track::add(Blob &blob)
{
	this->trackUpdated = true;
	this->matchCount++;
	this->noMatchCount = 0;

	blob.setFlowTails(v_blobs.back().getFlowHeads());
	blob.findFlows();
	this->v_blobs.push_back(blob);

	this->initialCuboidLength = this->optimizedCuboidLength;
	this->initialCuboidWidth = this->optimizedCuboidWidth;
	this->initialCuboidHeight = this->optimizedCuboidHeight;

	vector<Point2f> flowTails, flowHeads;
	flowTails = blob.getFlowTails();
	flowHeads = blob.getFlowHeads();



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
	this->initialMotionX = sum_flowLengthX / flowTails.size();
	this->initialMotionY = sum_flowLengthY / flowTails.size();

	sort(v_flowLengthX.begin(), v_flowLengthX.end());
	sort(v_flowLengthY.begin(), v_flowLengthY.end());

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

	this->cuboidCount++;
	optimizationData << "frame-number" << currentFrameCount;
	optimizationData << "track-number" << this->trackNumber;
	optimizationData << "cuboid-number" << cuboidCount;
	optimizationData << "flow-tails" << flowTails;
	optimizationData << "flow-heads" << flowHeads;

	int n = 5;
	VectorXf parameters(n);
	parameters(0) = this->initialCuboidLength;
	parameters(1) = this->initialCuboidWidth;
	parameters(2) = this->initialCuboidHeight;
	parameters(3) = this->initialMotionX;
	parameters(4) = this->initialMotionY;

	Point3f point = this->nextPoint;

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
	int status = lm.minimize(parameters);

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

	this->optimizedCuboidLength = parameters(0);
	this->optimizedCuboidWidth = parameters(1);
	this->optimizedCuboidHeight = parameters(2);
	this->optimizedMotionX = parameters(3);
	this->optimizedMotionY = parameters(4);

	float sumErrors = 0;
	for (int i = 0; i < lm.fvec.size(); i++)
	{
		sumErrors += pow(lm.fvec(i), 2);
	}



	optimizationData << "optimization-status" << status;

	optimizationData << "initial-length" << this->initialCuboidLength;

	optimizationData << "optimized-length" << this->optimizedCuboidLength;

	optimizationData << "initial-width" << this->initialCuboidWidth;

	optimizationData << "optimized-width" << this->optimizedCuboidWidth;

	optimizationData << "initial-height" << this->initialCuboidHeight;

	optimizationData << "optimized-height" << this->optimizedCuboidHeight;

	optimizationData << "initial-motion-x"<< this->initialMotionX;

	optimizationData << "optimized-motion-x"<< this->optimizedMotionX;

	optimizationData << "initial-motion-y" << this->initialMotionY;

	optimizationData << "optimized-motion-y"<< this->optimizedMotionY;

	optimizationData << "sum-of-squared-errors" << sumErrors;
	
	Cuboid cuboid(point, this->optimizedCuboidLength, this->optimizedCuboidWidth, this->optimizedCuboidHeight, atan(this->optimizedMotionX / this->optimizedMotionY));
	cuboid.setCuboidNumber(matchCount);
	cuboid.findFlowsOnPlanes(flowTails, flowHeads);
	this->v_cuboids.push_back(cuboid);
	

	Point3f point2(point.x + this->optimizedMotionX, point.y + this->optimizedMotionY, 0.0);
	this->nextPoint = point2;

	Point3f centroid1 = v_cuboids.rbegin()[0].getCentroid();
	Point3f centroid2 = v_cuboids.rbegin()[1].getCentroid();
	float speed = sqrt(pow((centroid1.x - centroid2.x), 2) + pow((centroid1.y - centroid2.y), 2)) / (1 / frameRate) * 3.6;

	this->instantaneousSpeeds.push_back(speed);
}
////////////////////////////////////////////////////////////////////////////////////////////////

float Track::findAverageSpeed()
{
	double total = 0;

	for (int i = 0; i < instantaneousSpeeds.size(); i++)
	{
		total += instantaneousSpeeds[i];
	}
	return total / (instantaneousSpeeds.size()-1);
}

////////////////////////////////////////////////////////////////////////////////////////////////

void Track::drawBlob(Mat &outputFrame)
{
	Blob blob = this->v_blobs.back();
	rectangle(outputFrame, blob.getBoundingRect(), this->trackColor, 2, CV_AA);
	circle(outputFrame, blob.getCenter(), 1, this->trackColor, -1, CV_AA);
	/*vector<vector<Point>> contours;
	contours.push_back(blob.getContour());
	drawContours(outputFrame, contours, -1, BLUE, 1, CV_AA);*/

	vector<Point2f> flowTails, flowHeads;
	flowTails = blob.getFlowTails();
	flowHeads = blob.getFlowHeads();
	for (int i = 0; i < flowTails.size(); i++)
	{
		circle(outputFrame, flowTails[i], 1, RED, -1, CV_AA);
		arrowedLine(outputFrame, flowTails[i], flowHeads[i], this->trackColor, 1, CV_AA);
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////

void Track::drawCuboid(Mat &outputFrame)
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

	vector<Point3f> objectPoints1, objectPoint2;
	vector<Point2f> imagePoints1, imagePoints2;

	objectPoints1 = cuboid.getCP_Flows()[0];
	objectPoint2 = cuboid.getCP_Flows()[1];
	
	projectPoints(objectPoints1, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints1);
	projectPoints(objectPoint2, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints2);
	for (int i = 0; i < imagePoints1.size(); i++)
	{
		circle(outputFrame, imagePoints1[i], 1, RED, -1, CV_AA);
		arrowedLine(outputFrame, imagePoints1[i], imagePoints2[i], this->trackColor, 1, CV_AA);
	}
	putText(outputFrame, "T No.: " + to_string(this->trackNumber), Point2f(ipVertices[0].x, ipVertices[0].y + 10), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "C No.: " + to_string(cuboid.getCuboidNumber()), Point2f(ipVertices[0].x, ipVertices[0].y + 20), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

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
