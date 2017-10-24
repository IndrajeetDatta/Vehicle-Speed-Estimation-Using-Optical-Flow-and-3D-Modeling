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



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Track::Track(Blob &blob)
{
	blob.findFeatures();
	blob.findFlows();
	this->v_blobs.push_back(blob);

	this->beingTracked = true;
	this->trackUpdated = false;
	this->trackColor = Scalar(rand() % 256, rand() % 256, rand() % 256);

	this->initialCuboidLength = 5.0; //5.0;
	this->initialCuboidWidth = 2.0; // 2.0;
	this->initialCuboidHeight = 1.5; // 1.5;

	vector<Point2f> flowTails, flowHeads;
	flowTails = blob.getFlowTails();
	flowHeads = blob.getFlowHeads();

	vector<float> v_flowLengthX, v_flowLengthY;
	float sum_flowLengthX = 0, sum_flowLengthY = 0;

	cout << "Flow Tails" << "\t" << "Flow Heads" << endl;
	for (int i = 0; i < flowTails.size(); i++)
	{
		cout << flowTails[i] << "\t" << flowHeads[i] << endl;
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

	cout << "Cuboid Optimization Started: " << endl;
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


	cout << "Optimization results" << endl;
	cout << "\t Status: " << status << endl;
	cout << "\t No. of iterations: " << lm.iter << endl;
	cout << "\t No. of function evaluations: " << lm.nfev << endl;
	cout << "\t No. of jacobian evaliations: " << lm.njev << endl;
	cout << "\t Sum of squared errors: " << sumErrors << endl;
	cout << "\t Intial Length: " << this->initialCuboidLength << " Optimized Length : " << this->optimizedCuboidLength << endl;
	cout << "\t Intial Width: " << this->initialCuboidWidth << " Optimized Length : " << this->optimizedCuboidWidth << endl;
	cout << "\t Intial Height: " << this->initialCuboidHeight << " Optimized Height: " << this->optimizedCuboidHeight << endl;
	cout << "\t Initial Motion in X-axis: " << this->initialMotionX << " Optimized Motion in X-axis: " << this->optimizedMotionX << endl;
	cout << "\t Intial Motion in Y-axis: " << this->initialMotionY << " Optimized Motion in Y-Axis: " << this->optimizedMotionY << endl;
	cout << endl << endl;





	Cuboid cuboid(point, this->optimizedCuboidLength, this->optimizedCuboidWidth, this->optimizedCuboidHeight, atan(this->optimizedMotionX / this->optimizedMotionY));
	cuboid.findFlowsOnPlanes(flowTails, flowHeads);

	this->v_cuboids.push_back(cuboid);
	cuboidCount++;

	Point3f point2(point.x + this->optimizedMotionX, point.y + this->optimizedMotionY, 0.0);
	this->nextPoint = point2;


}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Track::add(Blob &blob)
{
	blob.setFlowTails(v_blobs.back().getFlowHeads());
	blob.findFlows();
	this->v_blobs.push_back(blob);

	this->trackUpdated = true;
	this->matchCount++;
	this->noMatchCount = 0;





	this->initialCuboidLength = this->optimizedCuboidLength;
	this->initialCuboidWidth = this->optimizedCuboidWidth;
	this->initialCuboidHeight = this->optimizedCuboidHeight;

	vector<Point2f> flowTails, flowHeads;
	flowTails = blob.getFlowTails();
	flowHeads = blob.getFlowHeads();

	vector<float> v_flowLengthX, v_flowLengthY;
	float sum_flowLengthX = 0, sum_flowLengthY = 0;

	cout << "Flow Tails" << "\t" << "Flow Heads" << endl;
	for (int i = 0; i < flowTails.size(); i++)
	{
		cout << flowTails[i] << "\t" << flowHeads[i] << endl;
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

	cout << "Cuboid Optimization Started: " << endl;
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


	cout << "Optimization results" << endl;
	cout << "\t Status: " << status << endl;
	cout << "\t No. of iterations: " << lm.iter << endl;
	cout << "\t No. of function evaluations: " << lm.nfev << endl;
	cout << "\t No. of jacobian evaliations: " << lm.njev << endl;
	cout << "\t Sum of squared errors: " << sumErrors << endl;
	cout << "\t Intial Length: " << this->initialCuboidLength << " Optimized Length : " << this->optimizedCuboidLength << endl;
	cout << "\t Intial Width: " << this->initialCuboidWidth << " Optimized Length : " << this->optimizedCuboidWidth << endl;
	cout << "\t Intial Height: " << this->initialCuboidHeight << " Optimized Height: " << this->optimizedCuboidHeight << endl;
	cout << "\t Initial Motion in X-axis: " << this->initialMotionX << " Optimized Motion in X-axis: " << this->optimizedMotionX << endl;
	cout << "\t Intial Motion in Y-axis: " << this->initialMotionY << " Optimized Motion in Y-Axis: " << this->optimizedMotionY << endl;

	Cuboid cuboid(point, this->optimizedCuboidLength, this->optimizedCuboidWidth, this->optimizedCuboidHeight, atan(this->optimizedMotionX / this->optimizedMotionY));
	cuboid.findFlowsOnPlanes(flowTails, flowHeads);

	this->v_cuboids.push_back(cuboid);
	cuboidCount++;

	Point3f point2(point.x + this->optimizedMotionX, point.y + this->optimizedMotionY, 0.0);
	this->nextPoint = point2;

	Point3f centroid1 = v_cuboids.rbegin()[0].getCentroid();
	Point3f centroid2 = v_cuboids.rbegin()[1].getCentroid();
	float speed = sqrt(pow((centroid1.x - centroid2.x), 2) + pow((centroid1.y - centroid2.y), 2)) / (1 / frameRate) * 3.6;

	instantaneousSpeeds_.push_back(speed);
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////
float Track::findAverageSpeed()
{
	double total = 0;

	for (int i = 0; i < instantaneousSpeeds_.size(); i++)
	{
		total += instantaneousSpeeds_[i];
	}
	return total / instantaneousSpeeds_.size();
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Track::drawCuboid(Mat &outputFrame, Scalar color, int lineThickness)
{
	Cuboid cuboid = v_cuboids.back();
	vector<Point2f> imagePoints;

	projectPoints(cuboid.getVertices(), rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

	line(outputFrame, imagePoints[0], imagePoints[1], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[1], imagePoints[2], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[2], imagePoints[3], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[3], imagePoints[0], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[4], imagePoints[5], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[5], imagePoints[6], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[6], imagePoints[7], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[7], imagePoints[4], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[0], imagePoints[4], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[1], imagePoints[5], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[2], imagePoints[6], color, lineThickness, CV_AA);
	line(outputFrame, imagePoints[3], imagePoints[7], color, lineThickness, CV_AA);


	putText(outputFrame, "L: " + to_string(this->optimizedCuboidLength), Point2f(imagePoints[0].x, imagePoints[0].y + 20), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);


	putText(outputFrame, "W: " + to_string(this->optimizedCuboidWidth), Point2f(imagePoints[0].x, imagePoints[0].y + 30), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "H: " + to_string(this->optimizedCuboidHeight), Point2f(imagePoints[0].x, imagePoints[0].y + 40), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "A: " + to_string(this->optimizedAngleOfMotion), Point2f(imagePoints[0].x, imagePoints[0].y + 50), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "M.X.: " + to_string(this->optimizedMotionX), Point2f(imagePoints[0].x, imagePoints[0].y + 60), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "M.Y.: " + to_string(this->optimizedMotionY), Point2f(imagePoints[0].x, imagePoints[0].y + 70), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "I.S.: " + to_string(this->instantaneousSpeeds_.back()), Point2f(imagePoints[0].x, imagePoints[0].y + 80), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

	putText(outputFrame, "A.S.: " + to_string(this->findAverageSpeed()), Point2f(imagePoints[0].x, imagePoints[0].y + 90), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

}
void Track::drawBlobTrail(Mat &outputFrame, int trailLength, Scalar trailColor, int trailThickness)
{
	for (int i = 0; i < min((int)v_blobs.size(), trailLength); i++)
	{
		line(outputFrame, v_blobs.rbegin()[i].getCenter(), v_blobs.rbegin()[i + 1].getCenter(), trailColor, trailThickness, CV_AA);
	}
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Track::drawCuboidInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness)
{

	Cuboid cuboid = this->v_cuboids.back();

	vector<Point3f> objectPoints; vector<Point2f> imagePoints;
	objectPoints = cuboid.getVertices();
	projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

	circle(outputFrame, imagePoints[0], 10, trackColor, -1, CV_AA);
}


Track::~Track() {};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

