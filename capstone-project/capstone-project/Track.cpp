#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"

#include "Track.h"
#include "Blob.h"
#include "Cuboid.h"

#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>

using namespace std;
using namespace cv;
using namespace Eigen;

static vector<vector<float> > calculateErrors(Cuboid cuboid, Blob blob, Point3f cameraCenter);

typedef Matrix<Point2f, Dynamic, Dynamic> MatrixPoint2f;

struct LMFunctor
{
	MatrixPoint2f observedFlows;
	Blob blob;
	Cuboid cuboid;

	int operator()(const VectorXf &x, VectorXf &fvec) const
	{
		float length = x(0);
		float width = x(1);
		float height = x(2);
		float angleOfMotion = x(3);

		vector<vector<float> > planeParameters;

		vector<float> frontPlaneParameters;
		frontPlaneParameters.push_back(-cos(angleOfMotion));
		frontPlaneParameters.push_back(sin(angleOfMotion));
		frontPlaneParameters.push_back(0);
		frontPlaneParameters.push_back((length / 2) + cuboid.getCentroid().x * cos(angleOfMotion) - cuboid.getCentroid().y * sin(angleOfMotion));

		planeParameters.push_back(frontPlaneParameters);

		vector<float> rightPlaneParameters;
		rightPlaneParameters.push_back(cos(angleOfMotion));
		rightPlaneParameters.push_back(sin(angleOfMotion));
		rightPlaneParameters.push_back(0);
		rightPlaneParameters.push_back((width) / 2 - cuboid.getCentroid().x * cos(angleOfMotion) - cuboid.getCentroid().y * sin(angleOfMotion));

		planeParameters.push_back(rightPlaneParameters);

		vector<float> backPlaneParameters;
		backPlaneParameters.push_back(sin(angleOfMotion));
		backPlaneParameters.push_back(-cos(angleOfMotion));
		backPlaneParameters.push_back(0);
		backPlaneParameters.push_back((length / 2) - cuboid.getCentroid().x * sin(angleOfMotion) + cuboid.getCentroid().y * cos(angleOfMotion));

		planeParameters.push_back(backPlaneParameters);

		vector<float> leftPlaneParameters;
		leftPlaneParameters.push_back(-cos(angleOfMotion));
		leftPlaneParameters.push_back(-sin(angleOfMotion));
		leftPlaneParameters.push_back(0);
		leftPlaneParameters.push_back((width / 2) + cuboid.getCentroid().x * cos(angleOfMotion) + cuboid.getCentroid().y * sin(angleOfMotion));

		planeParameters.push_back(leftPlaneParameters);

		vector<float> topPlaneParameters;
		topPlaneParameters.push_back(0);
		topPlaneParameters.push_back(0);
		topPlaneParameters.push_back(-1);
		topPlaneParameters.push_back(cuboid.getCentroid().z + (height / 2));

		planeParameters.push_back(topPlaneParameters);

		vector<float> bottomPlaneParameters;
		bottomPlaneParameters.push_back(0);
		bottomPlaneParameters.push_back(0);
		bottomPlaneParameters.push_back(1);
		bottomPlaneParameters.push_back(-(cuboid.getCentroid().z) + (height / 2));

		planeParameters.push_back(bottomPlaneParameters);

		vector<vector<Point3f> > planeVertices = cuboid.getPlaneVertices();

		for (int i = 0; i < values(); i++) 
		{
			Point2f flowTail = observedFlows(i, 0);
			Point2f flowHead = observedFlows(i, 1);

			Point3f groundPlaneFlowTail = findWorldPoint(flowTail, 0, cameraMatrix, rotationMatrix, translationVector);

			Point3f groundPlaneFlowHead = findWorldPoint(flowHead, 0, cameraMatrix, rotationMatrix, translationVector);

			for (int i = 0; i < planeParameters.size(); i++)
			{

					float t1 = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlaneFlowTail.x - cameraCenter.x) + planeParameters[i][1] * (groundPlaneFlowTail.y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);

					Point3f point1 = Point3f(cameraCenter.x + (groundPlaneFlowTail.x - cameraCenter.x) * t1, cameraCenter.y + (groundPlaneFlowTail.y - cameraCenter.y) * t1, cameraCenter.z - cameraCenter.z * t1);

					float t2 = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlaneFlowHead.x - cameraCenter.x) + planeParameters[i][1] * (groundPlaneFlowHead.y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);

					Point3f point2 = Point3f(cameraCenter.x + (groundPlaneFlowHead.x - cameraCenter.x) * t2, cameraCenter.y + (groundPlaneFlowHead.y - cameraCenter.y) * t2, cameraCenter.z - cameraCenter.z * t2);

					bool inside = pointInside(planeVertices[i], point1);

					if (inside)
					{

						Point3f point3 = Point3f(point1.x + blob.getAverageFlowDistanceX(), point1.y + blob.getAverageFlowDistanceY(), point1.z);

						vector<Point3f> objectPoints; vector<Point2f> imagePoints;
						objectPoints.push_back(point3);
						projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);
						fvec(i) = sqrt(pow((flowHead.x - imagePoints[0].x), 2) + pow((flowHead.y - imagePoints[0].y), 2));

					}
				}
		}
		return 0;
	}

	// Compute the jacobian of the errors
	int df(const VectorXf &x, MatrixXf &fjac) const
	{
		// 'x' has dimensions n x 1
		// It contains the current estimates for the parameters.

		// 'fjac' has dimensions m x n
		// It will contain the jacobian of the errors, calculated numerically in this case.

		float epsilon;
		epsilon = 1e-5f;

		for (int i = 0; i < x.size(); i++) 
		{
			VectorXf xPlus(x);
			xPlus(i) += epsilon;
			VectorXf xMinus(x);
			xMinus(i) -= epsilon;

			VectorXf fvecPlus(values());
			operator()(xPlus, fvecPlus);

			VectorXf fvecMinus(values());
			operator()(xMinus, fvecMinus);

			VectorXf fvecDiff(values());
			fvecDiff = (fvecPlus - fvecMinus) / (2.0f * epsilon);

			fjac.block(0, i, values(), 1) = fvecDiff;
		}

		return 0;
	}

	// Number of data points, i.e. values.
	int m;

	// Returns 'm', the number of values.
	int values() const { return m; }

	// The number of parameters, i.e. inputs.
	int n;

	// Returns 'n', the number of inputs.
	int inputs() const { return n; }

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Track::Track(const Blob &blob)
{
	beingTracked_ = true;
	trackUpdated_ = false;
	trackColor_ = Scalar(rand() % 256, rand() % 256, rand() % 256);

	blobs_.push_back(blob);

	vector<Point2f> flowTails = blob.getFlowTails();
	vector<Point2f> flowHeads = blob.getFlowHeads();

	int m = flowTails.size();
	MatrixPoint2f observedFlows(m, 2);

	for (int i = 0; i < m; i++)
	{
		observedFlows(i, 0) = flowTails[i];
		observedFlows(i, 1) = flowHeads[i];
	}


	int n = 4;
	VectorXf parameters(n);
	parameters(0) = 5;
	parameters(1) = 2;
	parameters(2) = 1.5;
	parameters(3) = blob.getAngleOfMotion();

	Point3f point = findWorldPoint(blob.getBottomLeftCorner(), 0.0, cameraMatrix, rotationMatrix, translationVector);

	Cuboid cuboid1(point, parameters(0), parameters(1), parameters(2), parameters(3));

	LMFunctor functor;
	functor.blob = blob;
	functor.cuboid = cuboid1;
	functor.observedFlows = observedFlows;
	functor.m = m;
	functor.n = n;


	LevenbergMarquardt<LMFunctor, float> lm(functor);
	int status = lm.minimize(parameters);
	cout << "LM optimization status: " << status << endl;

	//
	// Results
	// The 'x' vector also contains the results of the optimization.
	//
	cout << "Optimization results" << endl;
	cout << "\tOptimizex Cuboid Length: " << parameters(0) << endl;
	cout << "\tOptimized Cuboid Width: " << parameters(1) << endl;
	cout << "\tOptimized Cuboid Height: " << parameters(2) << endl;
	cout << "\tOptimized Cuboid Angle Of Motion: " << parameters(3) << endl;

	cuboidLength = parameters(0);
	cuboidWidth = parameters(1);
	cuboidHeight = parameters(2);
	cuboidAngleOfMotion = parameters(3);

	Cuboid cuboid2(point, cuboidLength, cuboidWidth, cuboidHeight, cuboidAngleOfMotion);
	cuboids_.push_back(cuboid2);


}
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Track::add(const Blob &blob)
{
	blobs_.push_back(blob);

	blobs_.push_back(blob);

	vector<Point2f> flowTails = blob.getFlowTails();
	vector<Point2f> flowHeads = blob.getFlowHeads();

	int m = flowTails.size();
	MatrixPoint2f observedFlows(m, 2);

	for (int i = 0; i < m; i++)
	{
		observedFlows(i, 0) = flowTails[i];
		observedFlows(i, 1) = flowHeads[i];
	}


	int n = 4;
	VectorXf parameters(n);
	parameters(0) = cuboidLength;
	parameters(1) = cuboidWidth;
	parameters(2) = cuboidHeight;
	parameters(3) = cuboidAngleOfMotion;

	Point3f point(getPrevCuboid().getB1().x + blob.getAverageFlowDistanceX(), getPrevCuboid().getB1().y + blob.getAverageFlowDistanceY(), 0.0);

	Cuboid cuboid1(point, parameters(0), parameters(1), parameters(2), parameters(3));
	
	LMFunctor functor;
	functor.blob = blob;
	functor.cuboid = cuboid1;
	functor.observedFlows = observedFlows;
	functor.m = m;
	functor.n = n;


	LevenbergMarquardt<LMFunctor, float> lm(functor);
	int status = lm.minimize(parameters);
	cout << "LM optimization status: " << status << endl;

	//
	// Results
	// The 'x' vector also contains the results of the optimization.
	//
	cout << "Optimization results" << endl;
	cout << "\tOptimizex Cuboid Length: " << parameters(0) << endl;
	cout << "\tOptimized Cuboid Width: " << parameters(1) << endl;
	cout << "\tOptimized Cuboid Height: " << parameters(2) << endl;
	cout << "\tOptimized Cuboid Angle Of Motion: " << parameters(3) << endl;

	cuboidLength = parameters(0);
	cuboidWidth = parameters(1);
	cuboidHeight = parameters(2);
	cuboidAngleOfMotion = parameters(3);

	Cuboid cuboid2(point, cuboidLength, cuboidWidth, cuboidHeight, cuboidAngleOfMotion);
	cuboids_.push_back(cuboid2);

	
	Point3f centroid1 = cuboids_.rbegin()[0].getCentroid();
	Point3f centroid2 = cuboids_.rbegin()[1].getCentroid();
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

void Track::drawBlobTrail(Mat &outputFrame, int trailLength, Scalar trailColor, int trailThickness)
{
	for (int i = 0; i < min((int)blobs_.size(), trailLength); i++)
	{
		line(outputFrame, blobs_.rbegin()[i].getCenter(), blobs_.rbegin()[i + 1].getCenter(), trailColor, trailThickness, CV_AA);
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Track::drawCuboidTrail(Mat &outputFrame, int trailLength, Scalar trailColor, int trailThickness)
{

	for (int i = 0; i < min((int)cuboids_.size(), trailLength); i++)
	{
		bool inFrame = true;

		if (cuboids_.rbegin()[i].getProjectedCentroid().x > outputFrame.cols || cuboids_.rbegin()[i].getProjectedCentroid().y > outputFrame.rows)
		{
			inFrame = false;
		}
		if (inFrame)
		{
			line(outputFrame, cuboids_.rbegin()[i].getProjectedCentroid(), cuboids_.rbegin()[i + 1].getProjectedCentroid(), trailColor, trailThickness, CV_AA);
		}
	}
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Track::drawTrackInfoOnBlobs(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness)
{
	Blob blob = getPrevBlob();

	rectangle(outputFrame, blob.getTopRightCorner(), Point(blob.getTopRightCorner().x + blob.getWidth(), blob.getTopRightCorner().y - blob.getDiagonalSize() / 9), backgroundColor, -1, CV_AA);

	putText(outputFrame, "Track: " + to_string(trackNumber_), Point(blob.getTopRightCorner().x + 3, blob.getTopRightCorner().y - 3), fontFace, blob.getWidth() / 250, fontColor, fontThickness, CV_AA);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Track::drawTrackInfoOnCuboids(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness)
{

	Cuboid cuboid = getPrevCuboid();
	vector<vector<Point2f>> measuredFlowHeads = cuboid.getMeasuredOpticalFlowHeads();

	//cout << measuredFlowHeads.size() << endl;

	for (int i = 0; i < measuredFlowHeads.size(); i++)
	{
		for (int j = 0; j < measuredFlowHeads[i].size(); j++)
		{
			circle(outputFrame, measuredFlowHeads[i][j], 2, RED, -1, CV_AA);
		}
	}

	Point point = Point((cuboid.getProjectedVertices()[0].x + cuboid.getProjectedVertices()[1].x) / 2, (cuboid.getProjectedVertices()[0].y + cuboid.getProjectedVertices()[0].y) / 2);

	circle(outputFrame, point, 10, trackColor_, -1, CV_AA);

	putText(outputFrame, to_string(trackNumber_), Point2f(point.x - 7, point.y + 3), fontFace, cuboid.getWidth() / 6, fontColor, fontThickness, CV_AA);

	putText(outputFrame, "I.S.: " + to_string(instantaneousSpeeds_.back()) + " kmph", Point2f(cuboid.getProjectedVertices()[0].x + 3, cuboid.getProjectedVertices()[0].y + 23), fontFace, cuboid.getWidth() / 6, fontColor, fontThickness, CV_AA);

	putText(outputFrame, "A.S.: " + to_string(findAverageSpeed()) + " kmph", Point2f(cuboid.getProjectedVertices()[0].x + 3, cuboid.getProjectedVertices()[0].y + 33), fontFace, cuboid.getWidth() / 6, fontColor, fontThickness, CV_AA);
}


Track::~Track() {};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

static vector<vector<float> > calculateErrors(Cuboid cuboid, Blob blob, Point3f cameraCenter)
{
	vector<vector<float> > vovErrors;
	vector<vector<Point2f>> vovMeasuredOpticalFlowHeads;

	vector<vector<float> > planeParameters = cuboid.getPlaneParameters();
	vector<vector<Point3f> > planeVertices = cuboid.getPlaneVertices();

	vector<Point2f> flowTails = blob.getFlowTails();
	vector<Point2f> flowHeads = blob.getFlowHeads();

	vector<Point3f> groundPlaneFlowTails = blob.getGroundPlaneFlowTails();
	vector<Point3f> groundPlaneFlowHeads = blob.getGroundPlaneFlowHeads();

	
	for (int i = 0; i < planeParameters.size(); i++)
	{
		vector<float> vErrors;
		vector<Point2f> vMeasuredOpticalFlowHeads;
		for (int j = 0; j < groundPlaneFlowTails.size(); j++)
		{
			
			float t1 = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlaneFlowTails[j].x - cameraCenter.x) + planeParameters[i][1] * (groundPlaneFlowTails[j].y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);

			Point3f point1 = Point3f(cameraCenter.x + (groundPlaneFlowTails[j].x - cameraCenter.x) * t1, cameraCenter.y + (groundPlaneFlowTails[j].y - cameraCenter.y) * t1, cameraCenter.z - cameraCenter.z * t1);

			float t2 = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlaneFlowHeads[j].x - cameraCenter.x) + planeParameters[i][1] * (groundPlaneFlowHeads[j].y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);

			Point3f point2 = Point3f(cameraCenter.x + (groundPlaneFlowHeads[j].x - cameraCenter.x) * t2, cameraCenter.y + (groundPlaneFlowHeads[j].y - cameraCenter.y) * t2, cameraCenter.z - cameraCenter.z * t2);

			bool inside = pointInside(planeVertices[i], point1);

			if (inside)
			{

				Point3f point3 = Point3f(point1.x + blob.getAverageFlowDistanceX(), point1.y + blob.getAverageFlowDistanceY(), point1.z);

				vector<Point3f> objectPoints; vector<Point2f> imagePoints;
				objectPoints.push_back(point3);
				projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);
				vMeasuredOpticalFlowHeads.push_back(imagePoints[0]);
				

				float error = sqrt(pow((flowHeads[j].x - imagePoints[0].x), 2) + pow((flowHeads[j].y - imagePoints[0].y), 2));
				vErrors.push_back(error);

			}
		}
		vovMeasuredOpticalFlowHeads.push_back(vMeasuredOpticalFlowHeads);
		vovErrors.push_back(vErrors);
	}
	cuboid.setMeasuredOpticalFlowHeads(vovMeasuredOpticalFlowHeads);
	cout << cuboid.getMeasuredOpticalFlowHeads()[0] << endl;
	return vovErrors;
}

static vector<vector<float> > calculateErrors2(Cuboid cuboid, Blob blob, Point3f cameraCenter)
{
	vector<vector<float> > vovErrors;
	vector<vector<Point2f>> vovMeasuredOpticalFlowHeads;

	vector<vector<float> > planeParameters = cuboid.getPlaneParameters();
	vector<vector<Point3f> > planeVertices = cuboid.getPlaneVertices();

	vector<Point2f> flowTails = blob.getFlowTails();
	vector<Point2f> flowHeads = blob.getFlowHeads();

	vector<Point3f> groundPlaneFlowTails = blob.getGroundPlaneFlowTails();
	vector<Point3f> groundPlaneFlowHeads = blob.getGroundPlaneFlowHeads();


	for (int i = 0; i < planeParameters.size(); i++)
	{
		vector<float> vErrors;
		vector<Point2f> vMeasuredOpticalFlowHeads;
		for (int j = 0; j < groundPlaneFlowTails.size(); j++)
		{

			float t1 = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlaneFlowTails[j].x - cameraCenter.x) + planeParameters[i][1] * (groundPlaneFlowTails[j].y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);

			Point3f point1 = Point3f(cameraCenter.x + (groundPlaneFlowTails[j].x - cameraCenter.x) * t1, cameraCenter.y + (groundPlaneFlowTails[j].y - cameraCenter.y) * t1, cameraCenter.z - cameraCenter.z * t1);

			float t2 = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlaneFlowHeads[j].x - cameraCenter.x) + planeParameters[i][1] * (groundPlaneFlowHeads[j].y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);

			Point3f point2 = Point3f(cameraCenter.x + (groundPlaneFlowHeads[j].x - cameraCenter.x) * t2, cameraCenter.y + (groundPlaneFlowHeads[j].y - cameraCenter.y) * t2, cameraCenter.z - cameraCenter.z * t2);

			bool inside = pointInside(planeVertices[i], point1);

			if (inside)
			{

				Point3f point3 = Point3f(point1.x + blob.getAverageFlowDistanceX(), point1.y + blob.getAverageFlowDistanceY(), point1.z);

				vector<Point3f> objectPoints; vector<Point2f> imagePoints;
				objectPoints.push_back(point3);
				projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);
				vMeasuredOpticalFlowHeads.push_back(imagePoints[0]);


				float error = sqrt(pow((flowHeads[j].x - imagePoints[0].x), 2) + pow((flowHeads[j].y - imagePoints[0].y), 2));
				vErrors.push_back(error);

			}
		}
		vovMeasuredOpticalFlowHeads.push_back(vMeasuredOpticalFlowHeads);
		vovErrors.push_back(vErrors);
	}
	cuboid.setMeasuredOpticalFlowHeads(vovMeasuredOpticalFlowHeads);
	cout << cuboid.getMeasuredOpticalFlowHeads()[0] << endl;
	return vovErrors;
}
