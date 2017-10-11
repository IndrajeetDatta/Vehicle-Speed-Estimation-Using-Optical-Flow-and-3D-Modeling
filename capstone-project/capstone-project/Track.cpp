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


typedef Matrix<Point2f, Dynamic, Dynamic> MatrixPoint2f;

struct LMFunctor
{
	MatrixPoint2f observedFlows;
	Cuboid cuboid;

	int operator()(const VectorXf &x, VectorXf &fvec) const
	{
		float param_length = x(0);
		float param_width = x(1);
		float param_height = x(2);
		float param_angleOfMotion = x(3);
		float param_motionX = x(4);
		float param_motionY = x(5);

		vector<vector<float> > planeParameters;

		vector<float> frontPlaneParameters;
		frontPlaneParameters.push_back(-cos(param_angleOfMotion));
		frontPlaneParameters.push_back(sin(param_angleOfMotion));
		frontPlaneParameters.push_back(0);
		frontPlaneParameters.push_back((param_length / 2) + cuboid.getCentroid().x * cos(param_angleOfMotion) - cuboid.getCentroid().y * sin(param_angleOfMotion));

		planeParameters.push_back(frontPlaneParameters);

		vector<float> rightPlaneParameters;
		rightPlaneParameters.push_back(cos(param_angleOfMotion));
		rightPlaneParameters.push_back(sin(param_angleOfMotion));
		rightPlaneParameters.push_back(0);
		rightPlaneParameters.push_back((param_width) / 2 - cuboid.getCentroid().x * cos(param_angleOfMotion) - cuboid.getCentroid().y * sin(param_angleOfMotion));

		planeParameters.push_back(rightPlaneParameters);

		vector<float> backPlaneParameters;
		backPlaneParameters.push_back(sin(param_angleOfMotion));
		backPlaneParameters.push_back(-cos(param_angleOfMotion));
		backPlaneParameters.push_back(0);
		backPlaneParameters.push_back((param_length / 2) - cuboid.getCentroid().x * sin(param_angleOfMotion) + cuboid.getCentroid().y * cos(param_angleOfMotion));

		planeParameters.push_back(backPlaneParameters);

		vector<float> leftPlaneParameters;
		leftPlaneParameters.push_back(-cos(param_angleOfMotion));
		leftPlaneParameters.push_back(-sin(param_angleOfMotion));
		leftPlaneParameters.push_back(0);
		leftPlaneParameters.push_back((param_width / 2) + cuboid.getCentroid().x * cos(param_angleOfMotion) + cuboid.getCentroid().y * sin(param_angleOfMotion));

		planeParameters.push_back(leftPlaneParameters);

		vector<float> topPlaneParameters;
		topPlaneParameters.push_back(0);
		topPlaneParameters.push_back(0);
		topPlaneParameters.push_back(-1);
		topPlaneParameters.push_back(cuboid.getCentroid().z + (param_height / 2));

		planeParameters.push_back(topPlaneParameters);

		vector<float> bottomPlaneParameters;
		bottomPlaneParameters.push_back(0);
		bottomPlaneParameters.push_back(0);
		bottomPlaneParameters.push_back(1);
		bottomPlaneParameters.push_back(-(cuboid.getCentroid().z) + (param_height / 2));

		planeParameters.push_back(bottomPlaneParameters);

		vector<vector<Point3f> > planeVertices = cuboid.getPlaneVertices();


		for (int i = 0; i < planeParameters.size(); i++)
		{

			for (int j = 0; j < values(); j++) 
			{

				Point2f flowTail = observedFlows(j, 0);
				Point2f flowHead = observedFlows(j, 1);


				Point3f groundPlaneFlowTail = findWorldPoint(flowTail, 0, cameraMatrix, rotationMatrix, translationVector);

				Point3f groundPlaneFlowHead = findWorldPoint(flowHead, 0, cameraMatrix, rotationMatrix, translationVector);

				Point3f groundPlanePredictedFlowHead = Point3f(groundPlaneFlowHead.x + param_motionX, groundPlaneFlowHead.y + param_motionY, 0);

				float t1 = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlaneFlowHead.x - cameraCenter.x) + planeParameters[i][1] * (groundPlaneFlowHead.y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);

				Point3f point1 = Point3f(cameraCenter.x + (groundPlaneFlowHead.x - cameraCenter.x) * t1, cameraCenter.y + (groundPlaneFlowHead.y - cameraCenter.y) * t1, cameraCenter.z - cameraCenter.z * t1);

				float t2 = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlanePredictedFlowHead.x - cameraCenter.x) + planeParameters[i][1] * (groundPlanePredictedFlowHead.y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);

				Point3f point2 = Point3f(cameraCenter.x + (groundPlanePredictedFlowHead.x - cameraCenter.x) * t2, cameraCenter.y + (groundPlanePredictedFlowHead.y - cameraCenter.y) * t2, cameraCenter.z - cameraCenter.z * t2);

				bool inside = pointInside(planeVertices[i], point1);

				if (inside)
				{
				
						vector<Point3f> objectPoints; vector<Point2f> imagePoints;
						objectPoints.push_back(point1);
						objectPoints.push_back(point2);
						projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);
						
						fvec(i) = sqrt(pow((imagePoints[0].x - imagePoints[1].x), 2) + pow((imagePoints[0].y - imagePoints[1].y), 2));
				}
			}
		}
		return 0;
	}

	int df(const VectorXf &x, MatrixXf &fjac) const
	{

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

	int m;

	int values() const { return m; }

	int n;

	int inputs() const { return n; }

};

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Track::Track(const Blob &blob)
{
	beingTracked_ = true; 
	trackUpdated_ = false; 
	trackColor_ = Scalar(rand() % 256, rand() % 256, rand() % 256);

	cuboidLength_ = 5;
	cuboidWidth_ = 2;
	cuboidHeight_ = 1.5;
	angleOfMotion_ = blob.getAngleOfMotion();
	motionX_ = blob.getAverageFlowDistanceX();
	motionY_ = blob.getAverageFlowDistanceY();



	int m = blob.getFlowTails().size();
	MatrixPoint2f observedFlows(m, 2);

	for (int i = 0; i < m; i++)
	{
		observedFlows(i, 0) = blob.getFlowTails()[i];
		observedFlows(i, 1) = blob.getFlowHeads()[i];
	}


	int n = 6;
	VectorXf parameters(n);
	parameters(0) = cuboidLength_;
	parameters(1) = cuboidWidth_;
	parameters(2) = cuboidHeight_;
	parameters(3) = angleOfMotion_;
	parameters(4) = motionX_;
	parameters(5) = motionY_;

	Point3f point1 = findWorldPoint(blob.getBottomLeftCorner(), 0.0, cameraMatrix, rotationMatrix, translationVector); 

	Cuboid cuboid1(point1, parameters(0), parameters(1), parameters(2), parameters(3)); 

	LMFunctor functor;
	functor.blob = blob;
	functor.cuboid = cuboid1;
	functor.observedFlows = observedFlows;
	functor.m = m;
	functor.n = n;


	LevenbergMarquardt<LMFunctor, float> lm(functor); 
	int status = lm.minimize(parameters);

	cout << "Optimization results" << endl;
	cout << "\t Optimizex Cuboid Length: " << parameters(0) << endl;
	cout << "\t Optimized Cuboid Width: " << parameters(1) << endl;
	cout << "\t Optimized Cuboid Height: " << parameters(2) << endl;
	cout << "\t Optimized Angle Of Motion: " << parameters(3) << endl;
	cout << "\t Optimized Motion in X-axis: " << parameters(3) << endl;
	cout << "\t Optimized Motion in Y-axis: " << parameters(4) << endl;

	cuboidLength_ = (parameters(0) > 0 && parameters(0) < cuboidLength_* 2) ? parameters(0) : cuboidLength_;

	cuboidWidth_ = (parameters(1) > 0 && parameters(1) < cuboidWidth_ * 2) ? parameters(1) : cuboidWidth_;

	cuboidHeight_ = (parameters(2) > 0 && parameters(2) < cuboidHeight_ * 2) ? parameters(2) : cuboidHeight_;
	angleOfMotion_ = parameters(3);
	motionX_ = (parameters(4) > 0 && parameters(4) < motionX_ * 2) ? parameters(4) : motionX_;
	motionY_ = (parameters(5) > 0 && parameters(5) < motionY_ * 2) ? parameters(5) : motionX_;

	/*Point3f point2 = findWorldPoint(Point2f(blobs_.back().getBottomLeftCorner().x + motionX_, blobs_.back().getBottomLeftCorner().y + motionY_), 0, cameraMatrix, rotationMatrix, translationVector);
*/
	Point3f point2 = findWorldPoint(blob.getBottomLeftCorner(), 0, cameraMatrix, rotationMatrix, translationVector);
	Cuboid cuboid2(point2, cuboidLength_, cuboidWidth_, cuboidHeight_, angleOfMotion_);
	cuboids_.push_back(cuboid2);
	blobs_.push_back(blob);


}
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Track::add(const Blob &blob)
{	


	int m = blob.getFlowTails.size();
	MatrixPoint2f observedFlows(m, 2);

	for (int i = 0; i < m; i++)
	{
		observedFlows(i, 0) = blob.getFlowTails()[i];
		observedFlows(i, 1) = blob.getFlowHeads()[i];
	}


	int n = 6;
	VectorXf parameters(n);
	parameters(0) = cuboidLength_;
	parameters(1) = cuboidWidth_;
	parameters(2) = cuboidHeight_;
	parameters(3) = blob.getAngleOfMotion();
	parameters(4) = motionX_;
	parameters(5) = motionY_;

	Point3f point1 = findWorldPoint(Point2f(blobs_.back().getBottomLeftCorner().x + motionX_, blobs_.back().getBottomLeftCorner().y + motionY_), 0.0, cameraMatrix, rotationMatrix, translationVector);

	Cuboid cuboid1(point1, parameters(0), parameters(1), parameters(2), parameters(3));
	
	LMFunctor functor;
	functor.blob = blob;
	functor.cuboid = cuboid1;
	functor.observedFlows = observedFlows;
	functor.m = m;
	functor.n = n;


	LevenbergMarquardt<LMFunctor, float> lm(functor);
	int status = lm.minimize(parameters);

	cout << "Optimization results" << endl;
	cout << "\t Initial Cuboid Length: " << cuboidLength_ << " Optimized Cuboid Length: " << parameters(0) << endl;
	cout << "\t Initial Cuboid Width: " << cuboidWidth_ << "  Optimized Cuboid Width: " << parameters(1) << endl;
	cout << "\t Initial Cuboid Height: " << cuboidHeight_ << " Optimized Cuboid Height: " << parameters(2) << endl;
	cout << "\t Initial Angle Of Motion: " << blob.getAngleOfMotion() << " Optimized Angle Of Motion: " << parameters(3) << endl;

	cout << "\t Initial Motion in X-axis: " << motionX_ << " Optimized Motion in X-axis: " << parameters(4) << endl;
	cout << "\t Initial Motion in Y-axis: " << motionY_ << " Optimized Motion in Y-axis: " << parameters(5) << endl;

	cuboidLength_ = (parameters(0) > 0 && parameters(0) < 5 * 2) ? parameters(0) : 5;
	cuboidWidth_ = (parameters(1) > 0 && parameters(1) < 2 * 2) ? parameters(1) : 2;
	cuboidHeight_ = (parameters(2) > 0 && parameters(2) < 1.5 * 2) ? parameters(2) : 1.5;
	angleOfMotion_ = parameters(3);
	motionX_ = (parameters(4) > 0 && parameters(4) < motionX_ * 2) ? parameters(4) : motionX_;
	motionY_ = (parameters(5) > 0 && parameters(5) < motionY_ * 2) ? parameters(5) : motionY_;

	Point3f point2 = findWorldPoint(Point2f(blobs_.back().getBottomLeftCorner().x + motionX_, blobs_.back().getBottomLeftCorner().y + motionY_), 0.0, cameraMatrix, rotationMatrix, translationVector);
	
	Cuboid cuboid2(point2, cuboidLength_, cuboidWidth_, cuboidHeight_, angleOfMotion_);
	cuboids_.push_back(cuboid2);
	blobs_.push_back(blob);
	
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

