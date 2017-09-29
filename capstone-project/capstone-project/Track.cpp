#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"

#include "Track.h"
#include "Blob.h"
#include "Cuboid.h"


using namespace std;
using namespace cv;

static vector<vector<float> > calculateErrors(Cuboid cuboid, Blob blob, Point3f cameraCenter);


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Track::Track(const Blob &blob)
{
	blobs_.push_back(blob);

	Point3f point = findWorldPoint(blob.getBottomLeftCorner(), 0.0, cameraMatrix, rotationMatrix, translationVector);
	Cuboid cuboid(point, initialCuboidLength, initialCuboidWidth, initialCuboidHeight, 0.0);
	cuboids_.push_back(cuboid);

	beingTracked_ = true;
	trackUpdated_ = false;
	trackColor_ = Scalar(rand() % 256, rand() % 256, rand() % 256);
}
	
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void Track::add(const Blob &blob)
{
	blobs_.push_back(blob);

	Point3f point(getPrevCuboid().getB1().x + blob.getAverageFlowDistanceX(), getPrevCuboid().getB1().y + blob.getAverageFlowDistanceY(), 0.0);

	Cuboid cuboid(point, initialCuboidLength, initialCuboidWidth, initialCuboidHeight, blob.getAngleOfMotion());
	cuboids_.push_back(cuboid);
	

	calculateErrors(cuboid, blob, cameraCenter);
	
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
			circle(outputFrame, measuredFlowHeads[i][j], 1, RED, -1, CV_AA);
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
	return vovErrors;
}

