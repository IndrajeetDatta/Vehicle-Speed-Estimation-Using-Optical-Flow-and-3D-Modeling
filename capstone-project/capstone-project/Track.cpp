#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"

#include "Track.h"
#include "Blob.h"
#include "Cuboid.h"


using namespace std;
using namespace cv;

Track::Track(const Blob &blob)
{
	blobs_.push_back(blob);

	Point3f point = findWorldPoint(blob.getBottomLeftCorner(), 0.0, cameraMatrix, rotationMatrix, translationVector);
	Cuboid cuboid(point, initialCuboidLength, initialCuboidWidth, initialCuboidHeight, 0.0);
	cuboids_.push_back(cuboid);

	beingTracked_ = true;
	trackUpdated_ = false;
	trackColor_ = Scalar(rand() % 256, rand() % 256, rand() % 256);
	vector<Point2f> flowHeads, flowTails;
	flowHeads = blob.getFlowHeads();
	flowTails = blob.getFlowTails();

	vector<Point3f> groundPlaneFlowHeads, groundPlaneFlowTails;
	for (int i = 0; i < flowTails.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(flowTails[i]), 0.0, cameraMatrix, rotationMatrix, translationVector);

		groundPlaneFlowTails.push_back(point);
	}
	for (int i = 0; i < flowHeads.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(flowHeads[i]), 0.0, cameraMatrix, rotationMatrix, translationVector);

		groundPlaneFlowHeads.push_back(point);
	}

	double totalDx = 0, totalDy = 0;

	for (int i = 0; i < groundPlaneFlowTails.size(); i++)
	{
		double dx = groundPlaneFlowHeads[i].x - groundPlaneFlowTails[i].x;

		double dy = groundPlaneFlowHeads[i].y - groundPlaneFlowTails[i].y;

		totalDx = totalDx + dx;

		totalDy = totalDy + dy;
	}

	averageFlowDistanceX_ = totalDx / groundPlaneFlowTails.size();
	averageFlowDistanceY_ = totalDy / groundPlaneFlowTails.size();
	angleOfMotion_ = atan(averageFlowDistanceX_ / averageFlowDistanceY_);



	planeProjectedFlowTails_ = findFlowsProjectedOnPlanes(cuboid.getPlaneParameters(), cuboid.getPlaneVertices(), groundPlaneFlowTails, cameraCenter);

	planeProjectedFlowHeads_ = findFlowsProjectedOnPlanes(cuboid.getPlaneParameters(), cuboid.getPlaneVertices(), groundPlaneFlowHeads, cameraCenter);

}
void Track::add(const Blob &blob)
{
	blobs_.push_back(blob);
	Point3f point2(getRecentCuboid().getB1().x + averageFlowDistanceX_, getRecentCuboid().getB1().y + averageFlowDistanceY_, 0.0);
	Cuboid cuboid(point2, initialCuboidLength, initialCuboidWidth, initialCuboidHeight, angleOfMotion_);
	cuboids_.push_back(cuboid);
	vector<Point2f> flowHeads, flowTails;

	vector<Point3f> groundPlaneFlowHeads, groundPlaneFlowTails;

	flowHeads = blob.getFlowHeads();

	flowTails = blob.getFlowTails();

	for (int i = 0; i < flowTails.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(flowTails[i]), 0.0, cameraMatrix, rotationMatrix, translationVector);

		groundPlaneFlowTails.push_back(point);
	}
	for (int i = 0; i < flowHeads.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(flowHeads[i]), 0.0, cameraMatrix, rotationMatrix, translationVector);

		groundPlaneFlowHeads.push_back(point);
	}

	double totalDx = 0, totalDy = 0;

	for (int i = 0; i < groundPlaneFlowTails.size(); i++)
	{
		double dx = groundPlaneFlowHeads[i].x - groundPlaneFlowTails[i].x;

		double dy = groundPlaneFlowHeads[i].y - groundPlaneFlowTails[i].y;

		totalDx = totalDx + dx;

		totalDy = totalDy + dy;
	}
	averageFlowDistanceX_ = totalDx / groundPlaneFlowTails.size();

	averageFlowDistanceY_ = totalDy / groundPlaneFlowTails.size();


	angleOfMotion_ = atan(averageFlowDistanceX_ / averageFlowDistanceY_);

	Point3f point = findWorldPoint(blob.getBottomLeftCorner(), 0.0, cameraMatrix, rotationMatrix, translationVector);





	vector<vector<Point3f> > planeProjectedFlowTails;

	planeProjectedFlowTails_ = findFlowsProjectedOnPlanes(cuboid.getPlaneParameters(), cuboid.getPlaneVertices(), groundPlaneFlowTails, cameraCenter);

	planeProjectedFlowHeads_ = findFlowsProjectedOnPlanes(cuboid.getPlaneParameters(), cuboid.getPlaneVertices(), groundPlaneFlowHeads, cameraCenter);

	vector<vector<float> >  errors;
	for (int i = 0; i < planeProjectedFlowTails_.size(); i++)
	{

		if (planeProjectedFlowTails_[i].size() > 0)
		{
			vector<float> temp;
			for (int j = 0; j < planeProjectedFlowTails_[i].size(); j++)
			{
				Point3f point(planeProjectedFlowTails_[i][j].x + averageFlowDistanceX_, planeProjectedFlowTails_[i][j].y + averageFlowDistanceY_, planeProjectedFlowTails_[i][j].z);

				float error_distance = distanceBetweenPoints(point, planeProjectedFlowHeads_[i][j]);
				cout << "Projected Flow Tail: " << planeProjectedFlowTails_[i][j] << endl;
				cout << "Projected Flow Tail After Motion: " << point << endl;
				cout << "Projected Flow Head: " << planeProjectedFlowHeads_[i][j] << endl;
				cout << "Error Distance: " << error_distance << endl;
				cout << endl; cout << endl;
				temp.push_back(error_distance);
			}
			errors.push_back(temp);
		}
	}
	instantaneousSpeed_.push_back(sqrt(pow(averageFlowDistanceX_, 2) + pow(averageFlowDistanceY_, 2)) / (1 / frameRate) *  3.6);

	double total = 0;

	for (int i = 0; i < instantaneousSpeed_.size(); i++)
	{
		total += instantaneousSpeed_[i];
	}

	averageSpeed_ = total / instantaneousSpeed_.size();

}



void Track::drawBlobTrail(Mat &outputFrame, int trailLength, Scalar trailColor, int trailThickness)
{
	for (int i = 0; i < min((int)blobs_.size(), trailLength); i++)
	{
		line(outputFrame, blobs_.rbegin()[i].getCenter(), blobs_.rbegin()[i + 1].getCenter(), trailColor, trailThickness, CV_AA);
	}
}
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
void Track::drawTrackInfoOnBlobs(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness)
{
	Blob blob = getRecentBlob();

	rectangle(outputFrame, blob.getTopRightCorner(), Point(blob.getTopRightCorner().x + blob.getWidth(), blob.getTopRightCorner().y - blob.getDiagonalSize() / 9), backgroundColor, -1, CV_AA);

	putText(outputFrame, "Track: " + to_string(trackNumber_), Point(blob.getTopRightCorner().x + 3, blob.getTopRightCorner().y - 3), fontFace, blob.getWidth() / 250, fontColor, fontThickness, CV_AA);
}
void Track::drawTrackInfoOnCuboids(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness)
{

	Cuboid cuboid = getRecentCuboid();

	Point point = Point((cuboid.getProjectedVertices()[0].x + cuboid.getProjectedVertices()[1].x) / 2, (cuboid.getProjectedVertices()[0].y + cuboid.getProjectedVertices()[0].y) / 2);

	circle(outputFrame, point, 10, trackColor_, -1, CV_AA);

	putText(outputFrame, to_string(trackNumber_), Point2f(point.x - 7, point.y + 3), fontFace, cuboid.getWidth() / 6, fontColor, fontThickness, CV_AA);

	putText(outputFrame, "I.S.: " + to_string(instantaneousSpeed_.back()) + " kmph", Point2f(cuboid.getProjectedVertices()[0].x + 3, cuboid.getProjectedVertices()[0].y + 23), fontFace, cuboid.getWidth() / 6, fontColor, fontThickness, CV_AA);

	putText(outputFrame, "A.S.: " + to_string(averageSpeed_) + " kmph", Point2f(cuboid.getProjectedVertices()[0].x + 3, cuboid.getProjectedVertices()[0].y + 33), fontFace, cuboid.getWidth() / 6, fontColor, fontThickness, CV_AA);
}
Track::~Track() {};