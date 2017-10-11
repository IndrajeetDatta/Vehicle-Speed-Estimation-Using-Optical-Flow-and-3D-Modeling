#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"
#include "Cuboid.h"

using namespace std;
using namespace cv;

Cuboid::Cuboid(Point3f &point, float initialLength, float intialWidth, float height, float angleOfMotion)
{
	length_ = initialLength;
	
	width_ = intialWidth;
	
	height_ = height;
	
	angleOfMotion_ = angleOfMotion;

	b1_ = Point3f(point.x, point.y, 0.0);

	b2_ = Point3f(point.x + (width_ * cos(angleOfMotion_)), point.y - (width_ * sin(angleOfMotion_)), 0.0);

	b3_ = Point3f(point.x + (length_ * sin(angleOfMotion_)), point.y + (length_ * cos(angleOfMotion_)), 0.0);

	b4_ = Point3f(point.x + (length_ * sin(angleOfMotion_)) + (width_ * cos(angleOfMotion_)), point.y + (length_ * cos(angleOfMotion_)) + (width_ * sin(angleOfMotion_)), 0.0);

	t1_ = Point3f(point.x, point.y, height_);

	t2_ = Point3f(point.x + (width_ * cos(angleOfMotion_)), point.y - (width_* sin(angleOfMotion_)), height_);

	t3_ = Point3f(point.x + (length_ * sin(angleOfMotion_)), point.y + (length_ * cos(angleOfMotion_)), height_);

	t4_ = Point3f(point.x + (length_ * sin(angleOfMotion_)) + (width_ * cos(angleOfMotion_)), point.y + (length_ * cos(angleOfMotion_)) + (width_ * sin(angleOfMotion_)), height_);

	vertices_.push_back(b1_);
	vertices_.push_back(b2_);
	vertices_.push_back(b3_);
	vertices_.push_back(b4_);
	vertices_.push_back(t1_);
	vertices_.push_back(t2_);
	vertices_.push_back(t3_);
	vertices_.push_back(t4_);

	vector<Point3f> frontPlaneVertices;
	frontPlaneVertices.push_back(b1_);
	frontPlaneVertices.push_back(t1_);
	frontPlaneVertices.push_back(t2_);
	frontPlaneVertices.push_back(b2_);

	planeVertices_.push_back(frontPlaneVertices);

	vector<Point3f> rightPlaneVertices;
	rightPlaneVertices.push_back(b3_);
	rightPlaneVertices.push_back(t3_);
	rightPlaneVertices.push_back(t1_);
	rightPlaneVertices.push_back(b1_);

	planeVertices_.push_back(rightPlaneVertices);

	vector<Point3f> backPlaneVertices;
	backPlaneVertices.push_back(b4_);
	backPlaneVertices.push_back(t4_);
	backPlaneVertices.push_back(t3_);
	backPlaneVertices.push_back(b3_);

	planeVertices_.push_back(backPlaneVertices);

	vector<Point3f> leftPlaneVertices;
	leftPlaneVertices.push_back(b2_);
	leftPlaneVertices.push_back(t2_);
	leftPlaneVertices.push_back(t4_);
	leftPlaneVertices.push_back(b4_);

	planeVertices_.push_back(leftPlaneVertices);

	vector<Point3f> topPlaneVertices;
	topPlaneVertices.push_back(t1_);
	topPlaneVertices.push_back(t3_);
	topPlaneVertices.push_back(t4_);
	topPlaneVertices.push_back(t2_);

	planeVertices_.push_back(topPlaneVertices);

	vector<Point3f> bottomPlaneVertices;
	bottomPlaneVertices.push_back(b1_);
	bottomPlaneVertices.push_back(b3_);
	bottomPlaneVertices.push_back(b4_);
	bottomPlaneVertices.push_back(b2_);

	planeVertices_.push_back(bottomPlaneVertices);

	centroid_ = Point3f((b1_.x + b2_.x) / 2, (b1_.y + b3_.y) / 2, height / 2);

	

	vector<Point3f> objectPoints;

	vector<Point2f> imagePoints;

	objectPoints.push_back(centroid_);

	projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

	projectedCentroid_ = imagePoints[0];

}
Cuboid::Cuboid() {};
Cuboid::~Cuboid() {};

void Cuboid::drawCuboid(Mat &outputFrame, Scalar color, int lineThickness)
{
	projectPoints(vertices_, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePlaneProjectedVertices_);

	if (imagePlaneProjectedVertices_[0].x < outputFrame.cols && imagePlaneProjectedVertices_[0].y < outputFrame.rows)
	{
		line(outputFrame, imagePlaneProjectedVertices_[0], imagePlaneProjectedVertices_[1], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[1], imagePlaneProjectedVertices_[3], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[3], imagePlaneProjectedVertices_[2], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[2], imagePlaneProjectedVertices_[0], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[4], imagePlaneProjectedVertices_[5], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[5], imagePlaneProjectedVertices_[7], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[7], imagePlaneProjectedVertices_[6], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[6], imagePlaneProjectedVertices_[4], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[0], imagePlaneProjectedVertices_[4], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[1], imagePlaneProjectedVertices_[5], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[2], imagePlaneProjectedVertices_[6], color, lineThickness, CV_AA);

		line(outputFrame, imagePlaneProjectedVertices_[3], imagePlaneProjectedVertices_[7], color, lineThickness, CV_AA);

		putText(outputFrame, "Length: " + to_string(length_), Point2f(imagePlaneProjectedVertices_[0].x, imagePlaneProjectedVertices_[0].y + 10), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

		putText(outputFrame, "Width: " + to_string(width_), Point2f(imagePlaneProjectedVertices_[0].x, imagePlaneProjectedVertices_[0].y + 30), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

		putText(outputFrame, "Height: " + to_string(height_), Point2f(imagePlaneProjectedVertices_[0].x, imagePlaneProjectedVertices_[0].y + 50), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);

		putText(outputFrame, "Angle: " + to_string(angleOfMotion_), Point2f(imagePlaneProjectedVertices_[0].x, imagePlaneProjectedVertices_[0].y + 70), CV_FONT_HERSHEY_SIMPLEX, 0.3, WHITE, 1, CV_AA);
	}
}

