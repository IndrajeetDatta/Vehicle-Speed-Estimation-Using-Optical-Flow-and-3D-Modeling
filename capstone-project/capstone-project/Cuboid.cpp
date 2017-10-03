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

	projectPoints(vertices_, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePlaneProjectedVertices_);

	vector<Point3f> objectPoints;

	vector<Point2f> imagePoints;

	objectPoints.push_back(centroid_);

	projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

	projectedCentroid_ = imagePoints[0];

	vector<float> frontPlaneParameters;
	frontPlaneParameters.push_back(-cos(angleOfMotion_));
	frontPlaneParameters.push_back(sin(angleOfMotion_));
	frontPlaneParameters.push_back(0);
	frontPlaneParameters.push_back((length_ / 2) + centroid_.x * cos(angleOfMotion_) - centroid_.y * sin(angleOfMotion_));

	planeParameters_.push_back(frontPlaneParameters);

	vector<float> rightPlaneParameters;
	rightPlaneParameters.push_back(cos(angleOfMotion_));
	rightPlaneParameters.push_back(sin(angleOfMotion_));
	rightPlaneParameters.push_back(0);
	rightPlaneParameters.push_back((width_) / 2 - centroid_.x * cos(angleOfMotion_) - centroid_.y * sin(angleOfMotion_));

	planeParameters_.push_back(rightPlaneParameters);

	vector<float> backPlaneParameters;
	backPlaneParameters.push_back(sin(angleOfMotion_));
	backPlaneParameters.push_back(-cos(angleOfMotion_));
	backPlaneParameters.push_back(0);
	backPlaneParameters.push_back((length_ / 2) - centroid_.x * sin(angleOfMotion_) + centroid_.y * cos(angleOfMotion_));

	planeParameters_.push_back(backPlaneParameters);

	vector<float> leftPlaneParameters;
	leftPlaneParameters.push_back(-cos(angleOfMotion_));
	leftPlaneParameters.push_back(-sin(angleOfMotion_));
	leftPlaneParameters.push_back(0);
	leftPlaneParameters.push_back((width_ / 2) + centroid_.x * cos(angleOfMotion_) + centroid_.y * sin(angleOfMotion_));

	planeParameters_.push_back(leftPlaneParameters);

	vector<float> topPlaneParameters;
	topPlaneParameters.push_back(0);
	topPlaneParameters.push_back(0);
	topPlaneParameters.push_back(-1);
	topPlaneParameters.push_back(centroid_.z + (height_ / 2));

	planeParameters_.push_back(topPlaneParameters);

	vector<float> bottomPlaneParameters;
	bottomPlaneParameters.push_back(0);
	bottomPlaneParameters.push_back(0);
	bottomPlaneParameters.push_back(1);
	bottomPlaneParameters.push_back(-(centroid_.z) + (height_ / 2));

	planeParameters_.push_back(bottomPlaneParameters);

}
Cuboid::Cuboid() {};
Cuboid::~Cuboid() {};

void Cuboid::drawCuboid(Mat &outputFrame, Scalar color, int lineThickness)
{
	bool inFrame = true;

	for (int i = 0; i < imagePlaneProjectedVertices_.size(); i++)
	{
		if (imagePlaneProjectedVertices_[i].x > outputFrame.cols || imagePlaneProjectedVertices_[i].y > outputFrame.rows)
		{
			inFrame = true;
		}

		if (inFrame)
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

			/*putText(outputFrame, "b1", imagePlaneProjectedVertices_[0], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "b2", imagePlaneProjectedVertices_[1], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "b3", imagePlaneProjectedVertices_[2], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "b4", imagePlaneProjectedVertices_[3], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "t1", imagePlaneProjectedVertices_[4], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "t2", imagePlaneProjectedVertices_[5], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "t3", imagePlaneProjectedVertices_[6], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "t4", imagePlaneProjectedVertices_[7], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);*/
		}
	}
}
