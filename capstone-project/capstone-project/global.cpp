#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"

using namespace cv;
using namespace std;

Mat cameraMatrix, distCoeffs, rotationVector, translationVector, rotationMatrix, inverseHomographyMatrix;

float frameRate, videoTimeElapsed, realTimeElapsed, totalFrameCount, frameHeight, frameWidth, fourCC;

int currentFrameCount;

const Scalar WHITE = Scalar(255, 255, 255), BLACK = Scalar(0, 0, 0), BLUE = Scalar(255, 0, 0), GREEN = Scalar(0, 255, 0), RED = Scalar(0, 0, 255), YELLOW = Scalar(0, 255, 255);

Mat currentFrame, nextFrame, currentFrame_gray, nextFrame_gray, currentFrame_blur, nextFrame_blur, morph, diff, thresh, videoMask, imgCuboids, imgTracks;

const Point3f cameraCenter = Point3f(1.80915, -8.95743, 8.52165);

const float initialCuboidLength = 5, initialCuboidWidth = 2, initialCuboidHeight = 1.5;

Point3f findWorldPoint(const Point2f &imagePoint, double zConst, const Mat &cameraMatrix, const Mat &rotationMatrix, const Mat &translationVector)
{
	Mat imagePointHV = Mat::ones(3, 1, DataType<double>::type);
	imagePointHV.at<double>(0, 0) = imagePoint.x;
	imagePointHV.at<double>(1, 0) = imagePoint.y;

	Mat A, B;

	A = rotationMatrix.inv() * cameraMatrix.inv() * imagePointHV;
	B = rotationMatrix.inv() * translationVector;

	double p = A.at<double>(2, 0);
	double q = zConst + B.at<double>(2, 0);
	double s = q / p;

	Mat worldPointHV = rotationMatrix.inv() * (s * cameraMatrix.inv() * imagePointHV - translationVector);

	Point3f worldPoint;
	worldPoint.x = worldPointHV.at<double>(0, 0);
	worldPoint.y = worldPointHV.at<double>(1, 0);
	worldPoint.z = 0.0;

	return worldPoint;
}
float distanceBetweenPoints(Point2f point1, Point2f point2)
{
	float x = abs(point1.x - point2.x);
	float y = abs(point1.y - point2.y);

	return(sqrt(pow(x, 2) + pow(y, 2)));
}
float distanceBetweenPoints(Point2f point1, Point point2)
{
	float x = abs(point1.x - point2.x);
	float y = abs(point1.y - point2.y);

	return(sqrt(pow(x, 2) + pow(y, 2)));
}

float distanceBetweenPoints(Point3f point1, Point3f point2)
{
	float x = abs(point1.x - point2.x);
	float y = abs(point1.y - point2.y);
	float z = abs(point1.z - point2.z);

	return(sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)));
}

bool pointInside(vector<Point3f> points, Point3f point)
{
	Point3f p1, p2, p3, p4, m;
	p1 = points[0];
	p2 = points[1];
	p3 = points[2];
	p4 = points[3];
	m = point;

	Vec3f v1(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
	Vec3f v3(p4.x - p3.x, p4.y - p3.y, p4.z - p3.z);
	Vec3f v4(m.x - p1.x, m.y - p1.y, m.z - p1.z);
	Vec3f v5(m.x - p3.x, m.y - p3.y, m.z - p3.z);
	Vec3f v1_norm(v1[0] / (sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])), v1[1] / (sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])), v1[2] / (sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])));

	Vec3f v3_norm(v3[0] / (sqrt(v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])), v3[1] / (sqrt(v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])), v3[2] / (sqrt(v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])));

	Vec3f v4_norm(v4[0] / (sqrt(v4[0] * v4[0] + v4[1] * v4[1] + v4[2] * v4[2])), v4[1] / (sqrt(v4[0] * v4[0] + v4[1] * v4[1] + v4[2] * v4[2])), v4[2] / (sqrt(v4[0] * v4[0] + v4[1] * v4[1] + v4[2] * v4[2])));

	Vec3f v5_norm(v5[0] / (sqrt(v5[0] * v5[0] + v5[1] * v5[1] + v5[2] * v5[2])), v5[1] / (sqrt(v5[0] * v5[0] + v5[1] * v5[1] + v5[2] * v5[2])), v5[2] / (sqrt(v5[0] * v5[0] + v5[1] * v5[1] + v5[2] * v5[2])));

	if (v1.dot(v4) >= 0 && v3.dot(v5) >= 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

