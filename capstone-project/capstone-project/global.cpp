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
FileStorage optimizationData("optimizationData.yml", FileStorage::WRITE);

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

bool pointInsideRect(vector<Point3f> points, Point3f point)
{

	Point3f p1, p2, p3, p4, m;

	p1 = points[0];
	p2 = points[1];
	p3 = points[2];
	p4 = points[3];
	m = point;

	Vec3f v12(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
	Vec3f v23(p3.x - p2.x, p3.y - p2.y, p3.z - p2.z);
	Vec3f v34(p4.x - p3.x, p4.y - p3.y, p4.z - p3.z);
	Vec3f v41(p1.x - p4.x, p1.y - p4.y, p1.z - p4.z);

	Vec3f v1m(m.x - p1.x, m.y - p1.y, m.z - p1.z);
	Vec3f v2m(m.x - p2.x, m.y - p2.y, m.z - p2.z);
	Vec3f v3m(m.x - p3.x, m.y - p3.y, m.z - p3.z);
	Vec3f v4m(m.x - p4.x, m.y - p4.y, m.z - p4.z);

	float v12_length = sqrt(pow(v12[0], 2) + pow(v12[1], 2) + pow(v12[2], 2));
	float v23_length = sqrt(pow(v23[0], 2) + pow(v23[1], 2) + pow(v23[2], 2));
	float v34_length = sqrt(pow(v34[0], 2) + pow(v34[1], 2) + pow(v34[2], 2));
	float v41_length = sqrt(pow(v41[0], 2) + pow(v41[1], 2) + pow(v41[2], 2));

	float v1m_length = sqrt(pow(v1m[0], 2) + pow(v1m[1], 2) + pow(v1m[2], 2));
	float v2m_length = sqrt(pow(v2m[0], 2) + pow(v2m[1], 2) + pow(v2m[2], 2));
	float v3m_length = sqrt(pow(v3m[0], 2) + pow(v3m[1], 2) + pow(v3m[2], 2));

	float v4m_length = sqrt(pow(v4m[0], 2) + pow(v4m[1], 2) + pow(v4m[2], 2));

	Vec3f v12_norm = v12 / v12_length;
	Vec3f v23_norm = v23 / v23_length;
	Vec3f v34_norm = v34 / v34_length;
	Vec3f v41_norm = v41 / v41_length;

	Vec3f v1m_norm = v1m / v1m_length;
	Vec3f v2m_norm = v2m / v2m_length;
	Vec3f v3m_norm = v3m / v3m_length;
	Vec3f v4m_norm = v4m / v4m_length;
	
	float dot1 = v12_norm.dot(v1m_norm);
	float dot2 = v23_norm.dot(v2m_norm);
	float dot3 = v34_norm.dot(v3m_norm);
	float dot4 = v41_norm.dot(v4m_norm);

	if (dot1 >= 0 && dot2 >= 0 && dot3 >= 0 && dot4 >= 0)
	{
		return true;
	}

	else
	{
		return false;
	}

}
float findMedian(vector<float> values)
{
	int m = values.size();

	sort(values.begin(), values.end());

	float medianValue;

	if (m % 2 == 0)
	{
		medianValue = (values[(m / 2) - 1] + values[m / 2]) / 2;
	}
	else
	{
		medianValue = values[(m + 1) / 2];
	}

	return medianValue;
}
void eliminateOutliers(vector<Point2f> flowTails, vector<Point2f> flowHeads)
{
	vector<float> v_flowLengths;
	int m = flowTails.size();

	for (int i = 0; i < m; i++)
	{
		float flowLength = sqrt(pow(flowHeads[i].x - flowTails[i].x, 2) + pow(flowHeads[i].y - flowTails[i].y, 2));
		v_flowLengths.push_back(flowLength);
	}

	sort(v_flowLengths.begin(), v_flowLengths.end());

	float medianFlowLength = findMedian(v_flowLengths);

	vector<int> indexes;
	for (int i = 0; i < m; i++)
	{
		float diff = abs(medianFlowLength - v_flowLengths[i]);
		if (diff > 0.3 * v_flowLengths[i])
		{
			indexes.push_back(i);
		}
	}
	for (int i = 0; i < indexes.size(); i++)
	{
		flowTails.erase(flowTails.begin() + (indexes[i] - i));
		flowHeads.erase(flowHeads.begin() + (indexes[i] - i));
	}
}

void eliminateOutliers(vector<Point3f> groundPlaneFlowTails, vector<Point3f> groundPlaneFlowHeads)
{
	vector<float> v_flowLengths;
	int m = groundPlaneFlowTails.size();

	for (int i = 0; i < m; i++)
	{
		float flowLength = sqrt(pow(groundPlaneFlowHeads[i].x - groundPlaneFlowTails[i].x, 2) + pow(groundPlaneFlowHeads[i].y - groundPlaneFlowTails[i].y, 2));
		v_flowLengths.push_back(flowLength);
	}

	float medianFlowLength = findMedian(v_flowLengths);

	vector<int> indexes;
	for (int i = 0; i < m; i++)
	{
		float diff = abs(medianFlowLength - v_flowLengths[i]);
		if (diff > 0.3 * v_flowLengths[i])
		{
			indexes.push_back(i);
		}
	}
	for (int i = 0; i < indexes.size(); i++)
	{
		groundPlaneFlowTails.erase(groundPlaneFlowTails.begin() + (indexes[i] - i));
		groundPlaneFlowHeads.erase(groundPlaneFlowHeads.begin() + (indexes[i] - i));
		v_flowLengths.erase(v_flowLengths.begin() + (indexes[i - i]));
	}
}