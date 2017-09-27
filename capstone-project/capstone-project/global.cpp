#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"

using namespace cv;
using namespace std;

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

vector<vector<Point3f> >findFlowsProjectedOnPlanes(vector<vector<float> > planeParameters, vector<vector<Point3f> > planeVertices, vector<Point3f> groundPlaneFlowPoints, Point3f cameraCenter)
{

	vector<vector<Point3f> > projectedFlowPointsOnPlanes;

	for (int i = 0; i < planeParameters.size(); i++)
	{
		vector<Point3f> points;
		for (int j = 0; j < groundPlaneFlowPoints.size(); j++)
		{
			float t = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlaneFlowPoints[j].x - cameraCenter.x) + planeParameters[i][1] * (groundPlaneFlowPoints[j].y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);


			Point3f projectedFlowPoint = Point3f(cameraCenter.x + (groundPlaneFlowPoints[j].x - cameraCenter.x) * t, cameraCenter.y + (groundPlaneFlowPoints[j].y - cameraCenter.y) * t, cameraCenter.z - cameraCenter.z * t);
			bool inside = pointInside(planeVertices[i], projectedFlowPoint);
			//cout << inside << endl;
			if (inside)
			{
				points.push_back(projectedFlowPoint);
				//cout << projectedFlowPoint << " is pushed backed." << endl;
				cout << endl;
			}
		}
		projectedFlowPointsOnPlanes.push_back(points);
	}

	return projectedFlowPointsOnPlanes;
}
