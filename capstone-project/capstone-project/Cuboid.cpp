#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"
#include "Cuboid.h"

using namespace std;
using namespace cv;

Cuboid::Cuboid(Point3f point, float length, float width, float height, float angleOfMotion)
{
	this->length = length;

	this->width = width;

	this->height = height;

	this->angleOfMotion = angleOfMotion;

	this->b1 = Point3f((point.x - (width / 2)) * cos(angleOfMotion) - (point.y - (length / 2)) * sin(angleOfMotion), (point.x - (width / 2)) * sin(angleOfMotion) + (point.y - (length / 2)) * cos(angleOfMotion), 0.0);

	this->b2 = Point3f((point.x - (width / 2)) * cos(angleOfMotion) - (point.y + (length / 2)) * sin(angleOfMotion), (point.x - (width / 2)) * sin(angleOfMotion) + (point.y + (length / 2)) * cos(angleOfMotion), 0.0);

	this->b3 = Point3f((point.x + (width / 2)) * cos(angleOfMotion) - (point.y + (length / 2)) * sin(angleOfMotion), (point.x + (width / 2)) * sin(angleOfMotion) + (point.y + (length / 2)) * cos(angleOfMotion), 0.0);

	this->b4 = Point3f((point.x + (width / 2)) * cos(angleOfMotion) - (point.y - (length / 2)) * sin(angleOfMotion), (point.x + (width / 2)) * sin(angleOfMotion) + (point.y - (length / 2)) * cos(angleOfMotion), 0.0);

	this->t1 = Point3f((point.x - (width / 2)) * cos(angleOfMotion) - (point.y - (length / 2)) * sin(angleOfMotion), (point.x - (width / 2)) * sin(angleOfMotion) + (point.y - (length / 2)) * cos(angleOfMotion), height);

	this->t2 = Point3f((point.x - (width / 2)) * cos(angleOfMotion) - (point.y + (length / 2)) * sin(angleOfMotion), (point.x - (width / 2)) * sin(angleOfMotion) + (point.y + (length / 2)) * cos(angleOfMotion), height);

	this->t3 = Point3f((point.x + (width / 2)) * cos(angleOfMotion) - (point.y + (length / 2)) * sin(angleOfMotion), (point.x + (width / 2)) * sin(angleOfMotion) + (point.y + (length / 2)) * cos(angleOfMotion), height);

	this->t4 = Point3f((point.x + (width / 2)) * cos(angleOfMotion) - (point.y - (length / 2)) * sin(angleOfMotion), (point.x + (width / 2)) * sin(angleOfMotion) + (point.y - (length / 2)) * cos(angleOfMotion), height);

	this->centroid = Point3f(point.x, point.y, point.z + (height / 2));

	this->v_vertices.push_back(b1);
	this->v_vertices.push_back(b2);
	this->v_vertices.push_back(b3);
	this->v_vertices.push_back(b4);
	this->v_vertices.push_back(t1);
	this->v_vertices.push_back(t2);
	this->v_vertices.push_back(t3);
	this->v_vertices.push_back(t4);

	projectPoints(this->v_vertices, rotationVector, translationVector, cameraMatrix, distCoeffs, this->v_ipVertices);



	vector<Point2f> v_convexHull(v_ipVertices.size());
	convexHull(v_ipVertices, v_convexHull);
	this->v_convexHull = v_convexHull;


	vector<Point3f> topPlaneVertices;
	topPlaneVertices.push_back(t4);
	topPlaneVertices.push_back(t1);
	topPlaneVertices.push_back(t2);
	topPlaneVertices.push_back(t3);

	this->v_planeVertices.push_back(topPlaneVertices);

	vector<Point3f> frontPlaneVertices;
	frontPlaneVertices.push_back(b4);
	frontPlaneVertices.push_back(b1);
	frontPlaneVertices.push_back(t1);
	frontPlaneVertices.push_back(t4);

	this->v_planeVertices.push_back(frontPlaneVertices);

	vector<Point3f> leftPlaneVertices;
	leftPlaneVertices.push_back(b3);
	leftPlaneVertices.push_back(b4);
	leftPlaneVertices.push_back(t4);
	leftPlaneVertices.push_back(t3);

	this->v_planeVertices.push_back(leftPlaneVertices);

	vector<Point3f> rightPlaneVertices;
	rightPlaneVertices.push_back(b2);
	rightPlaneVertices.push_back(b1);
	rightPlaneVertices.push_back(t1);
	rightPlaneVertices.push_back(t2);

	this->v_planeVertices.push_back(rightPlaneVertices);

	vector<Point3f> bottomPlaneVertices;
	bottomPlaneVertices.push_back(b4);
	bottomPlaneVertices.push_back(b1);
	bottomPlaneVertices.push_back(b2);
	bottomPlaneVertices.push_back(b3);

	this->v_planeVertices.push_back(bottomPlaneVertices);

	vector<Point3f> backPlaneVertices;
	backPlaneVertices.push_back(b3);
	backPlaneVertices.push_back(b2);
	backPlaneVertices.push_back(t2);
	backPlaneVertices.push_back(t3);

	this->v_planeVertices.push_back(backPlaneVertices);

	vector<float> topPlaneParameters;
	topPlaneParameters.push_back(0);
	topPlaneParameters.push_back(0);
	topPlaneParameters.push_back(-1);
	topPlaneParameters.push_back(centroid.z + (height / 2));

	this->v_planeParameters.push_back(topPlaneParameters);

	vector<float> frontPlaneParameters;
	frontPlaneParameters.push_back(-sin(angleOfMotion));
	frontPlaneParameters.push_back(cos(angleOfMotion));
	frontPlaneParameters.push_back(0);
	frontPlaneParameters.push_back((length / 2) + (centroid.x * sin(angleOfMotion)) - (centroid.y * cos(angleOfMotion)));

	this->v_planeParameters.push_back(frontPlaneParameters);

	vector<float> leftPlaneParameters;
	leftPlaneParameters.push_back(-cos(angleOfMotion));
	leftPlaneParameters.push_back(-sin(angleOfMotion));
	leftPlaneParameters.push_back(0);
	leftPlaneParameters.push_back((width / 2) + (centroid.x * cos(angleOfMotion)) + (centroid.y * sin(angleOfMotion)));

	this->v_planeParameters.push_back(leftPlaneParameters);

	vector<float> rightPlaneParameters;
	rightPlaneParameters.push_back(cos(angleOfMotion));
	rightPlaneParameters.push_back(sin(angleOfMotion));
	rightPlaneParameters.push_back(0);
	rightPlaneParameters.push_back((width / 2) - (centroid.x * cos(angleOfMotion)) - (centroid.y * sin(angleOfMotion)));

	this->v_planeParameters.push_back(rightPlaneParameters);

	vector<float> bottomPlaneParameters;
	bottomPlaneParameters.push_back(0);
	bottomPlaneParameters.push_back(0);
	bottomPlaneParameters.push_back(1);
	bottomPlaneParameters.push_back(-(centroid.z) + (height / 2));

	this->v_planeParameters.push_back(bottomPlaneParameters);

	vector<float> backPlaneParameters;
	backPlaneParameters.push_back(sin(angleOfMotion));
	backPlaneParameters.push_back(-cos(angleOfMotion));
	backPlaneParameters.push_back(0);
	backPlaneParameters.push_back((length / 2) - (centroid.x * sin(angleOfMotion)) + (centroid.y * cos(angleOfMotion)));

	this->v_planeParameters.push_back(backPlaneParameters);
}

void Cuboid::findFlowsOnPlanes(vector<Point2f> flowTails, vector<Point2f> flowHeads)
{
	for (int i = 0; i < flowTails.size(); i++)
	{
		Point3f gp_ft = findWorldPoint(flowTails[i], 0, cameraMatrix, rotationMatrix, translationVector);
		Point3f gp_fh = findWorldPoint(flowHeads[i], 0, cameraMatrix, rotationMatrix, translationVector);

		float smallestDistance = 10000; Point3f cp_ft, cp_fh;
		for (int j = 0; j < v_planeParameters.size(); j++)
		{
			float t1 = -(this->v_planeParameters[j][0] * cameraCenter.x + this->v_planeParameters[j][1] * cameraCenter.y + this->v_planeParameters[j][2] * cameraCenter.z + this->v_planeParameters[j][3]) / (this->v_planeParameters[j][0] * (gp_ft.x - cameraCenter.x) + this->v_planeParameters[j][1] * (gp_ft.y - cameraCenter.y) + this->v_planeParameters[j][2] * (gp_ft.z - cameraCenter.z));

			Point3f point1(cameraCenter.x + ((gp_ft.x - cameraCenter.x) * t1), cameraCenter.y + ((gp_ft.y - cameraCenter.y) * t1), cameraCenter.z + ((gp_ft.z - cameraCenter.z) * t1));
			//cout << point1 << endl;
			float t2 = -(this->v_planeParameters[j][0] * cameraCenter.x + this->v_planeParameters[j][1] * cameraCenter.y + this->v_planeParameters[j][2] * cameraCenter.z + this->v_planeParameters[j][3]) / (this->v_planeParameters[j][0] * (gp_fh.x - cameraCenter.x) + this->v_planeParameters[j][1] * (gp_fh.y - cameraCenter.y) + this->v_planeParameters[j][2] * (gp_fh.z - cameraCenter.z));

			Point3f point2(cameraCenter.x + ((gp_fh.x - cameraCenter.x) * t2), cameraCenter.y + ((gp_fh.y - cameraCenter.y) * t2), cameraCenter.z + ((gp_fh.z - cameraCenter.z) * t2));
			/*cout << point2 << endl;
			cout << endl;*/
			bool inside = pointInsideRect(this->v_planeVertices[j], point1) && pointInsideRect(this->v_planeVertices[j], point2);

			if (inside)
			{
				float distance = (distanceBetweenPoints(point1, cameraCenter) + distanceBetweenPoints(point2, cameraCenter)) / 2;
				if (distance < smallestDistance)
				{

					cp_ft = point1;
					cp_fh = point2;
					smallestDistance = distance;
				}
			}
		}

		vector<Point2f> temp1;
		temp1.push_back(flowTails[i]);
		temp1.push_back(flowHeads[i]);
		this->v_ip_flows.push_back(temp1);

		vector<Point3f> temp2;
		temp2.push_back(cp_ft);
		temp2.push_back(cp_fh);
		this->v_cp_flows.push_back(temp2);

		vector<Point3f> temp3;
		temp3.push_back(gp_ft);
		temp3.push_back(gp_fh);
		this->v_gp_flows.push_back(temp3);
	}
}

Cuboid::Cuboid() {};
Cuboid::~Cuboid() {};



