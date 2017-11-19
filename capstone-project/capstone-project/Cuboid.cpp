#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"
#include "Cuboid.h"

using namespace std;
using namespace cv;

///////////////////////////////////----- CUBOID CONSTRUCTOR -----////////////////////////////////////

Cuboid::Cuboid(Point3f point, float length, float width, float height, float orientation)
{
	/// Initializing the member variables.

	this->length = length;

	this->width = width;

	this->height = height;

	this->orientation = orientation;

	// Finding the vertices of the cuboid in the world coordinate axis from the provided parameters of length, width, height, orientation and the estimated ground plane point of the centroid.

	Point3f b1 = Point3f((point.x - (width / 2)) * cos(orientation) - (point.y - (length / 2)) * sin(orientation), (point.x - (width / 2)) * sin(orientation) + (point.y - (length / 2)) * cos(orientation), 0.0);

	Point3f b2 = Point3f((point.x - (width / 2)) * cos(orientation) - (point.y + (length / 2)) * sin(orientation), (point.x - (width / 2)) * sin(orientation) + (point.y + (length / 2)) * cos(orientation), 0.0);

	Point3f b3 = Point3f((point.x + (width / 2)) * cos(orientation) - (point.y + (length / 2)) * sin(orientation), (point.x + (width / 2)) * sin(orientation) + (point.y + (length / 2)) * cos(orientation), 0.0);

	Point3f b4 = Point3f((point.x + (width / 2)) * cos(orientation) - (point.y - (length / 2)) * sin(orientation), (point.x + (width / 2)) * sin(orientation) + (point.y - (length / 2)) * cos(orientation), 0.0);

	Point3f t1 = Point3f((point.x - (width / 2)) * cos(orientation) - (point.y - (length / 2)) * sin(orientation), (point.x - (width / 2)) * sin(orientation) + (point.y - (length / 2)) * cos(orientation), height);

	Point3f t2 = Point3f((point.x - (width / 2)) * cos(orientation) - (point.y + (length / 2)) * sin(orientation), (point.x - (width / 2)) * sin(orientation) + (point.y + (length / 2)) * cos(orientation), height);

	Point3f t3 = Point3f((point.x + (width / 2)) * cos(orientation) - (point.y + (length / 2)) * sin(orientation), (point.x + (width / 2)) * sin(orientation) + (point.y + (length / 2)) * cos(orientation), height);

	Point3f t4 = Point3f((point.x + (width / 2)) * cos(orientation) - (point.y - (length / 2)) * sin(orientation), (point.x + (width / 2)) * sin(orientation) + (point.y - (length / 2)) * cos(orientation), height);

	// Finding the centroid of the cuboid.
	this->centroid = Point3f(point.x, point.y, point.z + (height / 2));

	// Pushing back the vertices found to the vector.
	this->v_vertices.push_back(b1);
	this->v_vertices.push_back(b2);
	this->v_vertices.push_back(b3);
	this->v_vertices.push_back(b4);
	this->v_vertices.push_back(t1);
	this->v_vertices.push_back(t2);
	this->v_vertices.push_back(t3);
	this->v_vertices.push_back(t4);

	// Projecting the vertices of the cuboid in the world coordinate axis to the image plane and storing the image plane vertices in the member variable 'v_ipVertices'.
	projectPoints(this->v_vertices, rotationVector, translationVector, cameraMatrix, distCoeffs, this->v_ipVertices);


	// Finding the convex hull of the cuboid formed in the image plane. (This is needed in the error function).
	vector<Point2f> v_convexHull(v_ipVertices.size());
	convexHull(v_ipVertices, v_convexHull);
	this->v_convexHull = v_convexHull;

	// Sorting the vertices of each plane of the cuboid into separate vectors and then storing all those vectors into a vector of vector. (This is need in the error function for ease of computation of optical flow vectors projected onto each cuboid plane).
		
	// Vertices of the top plane. (Chosen first for ease of computation due a lot of optical flow vectors projecting on this plane because of the location of the camera being on a ellevated foot bridge.)
	vector<Point3f> topPlaneVertices;
	topPlaneVertices.push_back(t4);
	topPlaneVertices.push_back(t1);
	topPlaneVertices.push_back(t2);
	topPlaneVertices.push_back(t3);

	this->v_planeVertices.push_back(topPlaneVertices);

	// Vertices of the front plane. (Chosen second because second most number of optical flow vectors will be projected on this plane).
	vector<Point3f> frontPlaneVertices;
	frontPlaneVertices.push_back(b4);
	frontPlaneVertices.push_back(b1);
	frontPlaneVertices.push_back(t1);
	frontPlaneVertices.push_back(t4);

	this->v_planeVertices.push_back(frontPlaneVertices);

	// Vertices of the left plane.
	vector<Point3f> leftPlaneVertices;
	leftPlaneVertices.push_back(b3);
	leftPlaneVertices.push_back(b4);
	leftPlaneVertices.push_back(t4);
	leftPlaneVertices.push_back(t3);

	this->v_planeVertices.push_back(leftPlaneVertices);

	// Vertices of the right plane.
	vector<Point3f> rightPlaneVertices;
	rightPlaneVertices.push_back(b2);
	rightPlaneVertices.push_back(b1);
	rightPlaneVertices.push_back(t1);
	rightPlaneVertices.push_back(t2);

	this->v_planeVertices.push_back(rightPlaneVertices);

	// Vertices of the bottom plane.
	vector<Point3f> bottomPlaneVertices;
	bottomPlaneVertices.push_back(b4);
	bottomPlaneVertices.push_back(b1);
	bottomPlaneVertices.push_back(b2);
	bottomPlaneVertices.push_back(b3);

	this->v_planeVertices.push_back(bottomPlaneVertices);

	// Vertices of the back plane.
	vector<Point3f> backPlaneVertices;
	backPlaneVertices.push_back(b3);
	backPlaneVertices.push_back(b2);
	backPlaneVertices.push_back(t2);
	backPlaneVertices.push_back(t3);

	this->v_planeVertices.push_back(backPlaneVertices);


	// Now the parameters (A, B, C, D) (in that order) of the plane equation Ax + By + Cz + D for each plane of the cuboid are found. This will be need in the error function as well. These parameters are found with respect to the length, width, height and the orientation of the cuboid. Changing the parameters of length, width, height and orientation will change the planes of the cuboid as well.

	// Finding the plane parameters of the top plane. (This is also kept in the same order as the vector of vector that store the vertices. This is for the ease of computation of the error in the error function as well).
	vector<float> topPlaneParameters;
	topPlaneParameters.push_back(0);
	topPlaneParameters.push_back(0);
	topPlaneParameters.push_back(-1);
	topPlaneParameters.push_back(centroid.z + (height / 2));

	this->v_planeParameters.push_back(topPlaneParameters);

	// Finding the plane parameters of the front plane.
	vector<float> frontPlaneParameters;
	frontPlaneParameters.push_back(-sin(orientation));
	frontPlaneParameters.push_back(cos(orientation));
	frontPlaneParameters.push_back(0);
	frontPlaneParameters.push_back((length / 2) + (centroid.x * sin(orientation)) - (centroid.y * cos(orientation)));

	this->v_planeParameters.push_back(frontPlaneParameters);

	// Finding the plane parameters of the left plane.
	vector<float> leftPlaneParameters;
	leftPlaneParameters.push_back(-cos(orientation));
	leftPlaneParameters.push_back(-sin(orientation));
	leftPlaneParameters.push_back(0);
	leftPlaneParameters.push_back((width / 2) + (centroid.x * cos(orientation)) + (centroid.y * sin(orientation)));

	this->v_planeParameters.push_back(leftPlaneParameters);

	// Finding the plane parameters of the right plane.
	vector<float> rightPlaneParameters;
	rightPlaneParameters.push_back(cos(orientation));
	rightPlaneParameters.push_back(sin(orientation));
	rightPlaneParameters.push_back(0);
	rightPlaneParameters.push_back((width / 2) - (centroid.x * cos(orientation)) - (centroid.y * sin(orientation)));

	this->v_planeParameters.push_back(rightPlaneParameters);

	// Finding the plane parameters of the bottom plane.
	vector<float> bottomPlaneParameters;
	bottomPlaneParameters.push_back(0);
	bottomPlaneParameters.push_back(0);
	bottomPlaneParameters.push_back(1);
	bottomPlaneParameters.push_back(-(centroid.z) + (height / 2));

	this->v_planeParameters.push_back(bottomPlaneParameters);

	// Finding the plane parameters of the back plane.
	vector<float> backPlaneParameters;
	backPlaneParameters.push_back(sin(orientation));
	backPlaneParameters.push_back(-cos(orientation));
	backPlaneParameters.push_back(0);
	backPlaneParameters.push_back((length / 2) - (centroid.x * sin(orientation)) + (centroid.y * cos(orientation)));

	this->v_planeParameters.push_back(backPlaneParameters);
}

Cuboid::~Cuboid() {};


