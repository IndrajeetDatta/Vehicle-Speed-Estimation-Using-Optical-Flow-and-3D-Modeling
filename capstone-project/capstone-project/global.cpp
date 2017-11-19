#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"

using namespace cv;
using namespace std;

//////////////////////////////----- GLOBAL VARIABLES -----///////////////////////////////////////////

Mat cameraMatrix, distCoeffs, rotationVector, translationVector, rotationMatrix, inverseHomographyMatrix; // Intrisic camera parameters (These are stored here so that all the classes could access them).

float frameRate, videoTimeElapsed, realTimeElapsed, totalFrameCount, frameHeight, frameWidth, fourCC;
// Video properties (These are stored here so that all the classes could access them).

int currentFrameCount; // Current frame count.

const Scalar WHITE = Scalar(255, 255, 255), BLACK = Scalar(0, 0, 0), BLUE = Scalar(255, 0, 0), GREEN = Scalar(0, 255, 0), RED = Scalar(0, 0, 255), YELLOW = Scalar(0, 255, 255); // Scalar values that store color (For the ease of calling them).

Mat currentFrame, nextFrame, currentFrame_gray, nextFrame_gray, currentFrame_blur, nextFrame_blur, morph, diff, thresh, videoMask, imgCuboids, imgTracks; // Frames (Used in blob class to find optical flows as well as in main. So, that's why here).
 
const Point3f cameraCenter = Point3f(1.80915, -8.95743, 8.52165); // Camera center


//////////////////////////////////////----- GLOBAL METHODS -----/////////////////////////////////////

////////////////////----- METHOD TO FIND WORLD POINT FROM IMAGE POINT -----//////////////////////////

// This method takes an image point and finds the projection of the point in the world coordinates with a constant z value.

Point3f findWorldPoint(const Point2f &imagePoint, double zConst, const Mat &cameraMatrix, const Mat &rotationMatrix, const Mat &translationVector)
{
	// We know w[x, y, 1] = K(R[X Y Zconst] + T) 
	// From this, we can find that [X Y Zconst] = R^-1 (wK^-1[x y 1] - T)
	// Let A = K^-1R^-1[x y 1] and B = R^-1T
	// Then, w = (Zconst + B(2,0)) /A(2,0)
	// ** The matrices above are needed to be transposed. Due to difficulty in writing here, it hasn't been shown that way.
	// This derivation is detailed in the report. Please refer to the report.

	Mat imagePointHV = Mat::ones(3, 1, DataType<double>::type); // Creating a 3X1 vector of ones to store the image point coordinates as a homogenous vector.

	imagePointHV.at<double>(0, 0) = imagePoint.x; // Changing the first value to the x value of the vector.

	imagePointHV.at<double>(1, 0) = imagePoint.y; // Changing the second value to the y value of the vector.

	// The third value remains 1.

	Mat A, B;

	A = rotationMatrix.inv() * cameraMatrix.inv() * imagePointHV; // A = R^-1K^-1[x y 1]
	B = rotationMatrix.inv() * translationVector; // B = R^-1T

	double p = A.at<double>(2, 0); // A(2,0)
	double q = zConst + B.at<double>(2, 0); // Zconst + B(2,0)
	double w = q / p; // w = Zconst + B(2,0) / A(2,0)

	Mat worldPointHV = rotationMatrix.inv() * (w * cameraMatrix.inv() * imagePointHV - translationVector); // [X Y Z] = R^-1(wK^-1[x y 1] - T)

	Point3f worldPoint;
	worldPoint.x = worldPointHV.at<double>(0, 0);
	worldPoint.y = worldPointHV.at<double>(1, 0);
	worldPoint.z = 0.0;

	return worldPoint;
}

////////////////////////////----- METHODS TO FIND DISTANCES BETWEEN TWO POINTS -----/////////////////

// This method finds the distance between any two 2 dimensional points with floating point values.
float distanceBetweenPoints(Point2f point1, Point2f point2) 
{
	float x = abs(point1.x - point2.x);
	float y = abs(point1.y - point2.y);

	return(sqrt(pow(x, 2) + pow(y, 2)));
}

/////////////////////////////////////////////////////////////////////////////////////////////////////

// This method finds the distance between two 2 dimensional points with one having a floating point values and another having integer values.
float distanceBetweenPoints(Point2f point1, Point point2)
{
	float x = abs(point1.x - point2.x);
	float y = abs(point1.y - point2.y);

	return(sqrt(pow(x, 2) + pow(y, 2)));
}

//////////////////////////////////////////////////////////////////////////////////////////////////

// This method finds the distance between two 3 dimensional points with floating point values.
float distanceBetweenPoints(Point3f point1, Point3f point2) 
{
	float x = abs(point1.x - point2.x);
	float y = abs(point1.y - point2.y);
	float z = abs(point1.z - point2.z);

	return(sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

//////////////////////////----- METHOD TO CHECK IF POINT LIES INSIDE RECTANGLE IN 3D-----////////////

// This method checks if a 3 dimensional point is inside a rectangle in 3 dimensional coordinate axis.
bool pointInsideRect(vector<Point3f> points, Point3f point)
{
	Point3f p1, p2, p3, p4, m; // p1, p2, p3, p4 are the four vertices of the rectangle in a 3 dimensional axis. m is a 3 dimensional point.

	// Initializing the points
	p1 = points[0];
	p2 = points[1];
	p3 = points[2];
	p4 = points[3];
	m = point;

	// Constructs four vectors correspoding to the edges of the rectangle.
	Vec3f v12(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
	Vec3f v23(p3.x - p2.x, p3.y - p2.y, p3.z - p2.z);
	Vec3f v34(p4.x - p3.x, p4.y - p3.y, p4.z - p3.z);
	Vec3f v41(p1.x - p4.x, p1.y - p4.y, p1.z - p4.z);

	// Constructs four vectors starting from each vertex of the rectangle to the point of concern.
	Vec3f v1m(m.x - p1.x, m.y - p1.y, m.z - p1.z);
	Vec3f v2m(m.x - p2.x, m.y - p2.y, m.z - p2.z);
	Vec3f v3m(m.x - p3.x, m.y - p3.y, m.z - p3.z);
	Vec3f v4m(m.x - p4.x, m.y - p4.y, m.z - p4.z);

	// Finds the length of each of the four vectors that correspond to the edges of the rectangle.
	float v12_length = sqrt(pow(v12[0], 2) + pow(v12[1], 2) + pow(v12[2], 2));
	float v23_length = sqrt(pow(v23[0], 2) + pow(v23[1], 2) + pow(v23[2], 2));
	float v34_length = sqrt(pow(v34[0], 2) + pow(v34[1], 2) + pow(v34[2], 2));
	float v41_length = sqrt(pow(v41[0], 2) + pow(v41[1], 2) + pow(v41[2], 2));

	// Finds the length of each of the four vectors that joins each edge of the rectangle to the point of concern.
	float v1m_length = sqrt(pow(v1m[0], 2) + pow(v1m[1], 2) + pow(v1m[2], 2));
	float v2m_length = sqrt(pow(v2m[0], 2) + pow(v2m[1], 2) + pow(v2m[2], 2));
	float v3m_length = sqrt(pow(v3m[0], 2) + pow(v3m[1], 2) + pow(v3m[2], 2));
	float v4m_length = sqrt(pow(v4m[0], 2) + pow(v4m[1], 2) + pow(v4m[2], 2));

	// Divides each of the vector that correspond to the edges of the cuboid by their lengths to get their normalized forms.
	Vec3f v12_norm = v12 / v12_length;
	Vec3f v23_norm = v23 / v23_length;
	Vec3f v34_norm = v34 / v34_length;
	Vec3f v41_norm = v41 / v41_length;

	// Divides each of the vector that joins each edge to the point of concern by their lengths to get their normalized forms.
	Vec3f v1m_norm = v1m / v1m_length;
	Vec3f v2m_norm = v2m / v2m_length;
	Vec3f v3m_norm = v3m / v3m_length;
	Vec3f v4m_norm = v4m / v4m_length;

	// Finds the dot product of vectors that correspond to each edge with the vector that joins the edge to the point of concern
	float dot1 = v12_norm.dot(v1m_norm);
	float dot2 = v23_norm.dot(v2m_norm);
	float dot3 = v34_norm.dot(v3m_norm);
	float dot4 = v41_norm.dot(v4m_norm);

	// The above four dot products need to be positive or zero since the represent the dot product of a vector originating at an edge and point towards its adjecent edge with another vector that   originates from the same edge and points towards the point of concern. If the point is inside  the rectangle, the angle between such two vector has to be an acute angle and hence the dot    product has to be positive (Between 1 and 0 for angle 0 degrees to 90 degrees).
	
	// Checking if all the dot products are positive.
	if (dot1 >= 0 && dot2 >= 0 && dot3 >= 0 && dot4 >= 0)
	{
		return true; // If yes, the point lies inside the rectangle.
	}

	else
	{
		return false; // Else, not.
	}

}

//////////////////////----- METHOD TO FIND MEDIAN OF VALUES -----///////////////////////////////////

// This method finds the median of a vector of floating point values.
float findMedian(vector<float> values) 
{
	int m = values.size(); // Stores the size of the vector.

	sort(values.begin(), values.end()); // Sorts the vector in ascending order of their values.

	float medianValue; 

	if (m % 2 == 0)
	{
		medianValue = (values[(m / 2) - 1] + values[m / 2]) / 2; // If the size of vector is an even number.
	}
	else
	{
		medianValue = values[(m + 1) / 2]; // If the size of vector is an odd number.
	}

	return medianValue;
}

// This method eliminates optical flow outliers that are longer/shorter than a threshold amound from the optical flow vector with the median length.

/////////////////////////----- METHOD TO ELIMINATE OPTICAL FLOW OUTLIERS -----//////////////////////

void eliminateOutliers(vector<Point2f> flowTails, vector<Point2f> flowHeads) 
{
	vector<float> v_flowLengths;

	// Stores the size of the optical flow vectors.
	int m = flowTails.size(); 
	
	// Finds the flow length of each optical flow vectors and pushes the amount to a vector of floats.
	for (int i = 0; i < m; i++) 
	{
		float flowLength = sqrt(pow(flowHeads[i].x - flowTails[i].x, 2) + pow(flowHeads[i].y - flowTails[i].y, 2));

		v_flowLengths.push_back(flowLength);
	}

	// Sorts the vector of floats that stores the lengths of all the optical flow vectors.
	sort(v_flowLengths.begin(), v_flowLengths.end()); 

	// Finds the median value of the vector that stores the vector of flow lengths.
	float medianFlowLength = findMedian(v_flowLengths); 

	vector<int> indexes;

	// Loops through all the flow vectors finds the flow vectors that are threshold amount longer/    shorter than the vector with median flow length. Stores the index of those optical flow        vectors into another vector.
	for (int i = 0; i < m; i++) 
	{
		float diff = abs(medianFlowLength - v_flowLengths[i]);
		if (diff > 0.3 * v_flowLengths[i])
		{
			indexes.push_back(i);
		}
	}

	// Loops through all the indexes found and then deletes each of the optical flow vectors having the indexes. 
	// ** Note that as the each flow vector is deleted the index for the next one reduces by one.
	for (int i = 0; i < indexes.size(); i++)
	{
		flowTails.erase(flowTails.begin() + (indexes[i] - i));
		flowHeads.erase(flowHeads.begin() + (indexes[i] - i));
	}
}


