#pragma once
#ifndef LMFUNCTOR_H
#define LMFUNCTOR_H
#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"

#include "Track.h"
#include "Blob.h"
#include "Cuboid.h"

#include <Eigen/Eigen>
#include <unsupported/Eigen/NonLinearOptimization>

using namespace std;
using namespace cv;
using namespace Eigen;


typedef Matrix<Point2f, Dynamic, Dynamic> MatrixPoint2f;
typedef Matrix<Point3f, Dynamic, Dynamic> MatrixPoint3f;

struct LMFunctor
{
	Point3f point;
	vector<Point2f> flowTails, flowHeads;
	float minLength, maxLength, minWidth, maxWidth, minHeight, maxHeight, minMotionX, maxMotionX, minMotionY, maxMotionY;

	// The operator finds the error vector for a given set of parameters.
	int operator()(const VectorXf &x, VectorXf &fvec) const
	{
		
		float param_length = x(0);
		float param_width = x(1);
		float param_height = x(2);
		float param_motionX = x(3);
		float param_motionY = x(4);

		// Checks if the parameters are with and upper and lower bound if not, set it equal to the upper or lower bound
		if (param_length < minLength) param_length = minLength;
		if (param_length > maxLength) param_length = maxLength;
		if (param_width < minWidth) param_width = minWidth;
		if (param_width > maxWidth) param_width = maxWidth;
		if (param_height < minHeight) param_height = minHeight;
		if (param_height > maxHeight) param_height = maxHeight;
		if (param_motionX < minMotionX) param_motionX = minMotionX;
		if (param_motionX > maxMotionX) param_motionX = maxMotionX;
		if (param_motionY < minMotionY) param_motionY = minMotionY;
		if (param_motionY > maxMotionY) param_motionY = maxMotionY;

		// Constructs a cuboid with the given parameters to find the cuboid plane parameters and the cuboid plane vertices.
		Cuboid cuboid(point, param_length, param_width, param_height, atan(param_motionX / param_motionY));

		vector<vector<float>> v_planeParameters = cuboid.getPlaneParameters();
		vector<vector<Point3f>> v_planeVertices = cuboid.getPlaneVertices();

		// Loops through each optical flow data point.
		for (int i = 0; i < values(); i++)
		{
			// Sets if point is found to be false.
			bool pointFound = false;

			// Finds the ground plane flow tail.
			Point3f gp_ft = findWorldPoint(flowTails[i], 0, cameraMatrix, rotationMatrix, translationVector);

			// Sets the smallest distance to an arbitrary high number
			float smallestDistance = 10000; Point3f cp_ft, cp_fh;

			// Loops through each plane of the cuboid.
			for (int j = 0; j < v_planeParameters.size(); j++)
			{
				// Finds the scalar that when multipled to the vector from the camera center point towards the ground plane point of the optical flow tail finds the point of intersection with the cuboid plane. ** Please refer to the report for the full derivation.
				float t = -(v_planeParameters[j][0] * cameraCenter.x + v_planeParameters[j][1] * cameraCenter.y + v_planeParameters[j][2] * cameraCenter.z + v_planeParameters[j][3]) / (v_planeParameters[j][0] * (gp_ft.x - cameraCenter.x) + v_planeParameters[j][1] * (gp_ft.y - cameraCenter.y) + v_planeParameters[j][2] * (gp_ft.z - cameraCenter.z));

				// Finds the point of intersection with the cuboid plane.
				Point3f point(cameraCenter.x + ((gp_ft.x - cameraCenter.x) * t), cameraCenter.y + ((gp_ft.y - cameraCenter.y) * t), cameraCenter.z + ((gp_ft.z - cameraCenter.z) * t));

				// Checks if the point is inside the rectangle of the cuboid plane.
				bool inside = pointInsideRect(v_planeVertices[j], point);

				// If it is inside..
				if (inside)
				{
					// Sets the point found to be true.
					pointFound = true;

					// Finds the distance of the point from the camera center.
					float distance = distanceBetweenPoints(point, cameraCenter);

					// If the distance is smaller than the arbitrarily high number set earlier..
					if (distance < smallestDistance)
					{
						// Set the variable 'cp_ft' to the point
						cp_ft = point;

						// Chnage the variable 'smallestDistance' to the new distance.
						smallestDistance = distance;
					}
				}
			}

			// After the end of this loop, if the point is found, and even found at multiple planes, the variable 'cp_ft' will contain the one with the smallest distance from the camera center.

			// So, if the point was found..
			if (pointFound)
			{
				// Add the parameter of motion to the point found.
				Point3f point = Point3f(cp_ft.x + param_motionX, cp_ft.y + param_motionY, cp_ft.z);

				// Find the image point of the point after added motion.
				vector<Point3f> objectPoints; vector<Point2f> imagePoints;
				objectPoints.push_back(point);
				projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

				// The error should be distance between the image plane project point after the motion and the optical flow head in the image plane. In case the parameters were perfect, the points should co-incide.

				// Setting each value of 'fvec' (which is Eigen's LM member variable) to the error found.
				fvec(i) = sqrt(pow((imagePoints[0].x - flowHeads[i].x), 2) + pow((imagePoints[0].y - flowHeads[i].y), 2));
			}

			// In case the point of intersection was not found with any plane..
			else
			{
				// There are two scenarios..

				// The point is an outlier so the optical flow motion should be zero. Hence, the error is the distance between the flow head and the flow tail.
				float error1 = sqrt(pow((flowHeads[i].x - flowTails[i].x), 2) + pow((flowHeads[i].y - flowTails[i].y), 2));


				// Or the point belongs to the cuboid, in that case we add the motion parameters to the flow tail and find the reprojection error of that point like usual and add the distance of the flow tail to the convex hull of the  cuboid (times a constant) to the error so the optimizer changes the parameters in a way that the distance from the cuboids convex hull to the flow tail is minimum and the point gets  inside the cuboid's convex hull.

				// Adding the motion to the point and finding its image plane coordinate
				Point3f point = Point3f(cp_ft.x + param_motionX, cp_ft.y + param_motionY, cp_ft.z);
				vector<Point3f> objectPoints; vector<Point2f> imagePoints;
				objectPoints.push_back(point);
				projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

				// Finding the distance of the flow tail from the cuboid's convex hull.
				float distance = pointPolygonTest(cuboid.getConvexHull(), flowTails[i], true);

				// Error is the distance between the point after motion and the observed flowhead plus the distance between the flow tails and the convex hull of the cuboid times a distance.
				float error2 = sqrt(pow((imagePoints[0].x - flowHeads[i].x), 2) + pow((imagePoints[0].y - flowHeads[i].y), 2)) + 0.5 * distance;

				// Now if error 1 is less that error 2, the error is error1. Else, otherwise.
				if (error1 < error2)
				{
					fvec(i) = error1;
				}
				else
				{
					fvec(i) = error2;
				}

			}
		}
		return 0;
	}


	// This method finds the Jacobian matrix at each iteration of the optimizer.
	int df(const VectorXf &x, MatrixXf &fjac) const
	{
		// Setting 'epsilon' to a small value that will be used to change the parameters
		float epsilon;
		epsilon = 1e-5f;

		// Looping through all the parameters
		for (int i = 0; i < x.size(); i++)
		{
			// Creates a new dynamic sized vector and copies the paramters, then adds 'epsilon' to the ith parameter.
			VectorXf xPlus(x);
			xPlus(i) += epsilon;

			// Creates a new dynamix sized vector and copies the parameters, the subtracts 'epsilon' to the ith parameter.
			VectorXf xMinus(x);
			xMinus(i) -= epsilon;

			// Creates a new dynmic sized vector that stores the values of error for paramters with ith parameter added 'epsilon'. Calls the operator to finds the errors.
			VectorXf fvecPlus(values());
			operator()(xPlus, fvecPlus);

			// Creates a new dynmic sized vector that stores the values of error for paramters with ith parameter subtracted  'epsilon'. Calls the operator to finds the errors.
			VectorXf fvecMinus(values());
			operator()(xMinus, fvecMinus);

			// Creates a new dynamic sized vector that stores the difference in the of the errors founds above
			VectorXf fvecDiff(values());
			fvecDiff = (fvecPlus - fvecMinus);

			// Creates a new blob in the Jacobian matrix with the values being the difference in errors divided the change in parameters (i.e. 2 * 'epsilon')
			VectorXf fjacBlock(values());
			fjacBlock = fvecDiff / (2.0f * epsilon);
			fjac.block(0, i, values(), 1) = fjacBlock;
		}
		/*
		// Printing out the optimized length, width, height, motion, the error vector as well as the Jacobian matrix.
		VectorXf err(values());
		operator()(x, err);

		float sum = 0;
		for (int i = 0; i < err.size(); i++)
		{
			sum += pow(err(i), 2);
		}
		
		cout << "Length: " << x(0) << endl;
		cout << "Width: " << x(1)  << endl;
		cout << "Height: " << x(2) << endl;
		cout << "Motion in X: " << x(3) << endl;
		cout << "Motion in Y: " <<  x(4) << endl;
		cout << "Sum of Squared Errors: " << sum << endl;
		cout << endl;
		cout << "Errors: " << endl << err << endl;
		cout << endl;
		cout << "Jacobian: " << endl << fjac << endl;
		cout << endl;
		*/
		return 0;
	}

	MatrixPoint2f observedFlows;
	int m;

	int values() const { return m; }

	int n;

	int inputs() const { return n; }


};

#endif // !LMFUNCTOR_H
