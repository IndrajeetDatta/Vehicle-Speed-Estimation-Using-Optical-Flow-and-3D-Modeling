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

	int operator()(const VectorXf &x, VectorXf &fvec) const
	{
		float param_length = x(0);
		float param_width = x(1);
		float param_height = x(2);
		float param_motionX = x(3);
		float param_motionY = x(4);

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

		Cuboid cuboid(point, param_length, param_width, param_height, atan(param_motionX / param_motionY));
		vector<vector<float>> v_planeParameters = cuboid.getPlaneParameters();
		vector<vector<Point3f>> v_planeVertices = cuboid.getPlaneVertices();


		for (int i = 0; i < values(); i++)
		{
			bool pointFound = false;
			Point3f gp_ft = findWorldPoint(flowTails[i], 0, cameraMatrix, rotationMatrix, translationVector);

			float smallestDistance = 10000; Point3f cp_ft, cp_fh;
			for (int j = 0; j < v_planeParameters.size(); j++)
			{
				float t = -(v_planeParameters[j][0] * cameraCenter.x + v_planeParameters[j][1] * cameraCenter.y + v_planeParameters[j][2] * cameraCenter.z + v_planeParameters[j][3]) / (v_planeParameters[j][0] * (gp_ft.x - cameraCenter.x) + v_planeParameters[j][1] * (gp_ft.y - cameraCenter.y) + v_planeParameters[j][2] * (gp_ft.z - cameraCenter.z));

				Point3f point(cameraCenter.x + ((gp_ft.x - cameraCenter.x) * t), cameraCenter.y + ((gp_ft.y - cameraCenter.y) * t), cameraCenter.z + ((gp_ft.z - cameraCenter.z) * t));

				bool inside = pointInsideRect(v_planeVertices[j], point);

				if (inside)
				{
					pointFound = true;
					float distance = distanceBetweenPoints(point, cameraCenter);
					if (distance < smallestDistance)
					{
						cp_ft = point;
						smallestDistance = distance;
					}
				}
			}
			if (pointFound)
			{
				Point3f point = Point3f(cp_ft.x + param_motionX, cp_ft.y + param_motionY, cp_ft.z);
				vector<Point3f> objectPoints; vector<Point2f> imagePoints;
				objectPoints.push_back(point);
				projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);
				fvec(i) = sqrt(pow((imagePoints[0].x - flowHeads[i].x), 2) + pow((imagePoints[0].y - flowHeads[i].y), 2));
			}
			else
			{
				float error1 = sqrt(pow((flowHeads[i].x - flowTails[i].x), 2) + pow((flowHeads[i].y - flowTails[i].y), 2));



				Point3f point = Point3f(cp_ft.x + param_motionX, cp_ft.y + param_motionY, cp_ft.z);
				vector<Point3f> objectPoints; vector<Point2f> imagePoints;
				objectPoints.push_back(point);
				projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

				float distance = pointPolygonTest(cuboid.getConvexHull(), flowTails[i], true);


				float error2 = sqrt(pow((imagePoints[0].x - flowHeads[i].x), 2) + pow((imagePoints[0].y - flowHeads[i].y), 2)) + 0.5 * distance;


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

	int df(const VectorXf &x, MatrixXf &fjac) const
	{
		float epsilon;
		epsilon = 1e-2f;

		for (int i = 0; i < x.size(); i++)
		{
			VectorXf xPlus(x);
			xPlus(i) += epsilon;

			VectorXf xMinus(x);
			xMinus(i) -= epsilon;

			VectorXf fvecPlus(values());
			operator()(xPlus, fvecPlus);

			VectorXf fvecMinus(values());
			operator()(xMinus, fvecMinus);

			VectorXf fvecDiff(values());
			fvecDiff = (fvecPlus - fvecMinus);

			VectorXf fjacBlock(values());
			fjacBlock = fvecDiff / (2.0f * epsilon);
			fjac.block(0, i, values(), 1) = fjacBlock;
		}

		VectorXf err(values());
		operator()(x, err);

		float sum = 0;
		for (int i = 0; i < err.size(); i++)
		{
			sum += pow(err(i), 2);
		}

		/*cout << "Length: " << x(0) << endl;
		cout << "Width: " << x(1)  << endl;
		cout << "Height: " << x(2) << endl;
		cout << "Motion in X: " << x(3) << endl;
		cout << "Motion in Y: " <<  x(4) << endl;
		cout << "Sum of Squared Errors: " << sum << endl;
		cout << endl;
		cout << "Errors: " << endl << err << endl;
		cout << endl;
		cout << "Jacobian: " << endl << fjac << endl;
		cout << endl;*/

		return 0;
	}

	MatrixPoint2f observedFlows;
	int m;

	int values() const { return m; }

	int n;

	int inputs() const { return n; }


};

#endif // !LMFUNCTOR_H
