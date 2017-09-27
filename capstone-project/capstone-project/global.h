#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

extern Mat cameraMatrix, distCoeffs, rotationVector, translationVector, rotationMatrix, inverseHomographyMatrix;

extern float frameRate, timeElapsed, totalFrameCount, frameHeight, frameWidth, fourCC;

extern int currentFrameCount;

extern const Scalar WHITE = Scalar(255, 255, 255), BLACK = Scalar(0, 0, 0), BLUE = Scalar(255, 0, 0), GREEN = Scalar(0, 255, 0), RED = Scalar(0, 0, 255), YELLOW = Scalar(0, 255, 255);

extern Mat currentFrame, nextFrame, currentFrame_gray, nextFrame_gray, currentFrame_blur, nextFrame_blur, morph, diff, thresh, videoMask, imgCuboids, imgTracks;

extern const Point3f cameraCenter = Point3f(1.80915, -8.95743, 8.52165);

extern const float initialCuboidLength = 5, initialCuboidWidth = 2, initialCuboidHeight = 1.5;

extern Point3f findWorldPoint(const Point2f &imagePoint, double zConst, const Mat &cameraMatrix, const Mat &rotationMatrix, const Mat &translationVector);

extern float distanceBetweenPoints(Point2f point1, Point point2);

extern float distanceBetweenPoints(Point3f point1, Point3f point2);

extern bool pointInside(vector<Point3f> points, Point3f point);

extern vector<vector<Point3f> >findFlowsProjectedOnPlanes(vector<vector<float> > planeParameters, vector<vector<Point3f> > planeVertices, vector<Point3f> groundPlaneFlowPoints, Point3f cameraCenter);




