#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

extern Mat cameraMatrix, distCoeffs, rotationVector, translationVector, rotationMatrix, inverseHomographyMatrix;

extern float frameRate, timeElapsed, totalFrameCount, frameHeight, frameWidth, fourCC;

extern int currentFrameCount;

extern const Scalar WHITE, BLACK, BLUE, GREEN, RED, YELLOW;

extern Mat currentFrame, nextFrame, currentFrame_gray, nextFrame_gray, currentFrame_blur, nextFrame_blur, morph, diff, thresh, videoMask, imgCuboids, imgTracks;

extern const Point3f cameraCenter;

extern const float initialCuboidLength, initialCuboidWidth, initialCuboidHeight;

Point3f findWorldPoint(const Point2f &imagePoint, double zConst, const Mat &cameraMatrix, const Mat &rotationMatrix, const Mat &translationVector);

float distanceBetweenPoints(Point2f point1, Point point2);

float distanceBetweenPoints(Point3f point1, Point3f point2);

bool pointInside(vector<Point3f> points, Point3f point);

vector<vector<Point3f> >findFlowsProjectedOnPlanes(vector<vector<float> > planeParameters, vector<vector<Point3f> > planeVertices, vector<Point3f> groundPlaneFlowPoints, Point3f cameraCenter);




