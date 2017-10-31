#pragma once
#include <iostream>
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

extern Mat cameraMatrix, distCoeffs, rotationVector, translationVector, rotationMatrix, inverseHomographyMatrix;

extern float frameRate, videoTimeElapsed, totalFrameCount, frameHeight, frameWidth, fourCC;

extern int currentFrameCount;

extern const Scalar WHITE, BLACK, BLUE, GREEN, RED, YELLOW;

extern Mat currentFrame, nextFrame, currentFrame_gray, nextFrame_gray, currentFrame_blur, nextFrame_blur, morph, diff, thresh, videoMask, imgCuboids, imgTracks;

extern const Point3f cameraCenter;

extern const float initialCuboidLength, initialCuboidWidth, initialCuboidHeight;

extern FileStorage optimizationData;

Point3f findWorldPoint(const Point2f &imagePoint, double zConst, const Mat &cameraMatrix, const Mat &rotationMatrix, const Mat &translationVector);

float distanceBetweenPoints(Point2f point1, Point2f point2);
float distanceBetweenPoints(Point2f point1, Point point2);
float distanceBetweenPoints(Point3f point1, Point3f point2);

bool pointInsideRect(vector<Point3f> points, Point3f point);

float findMedian(vector<float> values);
void eliminateOutliers(vector<Point3f> groundPlaneFlowTails, vector<Point3f> groundPlaneFlowHeads);
void eliminateOutliers(vector<Point2f> flowTails, vector<Point2f> flowHeads);
