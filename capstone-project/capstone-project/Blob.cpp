#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"
#include "Blob.h"

using namespace cv;
using namespace std;


Blob::Blob(const vector<Point> &contour)
{

	this->contour = contour;
	this->boundingRectangle = boundingRect(contour);
	this->area = boundingRectangle.area();
	this->width = boundingRectangle.width;
	this->height = boundingRectangle.height;
	this->aspectRatio = (float)boundingRectangle.width / (float)boundingRectangle.height;
	this->diagonalSize = sqrt(pow(boundingRectangle.width, 2) + pow(boundingRectangle.height, 2));

	this->center = Point((boundingRectangle.x + boundingRectangle.width / 2), (boundingRectangle.y + boundingRectangle.height / 2));

}

void Blob::findFeatures()
{
	Mat mask(Size(frameWidth, frameHeight), CV_8UC1, BLACK);

	vector<vector<Point> > contours;
	contours.push_back(contour);
	drawContours(mask, contours, -1, WHITE, -1, CV_AA);

	vector<Point2f> flowTails;
	goodFeaturesToTrack(currentFrame_gray, flowTails, 10000, 0.01, 5, mask);
	cornerSubPix(currentFrame_gray, flowTails, Size(10, 10), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.03));
	this->v_flowTails = flowTails;
}



void Blob::findFlows()
{
	vector<Point2f> flowHeads;
	int win_size = 10;
	vector<uchar> status;
	vector<float> error;

	calcOpticalFlowPyrLK(currentFrame_gray, nextFrame_gray, this->v_flowTails, flowHeads, status, error, Size(win_size * 2 + 1, win_size * 2 + 1), 5, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.3));
	this->v_flowHeads = flowHeads;
	eliminateOutliers(this->v_flowTails, this->v_flowHeads);
}




Blob::~Blob() {}