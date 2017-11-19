#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"
#include "Blob.h"

using namespace cv;
using namespace std;

//////////////////////////////////----- BLOB CONSTRUCTOR -----///////////////////////////////////

Blob::Blob(const vector<Point> &contour)
{
	// Initializing the member variables.

	this->contour = contour; // Contour of the blob.

	this->boundingRectangle = boundingRect(contour); // Bounding rectangle of the contour of the blob.

	this->area = boundingRectangle.area(); // Area of the bounding rectangle of the contour of the blob.

	this->width = boundingRectangle.width; // Width of the bounding rectangle of the contour of the blob.

	this->height = boundingRectangle.height; // Height of the bounding rectangle of the contour of the blob.

	this->aspectRatio = (float)boundingRectangle.width / (float)boundingRectangle.height; // Aspect ratio (width/height) of the bounding rectangle.

	this->diagonalSize = sqrt(pow(boundingRectangle.width, 2) + pow(boundingRectangle.height, 2)); // Diagonal size of the bounding rectangle.

	this->center = Point((boundingRectangle.x + boundingRectangle.width / 2), (boundingRectangle.y + boundingRectangle.height / 2)); // Center of the bounding rectangle.

}

/////////////----- METHOD TO FIND FEATURES POINTS TO TRACK INSIDE THE BLOB -----/////////////////////

void Blob::findFeatures() // This member function find feature points of the blob to track.
{
	// Masking the rest of the frame except the blob itself so that the feature points are only found inside the blob containing the vehicle.
	Mat mask(Size(frameWidth, frameHeight), CV_8UC1, BLACK);
	vector<vector<Point> > contours;
	contours.push_back(contour);
	drawContours(mask, contours, -1, WHITE, -1, CV_AA); 

	// Using OpenCV's goodFeaturesToTrack to find features points to track inside the blob and then using OpenCV's cornerSubPix to get accurate results to the sub-pixel level.
	vector<Point2f> flowTails;
	goodFeaturesToTrack(currentFrame_gray, flowTails, 10000, 0.01, 5, mask);
	cornerSubPix(currentFrame_gray, flowTails, Size(10, 10), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.03));
	this->v_flowTails = flowTails;
}

//////////----- METHOD TO FIND OPTICAL FLOW HEADS OF THE FEATURE POINTS OF THE BLOB-----//////////// 

void Blob::findFlows() // This member function finds the optical flow heads of the feature points of the blob.
{
	vector<Point2f> flowHeads;
	int win_size = 10;
	vector<uchar> status;
	vector<float> error;

	// Using OpenCV's inbuilt calcOpticalFlowPyrLK method to find flow tails of the feature points optained by the member function findFeatures.
	calcOpticalFlowPyrLK(currentFrame_gray, nextFrame_gray, this->v_flowTails, flowHeads, status, error, Size(win_size * 2 + 1, win_size * 2 + 1), 5, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.3));
	
	// Setting the member variable 'v_flowHeads' to the obtained flowHeads.
	this->v_flowHeads = flowHeads;

	// Eliminating outliers using 'eliminateOutliers' (See global.cpp for more info).
	eliminateOutliers(this->v_flowTails, this->v_flowHeads);
}




Blob::~Blob() {}