#include <iostream>
#include <opencv2\opencv.hpp>

#include "global.h"
#include "Blob.h"

using namespace cv;
using namespace std;

Blob::Blob(const vector<Point> &contour)
{

	contour_ = contour;
	
	boundingRectangle_ = boundingRect(contour);
	
	area_ = boundingRectangle_.area();
	
	width_ = boundingRectangle_.width;
	
	height_ = boundingRectangle_.height;
	
	aspectRatio_ = (float)boundingRectangle_.width / (float)boundingRectangle_.height;
	
	diagonalSize_ = sqrt(pow(boundingRectangle_.width, 2) + pow(boundingRectangle_.height, 2));

	center_ = Point((boundingRectangle_.x + boundingRectangle_.width / 2), (boundingRectangle_.y + boundingRectangle_.height / 2));

	topRightCorner_ = Point(boundingRectangle_.x, boundingRectangle_.y);
	
	bottomRightCorner_ = Point(boundingRectangle_.x + boundingRectangle_.width, boundingRectangle_.y + boundingRectangle_.height);

	bottomLeftCorner_ = Point(boundingRectangle_.x, boundingRectangle_.y + boundingRectangle_.height);

	Mat mask(Size(frameWidth, frameHeight), CV_8UC1, BLACK);
	
	vector<vector<Point> > contours;
	
	contours.push_back(contour_);
	
	drawContours(mask, contours, -1, WHITE, -1, CV_AA);
	
	vector<Point2f> flowTails;
	
	goodFeaturesToTrack(currentFrame_gray, flowTails, 100, 0.01, 5, mask);
	
	cornerSubPix(currentFrame_gray, flowTails, Size(10, 10), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.03));
	
	flowTails_ = flowTails;
	
	int win_size = 10;

	vector<uchar> status;
	
	vector<float> error;
	
	calcOpticalFlowPyrLK(currentFrame_gray, nextFrame_gray, flowTails_, flowHeads_, status, error, Size(win_size * 2 + 1, win_size * 2 + 1), 5, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.3));

}

Blob::~Blob() {}

void Blob::drawBlob(Mat &outputFrame, Scalar rectColor, int rectThickness, Scalar contourColor, int contourThickness, Scalar centerColor, int centerThickness)
{
	vector<vector<Point> > contours;
	
	contours.push_back(getContour());
	
	rectangle(outputFrame, getBoundingRect(), rectColor, 1, CV_AA);
	
	drawContours(outputFrame, contours, -1, contourColor, 1, CV_AA);
	
	circle(outputFrame, getCenter(), 1, centerColor, -1, CV_AA);
}

void Blob::drawBlobFlows(Mat &outputFrame, Scalar flowColor, int flowThickness)
{
	for (int i = 0; i < getFlowTails().size(); i++)
	{
		arrowedLine(outputFrame, getFlowTails()[i], getFlowHeads()[i], flowColor, flowThickness);
	}
}

void Blob::drawBlobInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness)
{
	rectangle(outputFrame, getTopRightCorner(), Point(getTopRightCorner().x + getWidth(), getTopRightCorner().y - getDiagonalSize() / 9), backgroundColor, -1, CV_AA);

	rectangle(outputFrame, Point(getBottomLeftCorner().x + 3, getBottomLeftCorner().y + 3), Point(getBottomLeftCorner().x + getWidth() - 3, getBottomLeftCorner().y + 40), backgroundColor, -1, CV_AA);

	putText(outputFrame, "(" + to_string(getBottomLeftCorner().x) + ", " + to_string(getBottomLeftCorner().y) + ")", Point(getBottomLeftCorner().x + 5, getBottomLeftCorner().y + 15), fontFace, getWidth() / 350, fontColor, fontThickness, CV_AA);

	putText(outputFrame, "Width: " + to_string((int)getWidth()), Point(getBottomLeftCorner().x + 5, getBottomLeftCorner().y + 25), fontFace, getWidth() / 350, fontColor, fontThickness, CV_AA);

	putText(outputFrame, "Height: " + to_string((int)getHeight()), Point(getBottomLeftCorner().x + 5, getBottomLeftCorner().y + 35), fontFace, getWidth() / 350, fontColor, fontThickness, CV_AA);
}