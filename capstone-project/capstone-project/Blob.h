#pragma once
#ifndef BLOB_H

class Blob
{

private:

	vector<Point> contour_;
	Rect boundingRectangle_;
	float area_, width_, height_, diagonalSize_, aspectRatio_;
	Point  center_, bottomLeftCorner_, bottomRightCorner_, topLeftCorner_, topRightCorner_;
	vector<Point2f> flowTails_, flowHeads_;
	vector<Point3f> groundPlaneFlowTails_, groundPlaneFlowHeads_;
	float averageFlowDistanceX_, averageFlowDistanceY_, angleOfMotion_;

public:

	Blob(const vector<Point> &contour);
	~Blob();

	//getter functions

	vector<Point> getContour() const { return contour_; }
	Rect getBoundingRect() const { return boundingRectangle_; }
	float getArea() const { return area_; }
	float getWidth() const { return width_; }
	float getHeight() const { return height_; }
	float getDiagonalSize() const { return diagonalSize_; }
	float getAspectRatio() const { return aspectRatio_; }
	Point getCenter() const { return center_; }
	Point getBottomLeftCorner() const { return bottomLeftCorner_; }
	Point getBottomRightCorner() const { return bottomRightCorner_; }
	Point getTopLeftCorner() const { return topLeftCorner_; }
	Point getTopRightCorner() const { return topRightCorner_; }
	vector<Point2f>getFlowTails() const { return flowTails_; }
	vector<Point2f> getFlowHeads() const { return flowHeads_; }
	vector<Point3f> getGroundPlaneFlowTails() const { return groundPlaneFlowTails_; }
	vector<Point3f> getGroundPlaneFlowHeads() const { return groundPlaneFlowHeads_; }
	float getAverageFlowDistanceX() const { return averageFlowDistanceX_; }
	float getAverageFlowDistanceY() const { return averageFlowDistanceY_; }
	float getAngleOfMotion() const { return angleOfMotion_; }

	void drawBlob(Mat &outputFrame, Scalar rectColor, int rectThickness, Scalar contourColor = BLUE, int contourThickness = 1, Scalar centerColor = GREEN, int centerThickness = 1);
	void drawBlobFlows(Mat &outputFrame, Scalar flowColor, int flowThickness);
	void drawBlobInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness);

};


#endif // !BLOB_H

