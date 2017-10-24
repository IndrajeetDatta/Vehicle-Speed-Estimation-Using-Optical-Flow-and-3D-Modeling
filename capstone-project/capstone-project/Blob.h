#pragma once
#ifndef BLOB_H

class Blob
{

private:

	vector<Point> contour;
	Rect boundingRectangle;
	float area, width, height, diagonalSize, aspectRatio, angleOfMotion;
	Point  center;
	vector<Point2f> v_flowTails, v_flowHeads;

public:

	Blob(const vector<Point> &contour);
	~Blob();

	//getter functions

	vector<Point> getContour() const { return this->contour; }

	Rect getBoundingRect() const { return this->boundingRectangle; }

	float getArea() const { return this->area; }
	float getWidth() const { return this->width; }
	float getHeight() const { return this->height; }
	float getDiagonalSize() const { return this->diagonalSize; }
	float getAspectRatio() const { return this->aspectRatio; }
	float getAngleOfMotion() const { return this->angleOfMotion; }

	Point getCenter() const { return this->center; }
	vector<Point2f> getFlowTails() const { return this->v_flowTails; }
	vector<Point2f> getFlowHeads() const { return this->v_flowHeads; }

	void findFeatures();
	void findFlows();

	void setFlowTails(vector<Point2f> flowTails) { this->v_flowTails = flowTails; }

	void drawBlob(Mat &outputFrame, Scalar rectColor, int rectThickness, Scalar contourColor = BLUE, int contourThickness = 1, Scalar centerColor = GREEN, int centerThickness = 1);
	void drawBlobFlows(Mat &outputFrame, Scalar flowColor, int flowThickness);

};


#endif // !BLOB_H

