#pragma once
#ifndef BLOB_H

class Blob
{

private:

	vector<Point> contour;
	Rect boundingRectangle;
	float area, width, height, diagonalSize, aspectRatio;
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
	Point getCenter() const { return this->center; }
	vector<Point2f> getFlowTails() const { return this->v_flowTails; }
	vector<Point2f> getFlowHeads() const { return this->v_flowHeads; }

	void findFeatures();
	void findFlows();

	void setFlowTails(vector<Point2f> flowTails) { this->v_flowTails = flowTails; }

	

};


#endif // !BLOB_H
