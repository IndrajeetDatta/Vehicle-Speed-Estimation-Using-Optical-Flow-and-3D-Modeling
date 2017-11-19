#pragma once
#ifndef BLOB_H

class Blob
{

private:
	
	/////----- MEMBER VARIABLES -----/////

	vector<Point> contour; // Stores the contour of the blob.

	Rect boundingRectangle; // Stores the bounding rectangle of the contour of the blob.

	float area, width, height, diagonalSize, aspectRatio; // Stores the area, width, height, diagonalSize and aspectRatio of the bounding rectangle of the contour of the blob.

	Point  center; // Stores the center of the bounding rectangle of the contour of the blob.

	vector<Point2f> v_flowTails, v_flowHeads; // Vectors that stores the optical flow tails and the optical flow heads of the blobs.

public:

	Blob(const vector<Point> &contour); // Constructor
	~Blob(); // Destructor

	void findFeatures();
	void findFlows();
	vector<Point> getContour() const { return this->contour; } // Returns the contour of the blob.

	/////----- GETTER AND SETTER FUNCTIONS -----/////

	Rect getBoundingRect() const { return this->boundingRectangle; } // Returns the bounding rectangle of the contour of the blob.

	float getArea() const { return this->area; } // Returns the area of the bounding rectangle of the blob.

	float getWidth() const { return this->width; } // Returns the width of the bounding rectangle of the blob.

	float getHeight() const { return this->height; } // Returns the height of the bounding rectangle of the blob.

	float getDiagonalSize() const { return this->diagonalSize; } // Returns the diagonal size of the bounding rectangle of the blob.

	float getAspectRatio() const { return this->aspectRatio; } // Returns the aspect ratio (width/height) of the bounding rectangle of the blob.

	Point getCenter() const { return this->center; } // Returns the center of the bounding rectangle of the blob.

	vector<Point2f> getFlowTails() const { return this->v_flowTails; } // Returns the vector containing the optical flow tails of the blob.

	vector<Point2f> getFlowHeads() const { return this->v_flowHeads; } // Returns the vector containing the optical flow heads of the blob.

	void setFlowTails(vector<Point2f> flowTails) { this->v_flowTails = flowTails; } // Sets the vector 'v_flowTails' to the parameter. 
};


#endif // !BLOB_H
