#pragma once
#ifndef CUBOID_H
class Cuboid
{
private:

	/////------ MEMBER VARIABLES -----/////

	float length, width, height, orientation; // Stores the length, width, height and the orientation of the cuboid.

	Point3f centroid; // Stores the centroid of the cuboid.

	vector<Point3f> v_vertices; // Vector that stores the vertices in the world coordinates.

	vector<Point2f> v_ipVertices; // Vector that stores the vertices in the image coordinates.

	vector<Point2f> v_convexHull; // Stores the convex hull of the cuboid in the image plane.

	vector<vector<Point3f>> v_planeVertices; // Vector of vector that stores the vertices of each plane of the cuboid in separate vectors.

	vector<vector<float>> v_planeParameters; // Vector of vector that stores the plane parameters of the cuboid in separate vectors.

public:

	Cuboid(Point3f point, float length, float width, float height, float angleOfMotion); // Constructor

	~Cuboid(); // Destructor

	/////----- GETTER FUNCTIONS -----/////

	float getLength() const { return this->length; } // Returns the length of the cuboid.

	float getWidth() const { return this->width; } // Returns the width of the cuboid.

	float getHeight() const { return this->height; } // Returns the height of the cuboid.

	float getAngleOfMotion() const { return this->orientation; } // Returns the orientation of the cuboid.

	Point3f getCentroid() const { return this->centroid; } // Returns the centroid of the cuboid.

	vector<Point3f> getVertices() const { return this->v_vertices; } // Returns the vertices of teh cuboid.

	vector <vector<Point3f> > getPlaneVertices() const { return this->v_planeVertices; } // Returns the 'planeVertices'.

	vector<vector<float>> getPlaneParameters() const { return this->v_planeParameters; } // Returns the plane parameters.

	vector<Point2f> getIPVertices() const { return this->v_ipVertices; } // Returns the image plane vertices.

	vector<Point2f> getConvexHull() const { return this->v_convexHull; } // Returns the convex hull of the cuboid.
};



#endif // !CUBOID_H