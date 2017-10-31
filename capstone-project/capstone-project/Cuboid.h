#pragma once
#ifndef CUBOID_H
class Cuboid
{
private:

	int cuboidNumber = 0;
	float length, width, height, orientation;
	Point3f centroid;
	vector<Point3f> v_vertices;
	vector<Point2f> v_ipVertices;
	vector<Point2f> v_convexHull;
	vector<vector<Point3f>> v_planeVertices;
	vector<vector<float>> v_planeParameters;
	vector<vector<Point3f>> v_cp_flows;

public:

	Cuboid(Point3f point, float length, float width, float height, float angleOfMotion);
	Cuboid();
	~Cuboid();

	int getCuboidNumber() const { return this->cuboidNumber; }
	float getLength() const { return this->length; }
	float getWidth() const { return this->width; }
	float getHeight() const { return this->height; }
	float getAngleOfMotion() const { return this->orientation; }
	Point3f getCentroid() const { return this->centroid; }
	vector<Point3f> getVertices() const { return this->v_vertices; }
	vector <vector<Point3f> > getPlaneVertices() const { return this->v_planeVertices; }
	vector<Point2f> getIPVertices() const { return this->v_ipVertices; }
	vector<Point2f> getConvexHull() const { return this->v_convexHull; }
	vector<vector<float>> getPlaneParameters() const { return this->v_planeParameters; }
	vector <vector<Point3f>> getCP_Flows() const { return this->v_cp_flows; }
	
	void setCuboidNumber(int number) { this->cuboidNumber = number; }
	void Cuboid::findFlowsOnPlanes(vector<Point2f> flowTails, vector<Point2f> flowHeads);

};



#endif // !CUBOID_H