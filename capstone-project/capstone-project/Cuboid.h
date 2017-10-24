#pragma once
#ifndef CUBOID_H
class Cuboid
{
private:

	int cuboidNumber;
	float length, width, height, angleOfMotion, motionX, motionY;
	Point3f b1, b2, b3, b4, t1, t2, t3, t4;
	Point3f centroid;
	vector<Point3f> v_vertices;
	vector<Point2f> v_ipVertices;
	vector<Point2f> v_convexHull;
	vector<vector<Point3f>> v_planeVertices;
	vector<vector<float>> v_planeParameters;
	vector <vector<Point2f>> v_ip_flows;
	vector<vector<Point3f>> v_cp_flows;
	vector<vector<Point3f>> v_gp_flows;

public:

	Cuboid(Point3f point, float length, float width, float height, float angleOfMotion);
	Cuboid();
	~Cuboid();

	float getLength() const { return this->length; }
	float getWidth() const { return this->width; }
	float getHeight() const { return this->height; }
	float getAngleOfMotion() const { return this->angleOfMotion; }
	Point3f getCentroid() const { return this->centroid; }
	vector<Point3f> getVertices() const { return this->v_vertices; }
	vector <vector<Point3f> > getPlaneVertices() const { return this->v_planeVertices; }
	vector<Point2f> getIPVertices() const { return this->v_ipVertices; }
	vector<Point2f> getConvexHull() const { return this->v_convexHull; }
	vector<vector<float>> getPlaneParameters() const { return this->v_planeParameters; }
	vector<vector<Point2f>> getIPFlows() const { return this->v_ip_flows; }
	vector<vector<Point3f>> getCPFlows() const { return this->v_cp_flows; }
	vector<vector<Point3f>> getGPFlows() const { return this->v_gp_flows; }
	void Cuboid::findFlowsOnPlanes(vector<Point2f> flowTails, vector<Point2f> flowHeads);

};



#endif // !CUBOID_H
