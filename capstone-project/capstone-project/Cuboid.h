#pragma once
#ifndef CUBOID_H
class Cuboid
{
private:

	float length_, width_, height_, angleOfMotion_;
	Point3f b1_, b2_, b3_, b4_, t1_, t2_, t3_, t4_;
	Point3f centroid_;
	Point projectedCentroid_;
	vector<Point3f> vertices_;
	vector<Point2f> imagePlaneProjectedVertices_;
	vector<vector<float> > planeParameters_;
	vector<vector<Point3f>> planeVertices_;
	vector <vector<Point2f>> measuredOpticalFlowHeads_;

public:

	Cuboid(Point3f &point, float length, float width, float height, float angleOfMotion);
	Cuboid();
	~Cuboid();

	float getLength() const { return length_; }
	float getWidth() const { return width_; }
	float getHeight() const { return height_; }
	float getAngleOfMotion() const { return angleOfMotion_; }
	Point3f getB1() const { return b1_; }
	Point3f getCentroid() const { return centroid_; }
	Point2f getProjectedCentroid() const { return projectedCentroid_; }
	vector<Point3f> getVertices() const { return vertices_; }
	vector <vector<Point3f> > getPlaneVertices() const{ return planeVertices_; }
	vector<Point2f> getProjectedVertices() const { return imagePlaneProjectedVertices_; }
	vector<vector<float> > getPlaneParameters() const { return planeParameters_; }
	void setMeasuredOpticalFlowHeads(vector<vector<Point2f>> flowHeads) { measuredOpticalFlowHeads_ = flowHeads; }
	vector<vector<Point2f>> getMeasuredOpticalFlowHeads() { return measuredOpticalFlowHeads_; }
	void drawCuboid(Mat &outputFrame, Scalar color, int lineThickness);
};



#endif // !CUBOID_H
