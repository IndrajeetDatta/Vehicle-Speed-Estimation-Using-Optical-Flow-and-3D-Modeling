#pragma once
#ifndef TRACK_H

#include "Blob.h"
#include "Cuboid.h"

class Track
{
private:

	vector<Blob> blobs_;
	vector<Cuboid> cuboids_;
	vector<float> instantaneousSpeed_;
	float averageSpeed_, averageFlowDistanceX_, averageFlowDistanceY_, angleOfMotion_;
	bool trackUpdated_;
	bool beingTracked_;
	int matchCount_;
	int noMatchCount_;
	int trackNumber_ = 0;
	Scalar trackColor_;
	vector<vector<Point3f> > planeProjectedFlowTails_, planeProjectedFlowHeads_;

public:

	Track(const Blob &blob);
	~Track();

	vector<Blob> getBlobs() const { return blobs_; }
	vector<Cuboid> getCuboids() const { return cuboids_; }
	bool isTrackUpdated() const { return trackUpdated_; }
	bool isBeingTracked() const { return beingTracked_; }
	int getMatchCount() const { return matchCount_; }
	int getNoMatchCount() const { return noMatchCount_; }
	int	getTrackNumber() const { return trackNumber_; }
	Scalar getTrackColor() const { return trackColor_; }
	Blob getRecentBlob() const { return blobs_.back(); }
	Cuboid getRecentCuboid() const { return cuboids_.back(); }
	vector<vector<Point3f> > getPlaneProjectedFlowTails() { return planeProjectedFlowTails_; }
	vector<vector<Point3f> > getPlaneProjectedFlowHeads() { return planeProjectedFlowHeads_; }

	void add(const Blob &blob);
	void setBoolTrackUpdated(bool value) { trackUpdated_ = value; }
	void setBoolBeingTracked(bool value) { beingTracked_ = value; }
	void incrementMatchCount() { matchCount_++; }
	void incrementNoMatchCount() { noMatchCount_++; }
	void clearNoMatchCount() { noMatchCount_ = 0; }
	void setTrackNumber(int value) { trackNumber_ = value; }



	void drawBlobTrail(Mat &outputFrame, int trailLength, Scalar trailColor, int trailThickness);
	void drawCuboidTrail(Mat &outputFrame, int trailLength, Scalar trailColor, int trailThickness);
	void drawTrackInfoOnBlobs(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness);
	void drawTrackInfoOnCuboids(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness);
};


#endif // !TRACK_H

