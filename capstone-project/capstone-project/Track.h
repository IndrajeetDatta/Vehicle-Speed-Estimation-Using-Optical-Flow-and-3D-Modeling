#pragma once
#ifndef TRACK_H

#include "Blob.h"
#include "Cuboid.h"

class Track
{
private:

	vector<Blob> v_blobs;
	vector<Cuboid> v_cuboids;
	Point3f nextPoint;
	float initialCuboidLength, initialCuboidWidth, initialCuboidHeight, initialAngleOfMotion, initialMotionX, initialMotionY, optimizedCuboidLength, optimizedCuboidWidth, optimizedCuboidHeight, optimizedAngleOfMotion, optimizedMotionX, optimizedMotionY;

	vector<float> instantaneousSpeeds_;
	float averageSpeed;
	bool trackUpdated;
	bool beingTracked;
	int matchCount;
	int noMatchCount;
	int trackNumber = 0;
	int cuboidCount = 0;
	Scalar trackColor;

public:

	Track(Blob &blob);
	~Track();

	vector<Blob> getBlobs() const { return this->v_blobs; }
	vector<Cuboid> getCuboids() const { return this->v_cuboids; }
	bool isTrackUpdated() const { return this->trackUpdated; }
	bool isBeingTracked() const { return this->beingTracked; }
	int getMatchCount() const { return this->matchCount; }
	int getNoMatchCount() const { return this->noMatchCount; }
	int	getTrackNumber() const { return this->trackNumber; }
	Scalar getTrackColor() const { return this->trackColor; }
	Blob getPrevBlob() const { return this->v_blobs.back(); }
	Cuboid getPrevCuboid() const { return this->v_cuboids.back(); }


	void add(Blob &blob);
	void setBoolTrackUpdated(bool value) { this->trackUpdated = value; }
	void setBoolBeingTracked(bool value) { this->beingTracked = value; }
	void incrementMatchCount() { this->matchCount++; }
	void incrementNoMatchCount() { this->noMatchCount++; }
	void clearNoMatchCount() { this->noMatchCount = 0; }
	void setTrackNumber(int value) { this->trackNumber = value; }
	float findAverageSpeed();


	void drawBlobTrail(Mat &outputFrame, int trailLength, Scalar trailColor, int trailThickness);
	void drawCuboidInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness);
	void drawCuboid(Mat &outputFrame, Scalar color, int lineThickness);

};


#endif // !TRACK_H

