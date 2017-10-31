#pragma once
#ifndef TRACK_H

#include "Blob.h"
#include "Cuboid.h"

class Track
{
private:
	int trackNumber;
	vector<Blob> v_blobs;
	vector<Cuboid> v_cuboids;
	float initialCuboidLength, initialCuboidWidth, initialCuboidHeight, initialMotionX, initialMotionY, optimizedCuboidLength, optimizedCuboidWidth, optimizedCuboidHeight, optimizedMotionX, optimizedMotionY;
	Point3f nextPoint;
	

	vector<float> instantaneousSpeeds;
	float averageSpeed;
	bool trackUpdated;
	bool beingTracked;
	int matchCount;
	int noMatchCount;
	int cuboidCount = 0;
	Scalar trackColor;

public:

	Track(Blob &blob, int trackNumber);
	~Track();

	vector<Blob> getBlobs() const { return this->v_blobs; }
	vector<Cuboid> getCuboids() const { return this->v_cuboids; }
	bool isTrackUpdated() const { return this->trackUpdated; }
	bool isBeingTracked() const { return this->beingTracked; }
	int getMatchCount() const { return this->matchCount; }
	int getNoMatchCount() const { return this->noMatchCount; }
	int	getTrackNumber() const { return this->trackNumber; }
	Scalar getTrackColor() const { return this->trackColor; }
	Blob getLastBlob() const { return this->v_blobs.back(); }
	Cuboid getLastCuboid() const { return this->v_cuboids.back(); }


	void add(Blob &blob);
	void setBoolTrackUpdated(bool value) { this->trackUpdated = value; }
	void setBoolBeingTracked(bool value) { this->beingTracked = value; }
	void incrementMatchCount() { this->matchCount++; }
	void incrementNoMatchCount() { this->noMatchCount++; }
	void clearNoMatchCount() { this->noMatchCount = 0; }
	void setTrackNumber(int number) { this->trackNumber = number; }
	float findAverageSpeed();

	void Track::drawCuboid(Mat &outputFrame);
	void Track::drawBlob(Mat &outputFrame);
};


#endif // !TRACK_H
