#pragma once
#ifndef TRACK_H

#include "Blob.h"
#include "Cuboid.h"

class Track
{
private:

	/////----- MEMEBER VARIABLES -----/////

	int trackNumber; // Stores the track number.

	vector<Blob> v_blobs; // Stores the blobs in the track.

	vector<Cuboid> v_cuboids; // Stores the cuboids in the track.

	float initialCuboidLength, initialCuboidWidth, initialCuboidHeight, initialMotionX, initialMotionY, optimizedCuboidLength, optimizedCuboidWidth, optimizedCuboidHeight, optimizedMotionX, optimizedMotionY; // Stores the initial and optimized parameters of dimensions and motion of the cuboids between two specific frames.

	Point3f nextPoint; // Stores the next estimated ground plane point of the centroid of the next cuboid.


	vector<float> instantaneousSpeeds; // Stores the instaneous speeds of the cuboid.

	float averageSpeed; // Stores the average speed of the cuboids stored in the track until a specific frame.

	bool trackUpdated; // Stores 'true' if the track is updated.

	bool beingTracked; // Stores 'true' if the track is being track.

	int matchCount; // Keeps count of the number of consecutive frames with a match.

	int noMatchCount; // Keeps count of the the number of consecutive frames without a match.

	Scalar trackColor; // Stores an arbitrary scalar value to draw blobs and cuboids of a specific track in the same color.

public:
	

	Track(Blob &blob, int trackNumber); // Constructor
	~Track(); // Destructor

	/////----- MEMBER FUNCTIONS -----/////

	void add(Blob &blob); // Method to add new matched blobs.

	float findAverageSpeed(); // Method to find the averages speed of the cuboids in the track.

	void drawCuboid(Mat &outputFrame); // Method to draw cuboids projected to the image plane.

	void drawBlob(Mat &outputFrame); // Method to draw bounding boxes of the blobs.

	void drawFlows(Mat &outputFrame); // Method to draw the optical flow vectors.

	/////----- GETTER AND SETTER FUNCTIONS -----/////

	vector<Blob> getBlobs() const { return this->v_blobs; } // Returns the vector of blobs.

	vector<Cuboid> getCuboids() const { return this->v_cuboids; } // Returns the vector of cuboids.

	bool isTrackUpdated() const { return this->trackUpdated; } // Returns the boolean "trackUpdated".

	bool isBeingTracked() const { return this->beingTracked; } // Returns the boolean "beingTracked".

	int getMatchCount() const { return this->matchCount; } // Returns the match count.

	int getNoMatchCount() const { return this->noMatchCount; } // Returns the consecutive no match count.

	int	getTrackNumber() const { return this->trackNumber; } // Returns the track number.

	Scalar getTrackColor() const { return this->trackColor; } // Returns the track color.

	Blob getLastBlob() const { return this->v_blobs.back(); } // Returns the last blobs in the vector of matched blobs.

	Cuboid getLastCuboid() const { return this->v_cuboids.back(); } // Returns the last cuboid in the vector of cuboids.

	void setBoolTrackUpdated(bool value) { this->trackUpdated = value; } // Sets the boolean 'trackUpdated' to the parameter passed in.

	void setBoolBeingTracked(bool value) { this->beingTracked = value; } // Sets the boolean 'beingTracked' to the parameter passed in.

	void incrementMatchCount() { this->matchCount++; } // Increments the match counter by 1.

	void incrementNoMatchCount() { this->noMatchCount++; } // Increments the no match counter by 1.

	void clearNoMatchCount() { this->noMatchCount = 0; } // Clears the no match counter by setting it to 0.
};


#endif // !TRACK_H
