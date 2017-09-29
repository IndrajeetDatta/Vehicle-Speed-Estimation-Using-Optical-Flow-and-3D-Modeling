#include <opencv2/opencv.hpp>
#include <iostream>
#include <conio.h>
#include <string>
#include <time.h>

#include "global.h"
#include "Blob.h"
#include "Cuboid.h"
#include "Track.h"


using namespace cv;
using namespace std;

int trackCount = 0;

vector<Track> tracks;

void matchBlobs(vector<Blob> &blobs);
void displayInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, double fontScale, Scalar fontColor);

int main(void)
{

	cout << "Vehicle Speed Estimation Using Optical Flow And 3D Modeling" << endl; cout << endl;
	
	FileStorage fs("parameters.yml", FileStorage::READ);
	fs["Camera Matrix"] >> cameraMatrix;
	fs["Distortion Coefficients"] >> distCoeffs;
	fs["Rotation Vector"] >> rotationVector;
	fs["Translation Vector"] >> translationVector;;
	fs["Rotation Matrix"] >> rotationMatrix;

	cout << "Camera Matrix: " << endl << cameraMatrix << endl; cout << endl;
	cout << "Distortion Coefficients: " << endl << distCoeffs << endl; cout << endl;
	cout << "Rotation Vector" << endl << rotationVector << endl; cout << endl;
	cout << "Translation Vector: " << endl << translationVector << endl; cout << endl;
	cout << "Rotation Matrix: " << endl << rotationMatrix << endl; cout << endl;


	VideoCapture capture;
	capture.open("traffic_chiangrak2.MOV");

	if (!capture.isOpened())
	{
		cout << "Error loading file." << endl;
		system("pause");
		return -1;
	}

	frameRate = capture.get(CV_CAP_PROP_FPS);
	totalFrameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
	frameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	frameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	fourCC = capture.get(CV_CAP_PROP_FOURCC);
	Size frameSize(frameWidth, frameHeight);

	cout << "Video frame rate: " << frameRate << endl;
	cout << "Video total frame count: " << totalFrameCount << endl;
	cout << "Video frame height: " << frameHeight << endl;
	cout << "Video frame width: " << frameWidth << endl;
	cout << "Video Codec: " << fourCC << endl;

	capture.read(currentFrame);
	capture.read(nextFrame);

	Mat mask(frameHeight, frameWidth, CV_8UC1, Scalar(1, 1, 1));
	Point mask_points[1][3];
	mask_points[0][0] = Point(0, 0);
	mask_points[0][1] = Point(0, 342);
	mask_points[0][2] = Point(296, 0);
	const Point* ppt[1] = { mask_points[0] };
	int npt[] = { 3 };
	fillPoly(mask, ppt, npt, 1, Scalar(0, 0, 0), CV_AA);

	int key = 0;

	time_t startTime = time(0);

	while (capture.isOpened() && key != 27)
	{
		currentFrameCount = capture.get(CV_CAP_PROP_POS_FRAMES) - 1;
		videoTimeElapsed = capture.get(CV_CAP_PROP_POS_MSEC) / 1000;
		realTimeElapsed = difftime(time(0), startTime);

		cvtColor(currentFrame, currentFrame_gray, CV_BGR2GRAY);
		cvtColor(nextFrame, nextFrame_gray, CV_BGR2GRAY);

		GaussianBlur(currentFrame_gray, currentFrame_blur, Size(5, 5), 0);
		GaussianBlur(nextFrame_gray, nextFrame_blur, Size(5, 5), 0);

		absdiff(currentFrame_blur, nextFrame_blur, diff);

		threshold(diff, thresh, 50, 255.0, CV_THRESH_BINARY);

		morph = thresh.clone().mul(mask);

		for (int i = 0; i < 6; i++)
		{
			dilate(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
			dilate(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
			erode(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
		}

		vector<vector<Point> > contours;
		findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

		vector<Blob> currentFrameBlobs;
		for (int i = 0; i < contours.size(); i++)
		{
			Rect rect = boundingRect(contours[i]);
			double aspectRatio, diagonalSize;
			aspectRatio = (double)rect.width / (double)rect.height;
			diagonalSize = sqrt(pow(rect.width, 2) + pow(rect.height, 2));

			if (rect.area() > 300 && aspectRatio > 0.2 && aspectRatio < 4.0 && rect.width > 20 && rect.height > 20 && diagonalSize > 50.0)
			{
				Blob blob(contours[i]);
				currentFrameBlobs.push_back(blob);
			}
		}

		if (currentFrameCount == 1)
		{
			for (int i = 0; i < currentFrameBlobs.size(); i++)
			{
				Track track(currentFrameBlobs[i]);
				tracks.push_back(track);
			}
		}
		else
		{
			matchBlobs(currentFrameBlobs);
		}

		Mat imgTracks = currentFrame.clone();
		Mat imgCuboids = currentFrame.clone();

		for (int i = 0; i < tracks.size(); i++)
		{
			if (tracks[i].getNoMatchCount() < 3 && tracks[i].getMatchCount() > 10)
			{
				Blob blob = tracks[i].getPrevBlob();
				Cuboid cuboid = tracks[i].getPrevCuboid();
				Scalar trackColor = tracks[i].getTrackColor();
				
				blob.drawBlob(imgTracks, trackColor, 2);
				blob.drawBlobFlows(imgTracks, GREEN, 1);
				blob.drawBlobInfo(imgTracks, trackColor, CV_FONT_HERSHEY_SIMPLEX, WHITE, 1);

				tracks[i].drawBlobTrail(imgTracks, 10, trackColor, 2);
				tracks[i].drawTrackInfoOnBlobs(imgTracks, trackColor, CV_FONT_HERSHEY_SIMPLEX, WHITE, 1);

				cuboid.drawCuboid(imgCuboids, trackColor, 1);
				tracks[i].drawCuboidTrail(imgCuboids, 10, trackColor, 2);
				tracks[i].drawTrackInfoOnCuboids(imgCuboids, trackColor, CV_FONT_HERSHEY_SIMPLEX, WHITE, 1);

			}

			if (tracks[i].isTrackUpdated() == false)
			{
				tracks[i].incrementNoMatchCount();
			}

			if (tracks[i].getNoMatchCount() > 10)
			{
				tracks[i].setBoolBeingTracked(false);
			}
			if (tracks[i].isBeingTracked() == false)
			{
				tracks.erase(tracks.begin() + i);
			}
		}

		displayInfo(imgTracks, BLACK, CV_FONT_HERSHEY_SIMPLEX, 0.35, WHITE);
		displayInfo(imgCuboids, BLACK, CV_FONT_HERSHEY_SIMPLEX, 0.35, WHITE);

		rectangle(imgTracks, Point((imgTracks.cols * 2 / 3) - 10, 0), Point(imgTracks.cols, 14), BLACK, -1, CV_AA);
		putText(imgTracks, "Vehicle Detection and Tracking", Point((imgTracks.cols * 2 / 3) - 5, 10), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		//imshow("Tracks", imgTracks);
		imshow("Cuboids", imgCuboids);

		currentFrame = nextFrame.clone();
		capture.read(nextFrame);

		key = waitKey(1000 / frameRate);
	}
	return 0;
}

void matchBlobs(vector<Blob> &blobs)
{
	for (int i = 0; i < tracks.size(); i++)
	{
		tracks[i].setBoolTrackUpdated(false);
	}

	for (int i = 0; i < blobs.size(); i++)
	{
		double leastDistance = 100000;
		int index_leastDistance;
		Point center = blobs[i].getCenter();

		for (int j = 0; j < tracks.size(); j++)
		{
			double sumDistances = 0;

			vector<Point2f> flowHeads = tracks[j].getPrevBlob().getFlowHeads();

			for (int k = 0; k < flowHeads.size(); k++)
			{
				double distance = distanceBetweenPoints(flowHeads[k], center);
				sumDistances = sumDistances + distance;
			}

			double averageDistance = sumDistances / (double)flowHeads.size();

			if (averageDistance < leastDistance)
			{
				leastDistance = averageDistance;
				index_leastDistance = j;
			}
		}

		if (leastDistance < blobs[i].getDiagonalSize() * 0.5)
		{
			tracks[index_leastDistance].add(blobs[i]);
			tracks[index_leastDistance].setBoolTrackUpdated(true);
			tracks[index_leastDistance].incrementMatchCount();
			tracks[index_leastDistance].clearNoMatchCount();

			if (tracks[index_leastDistance].getMatchCount() == 10)
			{
				trackCount++;
				tracks[index_leastDistance].setTrackNumber(trackCount);
			}
		}
		else
		{
			Track track(blobs[i]);
			tracks.push_back(track);
		}
	}
}


void displayInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, double fontScale, Scalar fontColor)
{
	rectangle(outputFrame, Point(8, 0), Point(180, 15), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Time Elapsed: " + to_string((int)realTimeElapsed) + " s", Point(10, 10), fontFace, fontScale, fontColor, 0.35, CV_AA);

	rectangle(outputFrame, Point(8, 20), Point(180, 35), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Video Time Elapsed: " + to_string((int)videoTimeElapsed) + " s", Point(10, 30), fontFace, fontScale, fontColor, 0.35, CV_AA);

	rectangle(outputFrame, Point(8, 40), Point(180, 55), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Av. Processing Rate: " + to_string((int)(currentFrameCount / realTimeElapsed)) + " fps", Point(10, 50), fontFace, fontScale, fontColor, 0.35, CV_AA);

	rectangle(outputFrame, Point(8, 60), Point(180, 75), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Frame Count: " + to_string(currentFrameCount) + " / " + to_string((int)totalFrameCount), Point(10, 70), fontFace, 0.35, fontColor, fontScale, CV_AA);
	
	rectangle(outputFrame, Point(8, 80), Point(180, 95), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Vehicle Count: " + to_string(trackCount), Point(10, 90), fontFace, fontScale, fontColor, 0.35, CV_AA);
	
	rectangle(outputFrame, Point(8, 100), Point(180, 115), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Being Tracked: " + to_string(tracks.size()), Point(10, 110), fontFace, fontScale, fontColor, 0.35, CV_AA);
	
	putText(outputFrame, "Vehicle Speed Estimation Using Optical Flow And 3D Modeling by Indrajeet Datta", Point(5, imgTracks.rows - 10), fontFace, 0.3, fontColor, fontScale, CV_AA);
}
