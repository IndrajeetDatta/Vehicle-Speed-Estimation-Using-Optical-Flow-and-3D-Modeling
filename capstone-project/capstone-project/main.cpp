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

		vector<vector<Point>> convexHulls(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			convexHull(contours[i], convexHulls[i]);
		}

		vector<Blob> currentFrameBlobs;
		for (int i = 0; i < convexHulls.size(); i++)
		{
			Rect rect = boundingRect(convexHulls[i]);
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
			if (tracks[i].getNoMatchCount() < 3 && tracks[i].getMatchCount() > 5)
			{
				Blob blob = tracks[i].getPrevBlob();
				Cuboid cuboid = tracks[i].getPrevCuboid();
				Scalar trackColor = tracks[i].getTrackColor();

				tracks[i].drawCuboid(imgCuboids, trackColor, 1);

				vector<vector<Point2f>> ip_flows = cuboid.getIPFlows();
				for (int i = 0; i < ip_flows.size(); i++)
				{
					arrowedLine(imgCuboids, ip_flows[i][0], ip_flows[i][1], Scalar(0, 255, 0), 1, CV_AA);
				}
				vector<vector<Point3f>> cp_flows = cuboid.getCPFlows();
				vector<Point3f> objectPoints1, objectPoints2;
				for (int i = 0; i < cp_flows.size(); i++)
				{
					Point3f point1 = cp_flows[i][0];
					Point3f point2 = cp_flows[i][1];
					objectPoints1.push_back(point1);
					objectPoints2.push_back(point2);
				}
				vector<Point2f> imagePoints1, imagePoints2;
				projectPoints(objectPoints1, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints1);
				projectPoints(objectPoints2, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints2);
				for (int i = 0; i < imagePoints1.size(); i++)
				{
					circle(imgCuboids, imagePoints1[i], 1, Scalar(0, 0, 255), 1, CV_AA);
				}

				/*if (convexHull.size() > 0)
				{
				vector<vector<Point2f>> temp;
				temp.push_back(convexHull);
				drawContours(imgCuboids, temp, -1, RED, 1, CV_AA);
				}*/
				/*vector<vector<Point3f>> gp_flows = cuboid.getGPFlows();
				vector<Point3f> objectPoints3, objectPoints4;
				for (int i = 0; i < gp_flows.size(); i++)
				{
				Point3f point1 = gp_flows[i][0];
				Point3f point2 = gp_flows[i][1];
				objectPoints3.push_back(point1);
				objectPoints4.push_back(point2);
				}
				vector<Point2f> imagePoints3, imagePoints4;
				projectPoints(objectPoints3, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints3);
				projectPoints(objectPoints4, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints4);
				for (int i = 0; i < imagePoints3.size(); i++)
				{
				circle(imgCuboids, imagePoints3[i], 1, Scalar(0, 0, 255), 1, CV_AA);
				arrowedLine(imgCuboids, imagePoints3[i], imagePoints4[i], Scalar(0, 255, 0), 1, CV_AA);
				}*/


			}

			if (tracks[i].isTrackUpdated() == false)
			{
				tracks[i].incrementNoMatchCount();
			}

			if (tracks[i].getNoMatchCount() > 15)
			{
				tracks[i].setBoolBeingTracked(false);
			}
			if (tracks[i].isBeingTracked() == false)
			{
				tracks.erase(tracks.begin() + i);
			}
		}

		displayInfo(imgCuboids, BLACK, CV_FONT_HERSHEY_SIMPLEX, 0.35, WHITE);

		rectangle(imgCuboids, Point((imgCuboids.cols * 2 / 3) - 10, 0), Point(imgCuboids.cols, 14), BLACK, -1, CV_AA);
		putText(imgCuboids, "Vehicle Detection and Tracking", Point((imgCuboids.cols * 2 / 3) - 5, 10), CV_FONT_HERSHEY_SIMPLEX, 0.35, WHITE, 0.35, CV_AA);

		vector<Point3f> objectPoints; vector<Point2f> imagePoints;
		objectPoints.push_back(Point3f(0.0, 0.0, 0.0));
		objectPoints.push_back(Point3f(1.0, 0.0, 0.0));
		objectPoints.push_back(Point3f(0.0, 1.0, 0.0));
		objectPoints.push_back(Point3f(0.0, 0.0, 1.0));
		projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);
		arrowedLine(imgCuboids, imagePoints[0], imagePoints[1], RED, 1, CV_AA);
		arrowedLine(imgCuboids, imagePoints[0], imagePoints[2], BLUE, 1, CV_AA);
		arrowedLine(imgCuboids, imagePoints[0], imagePoints[3], GREEN, 1, CV_AA);

		imshow("Cuboids", imgCuboids);

		currentFrame = nextFrame.clone();
		capture.read(nextFrame);

		key = waitKey(/*1000 / frameRate*/ 0);
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


			if (tracks[index_leastDistance].getMatchCount() == 5)
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
