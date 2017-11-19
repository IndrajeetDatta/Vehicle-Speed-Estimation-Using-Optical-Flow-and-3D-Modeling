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
int vehicleCount = 0;

vector<Track> tracks;

void matchBlobs(vector<Blob> &blobs);
void displayInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, double fontScale, Scalar fontColor);

int main(void)
{

	cout << "Vehicle Speed Estimation Using Optical Flow And 3D Modeling by Indrajeet Datta" << endl; cout << endl;

	// Obtaining intrinsic parameters from .yml file.

	FileStorage fs("parameters.yml", FileStorage::READ);
	fs["Camera Matrix"] >> cameraMatrix;
	fs["Distortion Coefficients"] >> distCoeffs;
	fs["Rotation Vector"] >> rotationVector;
	fs["Translation Vector"] >> translationVector;;
	fs["Rotation Matrix"] >> rotationMatrix;

	// Printing out the intrinsic parameters.

	cout << "Camera Matrix: " << endl << cameraMatrix << endl; cout << endl;
	cout << "Distortion Coefficients: " << endl << distCoeffs << endl; cout << endl;
	cout << "Rotation Vector" << endl << rotationVector << endl; cout << endl;
	cout << "Translation Vector: " << endl << translationVector << endl; cout << endl;
	cout << "Rotation Matrix: " << endl << rotationMatrix << endl; cout << endl;
	cout << "------------------------------------------------------------------------------" << endl;
	cout << endl;

	// Opening video.

	VideoCapture capture;
	capture.open("traffic_chiangrak2.MOV");

	if (!capture.isOpened())
	{
		cout << "Error loading file." << endl;
		system("pause");
		return -1;
	}

	// Getting video properties.

	frameRate = capture.get(CV_CAP_PROP_FPS);
	totalFrameCount = capture.get(CV_CAP_PROP_FRAME_COUNT);
	frameHeight = capture.get(CV_CAP_PROP_FRAME_HEIGHT);
	frameWidth = capture.get(CV_CAP_PROP_FRAME_WIDTH);
	fourCC = capture.get(CV_CAP_PROP_FOURCC);

	// Printing out video properties.

	cout << "Video frame rate: " << frameRate << endl; cout << endl;
	cout << "Video total frame count: " << totalFrameCount << endl; cout << endl;
	cout << "Video frame height: " << frameHeight << endl; cout << endl;
	cout << "Video frame width: " << frameWidth << endl; cout << endl;
	cout << "Video Codec: " << fourCC << endl; cout << endl;
	cout << "------------------------------------------------------------------------------" << endl;
	cout << endl;

	// Reading the first two frames of the video.

	capture.read(currentFrame);
	capture.read(nextFrame);

	// Creating a mask to mask out the opposite traffic flow.

	Mat mask(frameHeight, frameWidth, CV_8UC1, Scalar(1, 1, 1));
	Point mask_points[1][3];
	mask_points[0][0] = Point(0, 0);
	mask_points[0][1] = Point(0, 342);
	mask_points[0][2] = Point(296, 0);
	const Point* ppt[1] = { mask_points[0] };
	int npt[] = { 3 };
	fillPoly(mask, ppt, npt, 1, Scalar(0, 0, 0), CV_AA);

	int key = 0;

	// Starting the loop to loop through each frame of the video.
	while (capture.isOpened() && key != 27)
	{
		// Getting the current frame count.
		currentFrameCount = capture.get(CV_CAP_PROP_POS_FRAMES) - 1;

		// If the video reaches the end, break out of the loop.
		if (currentFrameCount == totalFrameCount) break;

		// Converting the two frames obtained to black and white.
		cvtColor(currentFrame, currentFrame_gray, CV_BGR2GRAY);
		cvtColor(nextFrame, nextFrame_gray, CV_BGR2GRAY);

		// Using gaussian blur on the grayscale images reduce noise.
		GaussianBlur(currentFrame_gray, currentFrame_blur, Size(5, 5), 0);
		GaussianBlur(nextFrame_gray, nextFrame_blur, Size(5, 5), 0);

		// Finding the absolute difference.
		absdiff(currentFrame_blur, nextFrame_blur, diff);

		// Thresholding the difference frame.
		threshold(diff, thresh, 30, 255.0, CV_THRESH_BINARY);

		// Putting the mask created earlier on the thresholded image.
		morph = thresh.clone().mul(mask);

		// Implementing morphological operations on the threholded image.
		for (int i = 0; i < 3; i++)
		{
			dilate(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
			dilate(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
			erode(morph, morph, getStructuringElement(MORPH_RECT, Size(5, 5)));
		}

		// Finding the contours on the foreground blobs.
		vector<vector<Point> > contours;
		findContours(morph, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		Mat img_contours(frameHeight, frameWidth, CV_8UC3);
		drawContours(img_contours, contours, -1, WHITE, 1, CV_AA);

		// Finding the convex hulls of the contours obtained.
		vector<vector<Point>> convexHulls(contours.size());
		for (int i = 0; i < contours.size(); i++)
		{
			convexHull(contours[i], convexHulls[i]);
		}
		Mat img_convexHulls(frameHeight, frameWidth, CV_8UC3);
		drawContours(img_convexHulls, convexHulls, -1, WHITE, 1, CV_AA);

		// Pushing the foreground blobs into a vector of Blob objects.
		vector<Blob> currentFrameBlobs;
		for (int i = 0; i < convexHulls.size(); i++)
		{
			Rect rect = boundingRect(convexHulls[i]);
			double aspectRatio, diagonalSize;
			aspectRatio = (double)rect.width / (double)rect.height;
			diagonalSize = sqrt(pow(rect.width, 2) + pow(rect.height, 2));

			if (aspectRatio > 0.3 && aspectRatio < 4.0 && rect.width > 50.0 && rect.height > 50.0 && diagonalSize > 50.0)
			{
				Blob blob(convexHulls[i]);
				currentFrameBlobs.push_back(blob);
			}
		}

		// If it is the first frame of the video, pushing all into separate Track objects.
		if (currentFrameCount == 1)
		{
			for (int i = 0; i < currentFrameBlobs.size(); i++)
			{
				trackCount++;
				Track track(currentFrameBlobs[i], trackCount);
				tracks.push_back(track);
			}
		}

		// If it is not the first frame of the video, implement the matching algorithm.
		else
		{
			matchBlobs(currentFrameBlobs);
		}

		// Drawing the blobs and the cuboids on the current frame.
		Mat imgBlobs = currentFrame.clone();
		Mat imgCuboids = currentFrame.clone();

		for (int i = 0; i < tracks.size(); i++)
		{
			// Only draw if the match count is greater than 5 and no match count is less than 5 or if it is first 5 frames of the video.
			if (currentFrameCount <= 5 || tracks[i].getNoMatchCount() < 5 && tracks[i].getMatchCount() > 5)
			{
				Blob blob = tracks[i].getLastBlob();
				Cuboid cuboid = tracks[i].getLastCuboid();

				tracks[i].drawFlows(imgBlobs);
				tracks[i].drawBlob(imgBlobs);

				tracks[i].drawFlows(imgCuboids);
				tracks[i].drawCuboid(imgCuboids);


			}
			if (tracks[i].getMatchCount() == 5)
			{
				// If match count for a track reaches 5, consider it a vehicle and increase vehicle count.
				vehicleCount++;
			}
			if (tracks[i].isTrackUpdated() == false)
			{
				// If the track is not updated, increment no match count.
				tracks[i].incrementNoMatchCount();
			}

			if (tracks[i].getNoMatchCount() > 15)
			{
				// If the track has not been updated for 15 consecutive frames, set it as "not tracking".
				tracks[i].setBoolBeingTracked(false);
			}
			if (tracks[i].isBeingTracked() == false)
			{
				// If the track is not being tracked, then delete the track.
				tracks.erase(tracks.begin() + i);
			}
		}

		// Displays info about the video and tracking on the output frame.
		displayInfo(imgCuboids, BLACK, CV_FONT_HERSHEY_SIMPLEX, 0.35, YELLOW);
		displayInfo(imgBlobs, BLACK, CV_FONT_HERSHEY_SIMPLEX, 0.35, YELLOW);

		// Draws the world coordinate axis using 1m vectors in the positive directions.
		vector<Point3f> objectPoints; vector<Point2f> imagePoints;
		objectPoints.push_back(Point3f(0.0, 0.0, 0.0));
		objectPoints.push_back(Point3f(1.0, 0.0, 0.0));
		objectPoints.push_back(Point3f(0.0, 1.0, 0.0));
		objectPoints.push_back(Point3f(0.0, 0.0, 1.0));
		projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

		arrowedLine(imgCuboids, imagePoints[0], imagePoints[1], RED, 1, CV_AA);
		arrowedLine(imgCuboids, imagePoints[0], imagePoints[2], BLUE, 1, CV_AA);
		arrowedLine(imgCuboids, imagePoints[0], imagePoints[3], GREEN, 1, CV_AA);

		arrowedLine(imgBlobs, imagePoints[0], imagePoints[1], RED, 1, CV_AA);
		arrowedLine(imgBlobs, imagePoints[0], imagePoints[2], BLUE, 1, CV_AA);
		arrowedLine(imgBlobs, imagePoints[0], imagePoints[3], GREEN, 1, CV_AA);

		// Display the output frames.
		namedWindow("Cuboids");
		namedWindow("Blobs");
		moveWindow("Cuboids", 0, 0);
		moveWindow("Blobs", 640, 0);
		imshow("Cuboids", imgCuboids);
		imshow("Blobs", imgBlobs);

		/*imshow("Original", currentFrame);
		imshow("Grayscale", currentFrame_gray);
		imshow("Blur", currentFrame_blur);
		imshow("Difference", diff);
		imshow("Threshold", thresh);
		imshow("Morphed", morph);
		imshow("Contours", img_contours);
		imshow("Convex Hulls", img_convexHulls);*/

		// Clone the second frame and set it equal to the first frame.
		currentFrame = nextFrame.clone();
		// Read a new frame and set it equal to the second frame.
		capture.read(nextFrame);

		// wait for key press
		key = waitKey(1000 / frameRate /*0*/);

		// If space bar is pressed print out the following frames. (To make it easy to save results.)
		if (key == 32)
		{
			imwrite("original-frame" + to_string(currentFrameCount) + ".jpg", currentFrame);
			imwrite("grayscale-frame" + to_string(currentFrameCount) + ".jpg", currentFrame_gray);
			imwrite("blur-frame" + to_string(currentFrameCount) + ".jpg", currentFrame_blur);
			imwrite("difference-frame" + to_string(currentFrameCount) + ".jpg", diff);
			imwrite("threshold-frame" + to_string(currentFrameCount) + ".jpg", thresh);
			imwrite("morphed-frame" + to_string(currentFrameCount) + ".jpg", morph);
			imwrite("contours-frame" + to_string(currentFrameCount) + ".jpg", img_contours);
			imwrite("convexhull-frame" + to_string(currentFrameCount) + ".jpg", img_convexHulls);
			imwrite("cuboids-frame" + to_string(currentFrameCount) + ".jpg", imgCuboids);
			imwrite("blobs-frame" + to_string(currentFrameCount) + ".jpg", imgBlobs);
		}
	}
	return 0;
}

///////////////////////----- MATCHING ALGORITHM -----//////////////////////////////

void matchBlobs(vector<Blob> &blobs)
{
	// Loop through all the tracks and set them as "not tracking".
	for (int i = 0; i < tracks.size(); i++)
	{
		tracks[i].setBoolTrackUpdated(false);
	}

	// Loop through all the blobs in the current frame.
	for (int i = 0; i < blobs.size(); i++)
	{

		double leastDistance = 100000;
		int index_leastDistance;
		Point center = blobs[i].getCenter();

		// Loop through all the tracks and find the track which has its last added blob's average flow head nearest to the center of the blob of the current loop.
		for (int j = 0; j < tracks.size(); j++)
		{
			double sumDistances = 0;

			vector<Point2f> flowHeads = tracks[j].getLastBlob().getFlowHeads();

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

		// If the least distance found is less than half of the diagonal for the blob's bounding box i.e. if the flow head falls on the bounding box, add the blob to the found track.
		if (leastDistance < blobs[i].getDiagonalSize() * 0.5)
		{
			tracks[index_leastDistance].add(blobs[i]);
		}

		// Or else it is a new track and create a new track.
		else
		{
			trackCount++;
			Track track(blobs[i], trackCount);
			tracks.push_back(track);
		}
	}
}

/////////////////////----- METHOD FOR DISPLAYING TRACKING INFORMATION -----///////////////////////

void displayInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, double fontScale, Scalar fontColor)
{
	rectangle(outputFrame, Point(8, 5), Point(180, 20), backgroundColor, -1, CV_AA);
	rectangle(outputFrame, Point(8, 5), Point(180, 20), fontColor, 1, CV_AA);
	putText(outputFrame, "Frame Count: " + to_string(currentFrameCount) + " / " + to_string((int)totalFrameCount), Point(10, 17), fontFace, 0.35, fontColor, fontScale, CV_AA);

	rectangle(outputFrame, Point(8, 25), Point(180, 40), backgroundColor, -1, CV_AA);
	rectangle(outputFrame, Point(8, 25), Point(180, 40), fontColor, 1, CV_AA);
	putText(outputFrame, "Track Count: " + to_string(trackCount), Point(10, 37), fontFace, fontScale, fontColor, 0.35, CV_AA);

	rectangle(outputFrame, Point(8, 45), Point(180, 60), backgroundColor, -1, CV_AA);
	rectangle(outputFrame, Point(8, 45), Point(180, 60), fontColor, 1, CV_AA);
	putText(outputFrame, "Being Tracked: " + to_string(tracks.size()), Point(10, 57), fontFace, fontScale, fontColor, 0.35, CV_AA);

	/*putText(outputFrame, "Vehicle Speed Estimation Using Optical Flow And 3D Modeling by Indrajeet Datta", Point(5, outputFrame.rows - 10), fontFace, 0.3, WHITE, fontScale, CV_AA);*/
}