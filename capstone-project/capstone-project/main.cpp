#include <opencv2/opencv.hpp>
#include <iostream>
#include <conio.h>
#include <string>
#include <iomanip>

using namespace cv;
using namespace std;

Mat cameraMatrix, distCoeffs, rotationVector, translationVector, rotationMatrix, inverseHomographyMatrix;

float frameRate, timeElapsed, totalFrameCount, frameHeight, frameWidth, fourCC;

int currentFrameCount;

const Scalar WHITE = Scalar(255, 255, 255), BLACK = Scalar(0, 0, 0), BLUE = Scalar(255, 0, 0), GREEN = Scalar(0, 255, 0), RED = Scalar(0, 0, 255), YELLOW = Scalar(0, 255, 255);

Mat currentFrame, nextFrame, currentFrame_gray, nextFrame_gray, currentFrame_blur, nextFrame_blur, morph, diff, thresh, videoMask, imgCuboids, imgTracks;

const Point3f cameraCenter = Point3f(1.80915, -8.95743, 8.52165);

const float initialCuboidLength = 5, initialCuboidWidth = 2, initialCuboidHeight = 1.5;

Point3f findWorldPoint(const Point2f &imagePoint, double zConst, const Mat &cameraMatrix, const Mat &rotationMatrix, const Mat &translationVector);


vector<vector<Point3f> >findFlowsProjectedOnPlanes(vector<vector<float> > planeParameters, vector<vector<Point3f> > planeVertices, vector<Point3f> groundPlaneFlowPoints, Point3f cameraCenter);

void displayInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, double fontScale, Scalar fontColor);

double distanceBetweenPoints(Point2f point1, Point point2);
bool pointInside(vector<Point3f> points, Point3f point);
float distanceBetweenPoints(Point3f point1, Point3f point2);

class Blob
{

private:

	vector<Point> contour_;
	Rect boundingRectangle_;
	float area_, width_, height_, diagonalSize_, aspectRatio_;
	Point  center_, bottomLeftCorner_, bottomRightCorner_, topLeftCorner_, topRightCorner_;
	vector<Point2f> flowTails_, flowHeads_;

public:

	Blob(const vector<Point> &contour);
	~Blob();

	//getter functions

	vector<Point> getContour() const { return contour_; }
	Rect getBoundingRect() const { return boundingRectangle_; }
	float getArea() const { return area_; }
	float getWidth() const { return width_; }
	float getHeight() const { return height_; }
	float getDiagonalSize() const { return diagonalSize_; }
	float getAspectRatio() const { return aspectRatio_; }
	Point getCenter() const { return center_; }
	Point getBottomLeftCorner() const { return bottomLeftCorner_; }
	Point getBottomRightCorner() const { return bottomRightCorner_; }
	Point getTopLeftCorner() const { return topLeftCorner_; }
	Point getTopRightCorner() const { return topRightCorner_; }
	vector<Point2f>getFlowTails() const { return flowTails_; }
	vector<Point2f> getFlowHeads() const { return flowHeads_; }

	void drawBlob(Mat &outputFrame, Scalar rectColor, int rectThickness, Scalar contourColor = BLUE, int contourThickness = 1, Scalar centerColor = GREEN, int centerThickness = 1);
	void drawBlobFlows(Mat &outputFrame, Scalar flowColor, int flowThickness);
	void drawBlobInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness);

};

Blob::Blob(const vector<Point> &contour)
{

	contour_ = contour;
	boundingRectangle_ = boundingRect(contour);
	area_ = boundingRectangle_.area();
	width_ = boundingRectangle_.width;
	height_ = boundingRectangle_.height;
	aspectRatio_ = (float)boundingRectangle_.width / (float)boundingRectangle_.height;
	diagonalSize_ = sqrt(pow(boundingRectangle_.width, 2) + pow(boundingRectangle_.height, 2));

	center_ = Point((boundingRectangle_.x + boundingRectangle_.width / 2), (boundingRectangle_.y + boundingRectangle_.height / 2));

	topRightCorner_ = Point(boundingRectangle_.x, boundingRectangle_.y);
	bottomRightCorner_ = Point(boundingRectangle_.x + boundingRectangle_.width, boundingRectangle_.y + boundingRectangle_.height);

	bottomLeftCorner_ = Point(boundingRectangle_.x, boundingRectangle_.y + boundingRectangle_.height);

	Mat mask(Size(frameWidth, frameHeight), CV_8UC1, BLACK);
	vector<vector<Point> > contours;
	contours.push_back(contour_);
	drawContours(mask, contours, -1, WHITE, -1, CV_AA);
	vector<Point2f> flowTails;
	goodFeaturesToTrack(currentFrame_gray, flowTails, 100, 0.01, 5, mask);
	cornerSubPix(currentFrame_gray, flowTails, Size(10, 10), Size(-1, -1), TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.03));
	flowTails_ = flowTails;
	int win_size = 10;

	vector<uchar> status;
	vector<float> error;
	calcOpticalFlowPyrLK(currentFrame_gray, nextFrame_gray, flowTails_, flowHeads_, status, error, Size(win_size * 2 + 1, win_size * 2 + 1), 5, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 20, 0.3));

}

Blob::~Blob() {}

void Blob::drawBlob(Mat &outputFrame, Scalar rectColor, int rectThickness, Scalar contourColor, int contourThickness, Scalar centerColor, int centerThickness)
{
	vector<vector<Point> > contours;
	contours.push_back(getContour());
	rectangle(outputFrame, getBoundingRect(), rectColor, 1, CV_AA);
	drawContours(outputFrame, contours, -1, contourColor, 1, CV_AA);
	circle(imgTracks, getCenter(), 1, centerColor, -1, CV_AA);
}

void Blob::drawBlobFlows(Mat &outputFrame, Scalar flowColor, int flowThickness)
{
	for (int i = 0; i < getFlowTails().size(); i++)
	{
		arrowedLine(outputFrame, getFlowTails()[i], getFlowHeads()[i], flowColor, flowThickness);
	}
}

void Blob::drawBlobInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness)
{
	rectangle(outputFrame, getTopRightCorner(), Point(getTopRightCorner().x + getWidth(), getTopRightCorner().y - getDiagonalSize() / 9), backgroundColor, -1, CV_AA);

	rectangle(outputFrame, Point(getBottomLeftCorner().x + 3, getBottomLeftCorner().y + 3), Point(getBottomLeftCorner().x + getWidth() - 3, getBottomLeftCorner().y + 40), backgroundColor, -1, CV_AA);

	putText(outputFrame, "(" + to_string(getBottomLeftCorner().x) + ", " + to_string(getBottomLeftCorner().y) + ")", Point(getBottomLeftCorner().x + 5, getBottomLeftCorner().y + 15), fontFace, getWidth() / 350, fontColor, fontThickness, CV_AA);


	putText(outputFrame, "Width: " + to_string((int)getWidth()), Point(getBottomLeftCorner().x + 5, getBottomLeftCorner().y + 25), fontFace, getWidth() / 350, fontColor, fontThickness, CV_AA);

	putText(outputFrame, "Height: " + to_string((int)getHeight()), Point(getBottomLeftCorner().x + 5, getBottomLeftCorner().y + 35), fontFace, getWidth() / 350, fontColor, fontThickness, CV_AA);
}
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
	vector<vector<Point3f> > planeVertices_;

public:

	Cuboid(Point3f &point, float length, float width, float height, float angleOfMotion);
	~Cuboid();

	float getLength() const { return length_; }
	float getWidth() const { return width_; }
	float getHeight() const { return height_; }
	float getAngleOfMotion() const { return angleOfMotion_; }
	Point3f getB1() const { return b1_; }
	Point3f getCentroid() const { return centroid_; }
	Point2f getProjectedCentroid() const { return projectedCentroid_; }
	vector<Point3f> getVertices() const { return vertices_; }
	vector <vector<Point3f> > getPlaneVertices() { return planeVertices_; }
	vector<Point2f> getProjectedVertices() const { return imagePlaneProjectedVertices_; }
	vector<vector<float> > getPlaneParameters() const { return planeParameters_; }

	void drawCuboid(Mat &outputFrame, Scalar color, int lineThickness);
};

Cuboid::Cuboid(Point3f &point, float initialLength, float intialWidth, float height, float angleOfMotion)
{
	length_ = initialLength;
	width_ = intialWidth;
	height_ = height;
	angleOfMotion_ = angleOfMotion;

	b1_ = Point3f(point.x, point.y, 0.0);

	b2_ = Point3f(point.x + (width_ * cos(angleOfMotion_)), point.y - (width_ * sin(angleOfMotion_)), 0.0);

	b3_ = Point3f(point.x + (length_ * sin(angleOfMotion_)), point.y + (length_ * cos(angleOfMotion_)), 0.0);

	b4_ = Point3f(point.x + (length_ * sin(angleOfMotion_)) + (width_ * cos(angleOfMotion_)), point.y + (length_ * cos(angleOfMotion_)) + (width_ * sin(angleOfMotion_)), 0.0);

	t1_ = Point3f(point.x, point.y, height_);

	t2_ = Point3f(point.x + (width_ * cos(angleOfMotion_)), point.y - (width_* sin(angleOfMotion_)), height_);

	t3_ = Point3f(point.x + (length_ * sin(angleOfMotion_)), point.y + (length_ * cos(angleOfMotion_)), height_);

	t4_ = Point3f(point.x + (length_ * sin(angleOfMotion_)) + (width_ * cos(angleOfMotion_)), point.y + (length_ * cos(angleOfMotion_)) + (width_ * sin(angleOfMotion_)), height_);

	vertices_.push_back(b1_);
	vertices_.push_back(b2_);
	vertices_.push_back(b3_);
	vertices_.push_back(b4_);
	vertices_.push_back(t1_);
	vertices_.push_back(t2_);
	vertices_.push_back(t3_);
	vertices_.push_back(t4_);

	vector<Point3f> frontPlaneVertices;
	frontPlaneVertices.push_back(b1_);
	frontPlaneVertices.push_back(t1_);
	frontPlaneVertices.push_back(t2_);
	frontPlaneVertices.push_back(b2_);

	planeVertices_.push_back(frontPlaneVertices);

	vector<Point3f> rightPlaneVertices;
	rightPlaneVertices.push_back(b3_);
	rightPlaneVertices.push_back(t3_);
	rightPlaneVertices.push_back(t1_);
	rightPlaneVertices.push_back(b1_);

	planeVertices_.push_back(rightPlaneVertices);

	vector<Point3f> backPlaneVertices;
	backPlaneVertices.push_back(b4_);
	backPlaneVertices.push_back(t4_);
	backPlaneVertices.push_back(t3_);
	backPlaneVertices.push_back(b3_);

	planeVertices_.push_back(backPlaneVertices);

	vector<Point3f> leftPlaneVertices;
	leftPlaneVertices.push_back(b2_);
	leftPlaneVertices.push_back(t2_);
	leftPlaneVertices.push_back(t4_);
	leftPlaneVertices.push_back(b4_);

	planeVertices_.push_back(leftPlaneVertices);

	vector<Point3f> topPlaneVertices;
	topPlaneVertices.push_back(t1_);
	topPlaneVertices.push_back(t3_);
	topPlaneVertices.push_back(t4_);
	topPlaneVertices.push_back(t2_);

	planeVertices_.push_back(topPlaneVertices);

	vector<Point3f> bottomPlaneVertices;
	bottomPlaneVertices.push_back(b1_);
	bottomPlaneVertices.push_back(b3_);
	bottomPlaneVertices.push_back(b4_);
	bottomPlaneVertices.push_back(b2_);

	planeVertices_.push_back(bottomPlaneVertices);

	centroid_ = Point3f((b1_.x + b2_.x) / 2, (b1_.y + b3_.y) / 2, height / 2);

	projectPoints(vertices_, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePlaneProjectedVertices_);

	vector<Point3f> objectPoints;

	vector<Point2f> imagePoints;

	objectPoints.push_back(centroid_);

	projectPoints(objectPoints, rotationVector, translationVector, cameraMatrix, distCoeffs, imagePoints);

	projectedCentroid_ = imagePoints[0];

	vector<float> frontPlaneParameters;
	frontPlaneParameters.push_back(-cos(angleOfMotion_));
	frontPlaneParameters.push_back(sin(angleOfMotion_));
	frontPlaneParameters.push_back(0);
	frontPlaneParameters.push_back((length_ / 2) + centroid_.x * cos(angleOfMotion_) - centroid_.y * sin(angleOfMotion_));

	planeParameters_.push_back(frontPlaneParameters);

	vector<float> rightPlaneParameters;
	rightPlaneParameters.push_back(cos(angleOfMotion_));
	rightPlaneParameters.push_back(sin(angleOfMotion_));
	rightPlaneParameters.push_back(0);
	rightPlaneParameters.push_back((width_) / 2 - centroid_.x * cos(angleOfMotion_) - centroid_.y * sin(angleOfMotion_));

	planeParameters_.push_back(rightPlaneParameters);

	vector<float> backPlaneParameters;
	backPlaneParameters.push_back(sin(angleOfMotion_));
	backPlaneParameters.push_back(-cos(angleOfMotion_));
	backPlaneParameters.push_back(0);
	backPlaneParameters.push_back((length_ / 2) - centroid_.x * sin(angleOfMotion_) + centroid_.y * cos(angleOfMotion_));

	planeParameters_.push_back(backPlaneParameters);

	vector<float> leftPlaneParameters;
	leftPlaneParameters.push_back(-cos(angleOfMotion_));
	leftPlaneParameters.push_back(-sin(angleOfMotion_));
	leftPlaneParameters.push_back(0);
	leftPlaneParameters.push_back((width_ / 2) + centroid_.x * cos(angleOfMotion_) + centroid_.y * sin(angleOfMotion_));

	planeParameters_.push_back(leftPlaneParameters);

	vector<float> topPlaneParameters;
	topPlaneParameters.push_back(0);
	topPlaneParameters.push_back(0);
	topPlaneParameters.push_back(-1);
	topPlaneParameters.push_back(centroid_.z + (height_ / 2));

	planeParameters_.push_back(topPlaneParameters);

	vector<float> bottomPlaneParameters;
	bottomPlaneParameters.push_back(0);
	bottomPlaneParameters.push_back(0);
	bottomPlaneParameters.push_back(1);
	bottomPlaneParameters.push_back(-(centroid_.z) + (height_ / 2));

	planeParameters_.push_back(bottomPlaneParameters);

}
void Cuboid::drawCuboid(Mat &outputFrame, Scalar color, int lineThickness)
{
	bool inFrame = true;

	for (int i = 0; i < imagePlaneProjectedVertices_.size(); i++)
	{
		if (imagePlaneProjectedVertices_[i].x > outputFrame.cols || imagePlaneProjectedVertices_[i].y > outputFrame.rows)
		{
			inFrame = true;
		}

		if (inFrame)
		{
			line(outputFrame, imagePlaneProjectedVertices_[0], imagePlaneProjectedVertices_[1], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[1], imagePlaneProjectedVertices_[3], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[3], imagePlaneProjectedVertices_[2], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[2], imagePlaneProjectedVertices_[0], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[4], imagePlaneProjectedVertices_[5], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[5], imagePlaneProjectedVertices_[7], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[7], imagePlaneProjectedVertices_[6], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[6], imagePlaneProjectedVertices_[4], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[0], imagePlaneProjectedVertices_[4], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[1], imagePlaneProjectedVertices_[5], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[2], imagePlaneProjectedVertices_[6], color, lineThickness, CV_AA);

			line(outputFrame, imagePlaneProjectedVertices_[3], imagePlaneProjectedVertices_[7], color, lineThickness, CV_AA);

			/*putText(outputFrame, "b1", imagePlaneProjectedVertices_[0], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "b2", imagePlaneProjectedVertices_[1], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "b3", imagePlaneProjectedVertices_[2], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "b4", imagePlaneProjectedVertices_[3], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "t1", imagePlaneProjectedVertices_[4], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "t2", imagePlaneProjectedVertices_[5], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "t3", imagePlaneProjectedVertices_[6], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);
			putText(outputFrame, "t4", imagePlaneProjectedVertices_[7], CV_FONT_HERSHEY_SIMPLEX, 0.25, WHITE, 0.25, CV_AA);*/
		}
	}
}
Cuboid::~Cuboid() {};

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
Track::Track(const Blob &blob)
{
	blobs_.push_back(blob);

	Point3f point = findWorldPoint(blob.getBottomLeftCorner(), 0.0, cameraMatrix, rotationMatrix, translationVector);
	Cuboid cuboid(point, initialCuboidLength, initialCuboidWidth, initialCuboidHeight, 0.0);
	cuboids_.push_back(cuboid);

	beingTracked_ = true;
	trackUpdated_ = false;
	trackColor_ = Scalar(rand() % 256, rand() % 256, rand() % 256);
	vector<Point2f> flowHeads, flowTails;
	flowHeads = blob.getFlowHeads();
	flowTails = blob.getFlowTails();

	vector<Point3f> groundPlaneFlowHeads, groundPlaneFlowTails;
	for (int i = 0; i < flowTails.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(flowTails[i]), 0.0, cameraMatrix, rotationMatrix, translationVector);

		groundPlaneFlowTails.push_back(point);
	}
	for (int i = 0; i < flowHeads.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(flowHeads[i]), 0.0, cameraMatrix, rotationMatrix, translationVector);

		groundPlaneFlowHeads.push_back(point);
	}

	double totalDx = 0, totalDy = 0;

	for (int i = 0; i < groundPlaneFlowTails.size(); i++)
	{
		double dx = groundPlaneFlowHeads[i].x - groundPlaneFlowTails[i].x;

		double dy = groundPlaneFlowHeads[i].y - groundPlaneFlowTails[i].y;

		totalDx = totalDx + dx;

		totalDy = totalDy + dy;
	}

	averageFlowDistanceX_ = totalDx / groundPlaneFlowTails.size();
	averageFlowDistanceY_ = totalDy / groundPlaneFlowTails.size();
	angleOfMotion_ = atan(averageFlowDistanceX_ / averageFlowDistanceY_);



	planeProjectedFlowTails_ = findFlowsProjectedOnPlanes(cuboid.getPlaneParameters(), cuboid.getPlaneVertices(), groundPlaneFlowTails, cameraCenter);

	planeProjectedFlowHeads_ = findFlowsProjectedOnPlanes(cuboid.getPlaneParameters(), cuboid.getPlaneVertices(), groundPlaneFlowHeads, cameraCenter);

}
void Track::add(const Blob &blob)
{
	blobs_.push_back(blob);
	Point3f point2(getRecentCuboid().getB1().x + averageFlowDistanceX_, getRecentCuboid().getB1().y + averageFlowDistanceY_, 0.0);
	Cuboid cuboid(point2, initialCuboidLength, initialCuboidWidth, initialCuboidHeight, angleOfMotion_);
	cuboids_.push_back(cuboid);
	vector<Point2f> flowHeads, flowTails;

	vector<Point3f> groundPlaneFlowHeads, groundPlaneFlowTails;

	flowHeads = blob.getFlowHeads();

	flowTails = blob.getFlowTails();

	for (int i = 0; i < flowTails.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(flowTails[i]), 0.0, cameraMatrix, rotationMatrix, translationVector);

		groundPlaneFlowTails.push_back(point);
	}
	for (int i = 0; i < flowHeads.size(); i++)
	{
		Point3f point = findWorldPoint(Point2f(flowHeads[i]), 0.0, cameraMatrix, rotationMatrix, translationVector);

		groundPlaneFlowHeads.push_back(point);
	}

	double totalDx = 0, totalDy = 0;

	for (int i = 0; i < groundPlaneFlowTails.size(); i++)
	{
		double dx = groundPlaneFlowHeads[i].x - groundPlaneFlowTails[i].x;

		double dy = groundPlaneFlowHeads[i].y - groundPlaneFlowTails[i].y;

		totalDx = totalDx + dx;

		totalDy = totalDy + dy;
	}
	averageFlowDistanceX_ = totalDx / groundPlaneFlowTails.size();

	averageFlowDistanceY_ = totalDy / groundPlaneFlowTails.size();


	angleOfMotion_ = atan(averageFlowDistanceX_ / averageFlowDistanceY_);

	Point3f point = findWorldPoint(blob.getBottomLeftCorner(), 0.0, cameraMatrix, rotationMatrix, translationVector);





	vector<vector<Point3f> > planeProjectedFlowTails;

	planeProjectedFlowTails_ = findFlowsProjectedOnPlanes(cuboid.getPlaneParameters(), cuboid.getPlaneVertices(), groundPlaneFlowTails, cameraCenter);

	planeProjectedFlowHeads_ = findFlowsProjectedOnPlanes(cuboid.getPlaneParameters(), cuboid.getPlaneVertices(), groundPlaneFlowHeads, cameraCenter);

	vector<vector<float> >  errors;
	for (int i = 0; i < planeProjectedFlowTails_.size(); i++)
	{

		if (planeProjectedFlowTails_[i].size() > 0)
		{
			vector<float> temp;
			for (int j = 0; j < planeProjectedFlowTails_[i].size(); j++)
			{
				Point3f point(planeProjectedFlowTails_[i][j].x + averageFlowDistanceX_, planeProjectedFlowTails_[i][j].y + averageFlowDistanceY_, planeProjectedFlowTails_[i][j].z);

				float error_distance = distanceBetweenPoints(point, planeProjectedFlowHeads_[i][j]);
				cout << "Projected Flow Tail: " << planeProjectedFlowTails_[i][j] << endl;
				cout << "Projected Flow Tail After Motion: " << point << endl;
				cout << "Projected Flow Head: " << planeProjectedFlowHeads_[i][j] << endl;
				cout << "Error Distance: " << error_distance << endl;
				cout << endl; cout << endl;
				temp.push_back(error_distance);
			}
			errors.push_back(temp);
		}
	}
	instantaneousSpeed_.push_back(sqrt(pow(averageFlowDistanceX_, 2) + pow(averageFlowDistanceY_, 2)) / (1 / frameRate) *  3.6);

	double total = 0;

	for (int i = 0; i < instantaneousSpeed_.size(); i++)
	{
		total += instantaneousSpeed_[i];
	}

	averageSpeed_ = total / instantaneousSpeed_.size();

}



void Track::drawBlobTrail(Mat &outputFrame, int trailLength, Scalar trailColor, int trailThickness)
{
	for (int i = 0; i < min((int)blobs_.size(), trailLength); i++)
	{
		line(outputFrame, blobs_.rbegin()[i].getCenter(), blobs_.rbegin()[i + 1].getCenter(), trailColor, trailThickness, CV_AA);
	}
}
void Track::drawCuboidTrail(Mat &outputFrame, int trailLength, Scalar trailColor, int trailThickness)
{

	for (int i = 0; i < min((int)cuboids_.size(), trailLength); i++)
	{
		bool inFrame = true;

		if (cuboids_.rbegin()[i].getProjectedCentroid().x > outputFrame.cols || cuboids_.rbegin()[i].getProjectedCentroid().y > outputFrame.rows)
		{
			inFrame = false;
		}
		if (inFrame)
		{
			line(outputFrame, cuboids_.rbegin()[i].getProjectedCentroid(), cuboids_.rbegin()[i + 1].getProjectedCentroid(), trailColor, trailThickness, CV_AA);
		}
	}
}
void Track::drawTrackInfoOnBlobs(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness)
{
	Blob blob = getRecentBlob();

	rectangle(outputFrame, blob.getTopRightCorner(), Point(blob.getTopRightCorner().x + blob.getWidth(), blob.getTopRightCorner().y - blob.getDiagonalSize() / 9), backgroundColor, -1, CV_AA);

	putText(outputFrame, "Track: " + to_string(trackNumber_), Point(blob.getTopRightCorner().x + 3, blob.getTopRightCorner().y - 3), fontFace, blob.getWidth() / 250, fontColor, fontThickness, CV_AA);
}
void Track::drawTrackInfoOnCuboids(Mat &outputFrame, Scalar backgroundColor, int fontFace, Scalar fontColor, int fontThickness)
{

	Cuboid cuboid = getRecentCuboid();

	Point point = Point((cuboid.getProjectedVertices()[0].x + cuboid.getProjectedVertices()[1].x) / 2, (cuboid.getProjectedVertices()[0].y + cuboid.getProjectedVertices()[0].y) / 2);

	circle(outputFrame, point, 10, trackColor_, -1, CV_AA);

	putText(outputFrame, to_string(trackNumber_), Point2f(point.x - 7, point.y + 3), fontFace, cuboid.getWidth() / 6, fontColor, fontThickness, CV_AA);

	putText(outputFrame, "I.S.: " + to_string(instantaneousSpeed_.back()) + " kmph", Point2f(cuboid.getProjectedVertices()[0].x + 3, cuboid.getProjectedVertices()[0].y + 23), fontFace, cuboid.getWidth() / 6, fontColor, fontThickness, CV_AA);

	putText(outputFrame, "A.S.: " + to_string(averageSpeed_) + " kmph", Point2f(cuboid.getProjectedVertices()[0].x + 3, cuboid.getProjectedVertices()[0].y + 33), fontFace, cuboid.getWidth() / 6, fontColor, fontThickness, CV_AA);
}
Track::~Track() {};

int trackCount = 0;

vector<Track> tracks;

void matchBlobs(vector<Blob> &blobs);

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

		_getch;

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


	while (capture.isOpened() && key != 27)
	{
		currentFrameCount = capture.get(CV_CAP_PROP_POS_FRAMES) - 1;

		timeElapsed = capture.get(CV_CAP_PROP_POS_MSEC) / 1000;

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

				Blob blob = tracks[i].getRecentBlob();

				Cuboid cuboid = tracks[i].getRecentCuboid();

				Scalar trackColor = tracks[i].getTrackColor();

				blob.drawBlob(imgTracks, trackColor, 2);

				blob.drawBlobFlows(imgTracks, GREEN, 1);

				blob.drawBlobInfo(imgTracks, trackColor, CV_FONT_HERSHEY_SIMPLEX, WHITE, 1);

				tracks[i].drawBlobTrail(imgTracks, 10, trackColor, 2);

				tracks[i].drawTrackInfoOnBlobs(imgTracks, trackColor, CV_FONT_HERSHEY_SIMPLEX, WHITE, 1);

				cuboid.drawCuboid(imgCuboids, trackColor, 1);

				tracks[i].drawCuboidTrail(imgCuboids, 10, trackColor, 2);

				tracks[i].drawTrackInfoOnCuboids(imgCuboids, trackColor, CV_FONT_HERSHEY_SIMPLEX, WHITE, 1);

				/*vector<vector<Point3f> > planeProjectedFlowTails = tracks[i].getPlaneProjectedFlowTails();
				vector<vector<Point3f> > planeProjectedFlowHeads = tracks[i].getPlaneProjectedFlowHeads();

				for (int j = 0; j < planeProjectedFlowTails.size(); j++)
				{
				if (planeProjectedFlowTails[j].size() > 0)
				{
				vector<Point2f> imagePoints;
				projectPoints(planeProjectedFlowTails[j], rotationVector, translationVector,	cameraMatrix, distCoeffs, imagePoints);
				for (int k = 0; k < imagePoints.size(); k++)
				{
				circle(imgCuboids, imagePoints[k], 1, GREEN, -1, CV_AA);
				}
				}
				}*/


			}

			if (tracks[i].isTrackUpdated() == false) tracks[i].incrementNoMatchCount();

			if (tracks[i].getNoMatchCount() > 10) tracks[i].setBoolBeingTracked(false);

			if (tracks[i].isBeingTracked() == false) tracks.erase(tracks.begin() + i);
		}

		displayInfo(imgTracks, BLACK, CV_FONT_HERSHEY_SIMPLEX, 0.35, WHITE);

		displayInfo(imgCuboids, BLACK, CV_FONT_HERSHEY_SIMPLEX, 0.35, WHITE);

		rectangle(imgTracks, Point((imgTracks.cols * 2 / 3) - 10, 0), Point(imgTracks.cols, 14), BLACK, -1, CV_AA);

		putText(imgTracks, "Vehicle Detection and Tracking", Point((imgTracks.cols * 2 / 3) - 5, 10), CV_FONT_HERSHEY_SIMPLEX, 0.35, GREEN, 0.35, CV_AA);

		imshow("Tracks", imgTracks);

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

			vector<Point2f> flowHeads = tracks[j].getRecentBlob().getFlowHeads();

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

double distanceBetweenPoints(Point2f point1, Point point2)
{
	point1 = (Point)point1;

	int intX = abs(point1.x - point2.x);

	int intY = abs(point1.y - point2.y);

	return(sqrt(pow(intX, 2) + pow(intY, 2)));
}

float distanceBetweenPoints(Point3f point1, Point3f point2)
{
	float x = abs(point1.x - point2.x);
	float y = abs(point1.y - point2.y);
	float z = abs(point1.z - point2.z);

	return(sqrt(pow(x, 2) + pow(y, 2) + pow(z, 2)));
}
vector<vector<Point3f> >findFlowsProjectedOnPlanes(vector<vector<float> > planeParameters, vector<vector<Point3f> > planeVertices, vector<Point3f> groundPlaneFlowPoints, Point3f cameraCenter)
{

	vector<vector<Point3f> > projectedFlowPointsOnPlanes;

	for (int i = 0; i < planeParameters.size(); i++)
	{
		vector<Point3f> points;
		for (int j = 0; j < groundPlaneFlowPoints.size(); j++)
		{
			float t = (-(planeParameters[i][0] * cameraCenter.x + planeParameters[i][1] * cameraCenter.y + planeParameters[i][2] * cameraCenter.z + planeParameters[i][3])) / planeParameters[i][0] * (groundPlaneFlowPoints[j].x - cameraCenter.x) + planeParameters[i][1] * (groundPlaneFlowPoints[j].y - cameraCenter.y) + planeParameters[i][2] * (-cameraCenter.z);


			Point3f projectedFlowPoint = Point3f(cameraCenter.x + (groundPlaneFlowPoints[j].x - cameraCenter.x) * t, cameraCenter.y + (groundPlaneFlowPoints[j].y - cameraCenter.y) * t, cameraCenter.z - cameraCenter.z * t);
			bool inside = pointInside(planeVertices[i], projectedFlowPoint);
			//cout << inside << endl;
			if (inside)
			{
				points.push_back(projectedFlowPoint);
				//cout << projectedFlowPoint << " is pushed backed." << endl;
				cout << endl;
			}
		}
		projectedFlowPointsOnPlanes.push_back(points);
	}

	return projectedFlowPointsOnPlanes;
}


void displayInfo(Mat &outputFrame, Scalar backgroundColor, int fontFace, double fontScale, Scalar fontColor)
{
	rectangle(outputFrame, Point(8, 20), Point(180, 35), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Frame Count: " + to_string(currentFrameCount) + " / " + to_string((int)totalFrameCount), Point(10, 30), fontFace, 0.35, fontColor, fontScale, CV_AA);
	rectangle(outputFrame, Point(8, 40), Point(180, 55), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Vehicle Count: " + to_string(trackCount), Point(10, 50), fontFace, fontScale, fontColor, 0.35, CV_AA);
	rectangle(outputFrame, Point(8, 60), Point(180, 75), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Being Tracked: " + to_string(tracks.size()), Point(10, 70), fontFace, fontScale, fontColor, 0.35, CV_AA);
	rectangle(outputFrame, Point(8, 80), Point(180, 95), backgroundColor, -1, CV_AA);
	putText(outputFrame, "Time Elapsed: " + to_string(timeElapsed), Point(10, 90), fontFace, fontScale, fontColor, 0.35, CV_AA);
	putText(outputFrame, "Vehicle Speed Estimation Using Optical Flow And 3D Modeling by Indrajeet Datta", Point(5, imgTracks.rows - 10), fontFace, 0.3, fontColor, fontScale, CV_AA);
}

Point3f findWorldPoint(const Point2f &imagePoint, double zConst, const Mat &cameraMatrix, const Mat &rotationMatrix, const Mat &translationVector)
{
	Mat imagePointHV = Mat::ones(3, 1, DataType<double>::type);
	imagePointHV.at<double>(0, 0) = imagePoint.x;
	imagePointHV.at<double>(1, 0) = imagePoint.y;

	Mat A, B;

	A = rotationMatrix.inv() * cameraMatrix.inv() * imagePointHV;
	B = rotationMatrix.inv() * translationVector;

	double p = A.at<double>(2, 0);
	double q = zConst + B.at<double>(2, 0);
	double s = q / p;

	Mat worldPointHV = rotationMatrix.inv() * (s * cameraMatrix.inv() * imagePointHV - translationVector);

	Point3f worldPoint;
	worldPoint.x = worldPointHV.at<double>(0, 0);
	worldPoint.y = worldPointHV.at<double>(1, 0);
	worldPoint.z = 0.0;

	return worldPoint;
}

bool pointInside(vector<Point3f> points, Point3f point)
{
	Point3f p1, p2, p3, p4, m;
	p1 = points[0];
	p2 = points[1];
	p3 = points[2];
	p4 = points[3];
	m = point;

	Vec3f v1(p2.x - p1.x, p2.y - p1.y, p2.z - p1.z);
	Vec3f v3(p4.x - p3.x, p4.y - p3.y, p4.z - p3.z);
	Vec3f v4(m.x - p1.x, m.y - p1.y, m.z - p1.z);
	Vec3f v5(m.x - p3.x, m.y - p3.y, m.z - p3.z);
	Vec3f v1_norm(v1[0] / (sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])), v1[1] / (sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])), v1[2] / (sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2])));

	Vec3f v3_norm(v3[0] / (sqrt(v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])), v3[1] / (sqrt(v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])), v3[2] / (sqrt(v3[0] * v3[0] + v3[1] * v3[1] + v3[2] * v3[2])));

	Vec3f v4_norm(v4[0] / (sqrt(v4[0] * v4[0] + v4[1] * v4[1] + v4[2] * v4[2])), v4[1] / (sqrt(v4[0] * v4[0] + v4[1] * v4[1] + v4[2] * v4[2])), v4[2] / (sqrt(v4[0] * v4[0] + v4[1] * v4[1] + v4[2] * v4[2])));

	Vec3f v5_norm(v5[0] / (sqrt(v5[0] * v5[0] + v5[1] * v5[1] + v5[2] * v5[2])), v5[1] / (sqrt(v5[0] * v5[0] + v5[1] * v5[1] + v5[2] * v5[2])), v5[2] / (sqrt(v5[0] * v5[0] + v5[1] * v5[1] + v5[2] * v5[2])));

	if (v1.dot(v4) >= 0 && v3.dot(v5) >= 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

