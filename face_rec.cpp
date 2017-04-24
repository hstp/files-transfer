#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String path = "C:\\Users\\Fujinet\\Documents\\Visual Studio 2013\\Projects\\OpenCV-Demo\\x64\\Debug\\";
String face_cascade_name = "haarcascade_frontalface_alt.xml";
CascadeClassifier face_cascade;
String window_name = "Capture - Face detection";

/** @function main */
int main(void)
{
	VideoCapture capture;
	Mat frame;

	//-- 1. Load the cascades
	if (!face_cascade.load(path + face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };

	//-- 2. Read the video stream
	capture.open(0);
	if (!capture.isOpened()) { printf("--(!)Error opening video capture\n"); return -1; }

	while (capture.read(frame))
	{
		if (frame.empty())
		{
			printf(" --(!) No captured frame -- Break!");
			break;
		}
		std::vector<Rect> faces;
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		equalizeHist(frame_gray, frame_gray);
		face_cascade.detectMultiScale(frame_gray, faces, 1.3, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));
		for (size_t i = 0; i < faces.size(); i++)
		{
			Point point1(faces[i].x, faces[i].y);
			Point point2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
			rectangle(frame, point1, point2, Scalar(0, 0, 255), 2);
		}
		imshow(window_name, frame);
		int c = waitKey(30);
		if ((char)c == 'q') { break; } // escape
	}
	capture.release();
	return 0;
}
