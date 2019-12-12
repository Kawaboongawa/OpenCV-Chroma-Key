#include "backgroundExtractor.h"
#include "foregroundExtractor.h"
#include "alphaBlending.h"
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

typedef unsigned int uint;

std::string background_path = "../data/images/backgroundTest2.jpg";
std::string video_path = "../data/greenscreen2.mp4";

std::string windowName = "Chroma-keying";
cv::Vec3i rgb;
int tolerance = 10;
int softness = 20;
bool colorSelected = false;
bool vidStarted = false;
const uint kernelSize = 15;
const uint halfKernel = kernelSize / 2;
const uint isOdd = kernelSize % 2;
const uint alpha = 0;
float2 ranges[3];
cv::Mat masks[3];
cv::Mat background, foreground, frame, resultImage, bgrSelectedColors, fgrdSelectedColors, bgrInpainted;



void ComputeHSVRange()
{
	cv::Mat imhsv, normalizedHSV, alpha;
	cvtColor(resultImage, imhsv, COLOR_BGR2HSV);
	std::vector<cv::Mat> imhsvChannels(3);
	split(imhsv, imhsvChannels);
	//cv::multiply(imhsvChannels[0], cv::Scalar(1 / 255.f), alpha);
	cv::Mat hHist;
	bool uniform = true, accumulate = false;
	int histSize = 181;
	float range[] = { 0, 181}; //the upper boundary is exclusive
	const float* histRange = { range };

	for (int i = 0; i < 3; ++i)
		masks[i] = cv::Mat::zeros(cv::Size(resultImage.cols, resultImage.rows), CV_8UC1);
	/*
	** Hue
	*/
		calcHist(&imhsvChannels[0], 1, 0, Mat(), hHist, 1, &histSize, &histRange, uniform, accumulate);
		GaussianBlur(hHist, hHist, cv::Size(1, 7), 1);
		ranges[0] = BGRDExtractor::computeRange(hHist);
		BGRDExtractor::getMask(imhsvChannels[0], masks[0], ranges[0]);
	//ranges[0] = float2(50, 60);
	/*
	** Saturation
	*/
	histSize = 256;
	float range2[] = { 0, 256 };
	const float* histRange2 = { range2 };
	BGRDExtractor::applyMask(imhsvChannels[1], masks[0]);
	calcHist(&imhsvChannels[1], 1, 0, Mat(), hHist, 1, &histSize, &histRange2, uniform, accumulate);
	hHist.ptr<float>(0)[0] = 0;
	GaussianBlur(hHist, hHist, cv::Size(1, 7), 1);
	ranges[1] = BGRDExtractor::computeRange(hHist);
	//BGRDExtractor::getMask(imhsvChannels[1], masks[1], ranges[1]);
	/*
	** Value
	*/
	BGRDExtractor::applyMask(imhsvChannels[2], masks[0]);
	calcHist(&imhsvChannels[2], 1, 0, Mat(), hHist, 1, &histSize, &histRange2, uniform, accumulate);
	hHist.ptr<float>(0)[0] = 0;
	GaussianBlur(hHist, hHist, cv::Size(1, 7), 1);
	uchar tmp = imhsvChannels[2].ptr<uchar>(0)[0];
	ranges[2] = BGRDExtractor::computeRange(hHist);


}

void setColor(int x, int y)
{
	Vec3b intensity = frame.at<Vec3b>(y, x);
	rgb[2] = intensity.val[0];
	rgb[1] = intensity.val[1];
	rgb[0] = intensity.val[2];
	setTrackbarPos("red", windowName, rgb[0]);
	setTrackbarPos("green", windowName, rgb[1]);
	setTrackbarPos("blue", windowName, rgb[2]);
	ComputeHSVRange();
	colorSelected = true;
	
}

void applyChroma()
{
	cv::Mat hsvForeground, fgrdRes, bgrRes;
	cv::cvtColor(frame, hsvForeground, cv::COLOR_BGR2HSV);
	std::vector<cv::Mat> foregroundHsvChannels(3);
	split(hsvForeground, foregroundHsvChannels);

	std::vector<cv::Mat> masks(3);
	for (int i = 0; i < 3; ++i)
		masks[i] = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
	for (uint i = 0; i < 3; ++i)
		BGRDExtractor::getMask(foregroundHsvChannels[i], masks[i], ranges[i]);


	/*
	** TOO SLOW and not did not improve image quality
	*/
	//cv::Mat entropyMask = cv::Mat::zeros(cv::Size(resultImage.cols, resultImage.rows), CV_8UC1);
	//BGRDExtractor::computeEntropyMask(foregroundHsvChannels[2], entropyMask, 3, 1);

	/*
	** Extract Absolute Foreground
	*/
	cv::Mat bgrdMask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
	cv::Mat fgrdMask = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC1);
	FRGDExtractor::getHUEMask(foregroundHsvChannels[0], fgrdMask, ranges[0], ceil((static_cast<float>(tolerance) / 4) / 100 * 180));

	/*
	** Extract Reflective region
	*/
	cv::Mat grayConfidence = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8U);
	FRGDExtractor::computeGrayConfidence(grayConfidence, foregroundHsvChannels[1], foregroundHsvChannels[2]);
	cv::Mat colorlessForeground = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_8UC3);
	FRGDExtractor::extractColorlessForground(frame, colorlessForeground, grayConfidence, 13);


	/*
	** Apply masks
	*/
	frame.copyTo(fgrdRes, fgrdMask);
	foreground = fgrdRes | colorlessForeground;
	FRGDExtractor::greenSpillReduction(foreground);
	
	cv::Mat bgrMask = masks[0] & masks[1] & masks[2];


	if (fgrdSelectedColors.empty())
	{
		frame.copyTo(bgrRes, bgrMask);
		BGRDExtractor::backgroundPropagation(bgrRes, bgrInpainted, bgrMask);
		//AlphaBlend::ExtractDominantColors(bgrInpainted, bgrSelectedColors);
		AlphaBlend::ExtractDominantColors(foreground, fgrdSelectedColors, 10);
		fgrdSelectedColors /= 255;
		bgrInpainted.convertTo(bgrInpainted, CV_32FC3);
		bgrInpainted /= 255;
	}

	cv::Mat alpha = cv::Mat::zeros(cv::Size(frame.cols, frame.rows), CV_32FC3);
	cv::Mat bgrfloat, fgrdfloat;

	cv::Mat fltInput;
	frame.convertTo(fltInput, CV_32FC3);
	cv::Mat fgrdUnknownColors = cv::Mat::zeros(frame.size(), CV_32FC3);
		//Timer t("alphablend");
	{
		Timer t("alphablend");
		AlphaBlend::computeAlpha(fltInput / 255, fgrdSelectedColors, bgrInpainted,
			bgrMask, fgrdMask, alpha, fgrdUnknownColors);
	}
	{
		Timer t("conversion finale");
		fgrdUnknownColors *= 255;
		fgrdUnknownColors.convertTo(fgrdUnknownColors, CV_8UC3);
		foreground += fgrdUnknownColors;
		AlphaBlend::alphaBlend(foreground, background, alpha, resultImage);
	}
}



// function which will be called on mouse input
void removeBlemish(int action, int x, int y, int flags, void *userdata)
{

	if (action == EVENT_LBUTTONDOWN)
		vidStarted = true;

	// Action to be taken when left mouse button is pressed
	if (action == EVENT_RBUTTONDOWN && !colorSelected)
		setColor(x, y);

}

void ValueModified(int, void*)
{
	if (colorSelected)
		applyChroma();
}


int main(int argc, char** argv)
{

	// Read image
	background = imread(background_path);
	FRGDExtractor::createSaturationThreshMap();
	//cv::Mat hsvForeground, test2, foreResult;
	cv::Mat test = imread("../data/images/test2.png");
	resultImage = test.clone();
	frame = test.clone();
	//background = imread("../data/images/green.png");
	//background.convertTo(background, CV_32FC3);
	// Make a dummy image, will be useful to clear the drawing
	//dummy = source.clone();
	namedWindow(windowName, WINDOW_NORMAL);
	//cv::setWindowProperty(windowName, WND_PROP_FULLSCREEN, WINDOW_FULLSCREEN);
	// highgui function called when mouse events occur
	setMouseCallback("Chroma-keying", removeBlemish);

	cv::createTrackbar("Tolerance", windowName, &tolerance, 100, ValueModified);

	cv::createTrackbar("Softness", windowName, &softness, 100, ValueModified);

	// Create a VideoCapture object and open the input file
	// If the input is the web camera, pass 0 instead of the video file name
	VideoCapture cap(video_path);

	// Check if camera opened successfully
	if (!cap.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	int k = 0;
	// loop until escape character is pressed
	while (k != 27)
	{

		// Capture frame-by-frame
		if (vidStarted || frame.empty())
		{
			//cap >> frame;
			//frame.copyTo(resultImage);
		}

		// If the frame is empty, break immediately
		if (frame.empty())
			break;

		if (colorSelected)
		{
			Timer T("Result for one frame");
			applyChroma();
		}
		// Display the resulting frame
		imshow("Chroma-keying", resultImage);
		//imshow("Chroma-keying", resultImage);
		k = waitKey(20) & 0xFF;
	}

	// When everything done, release the video capture object
	cap.release();

	return 0;

}
