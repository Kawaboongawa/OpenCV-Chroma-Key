#pragma once

#include <vector>
#include <chrono>
#include "opencv2/opencv.hpp"

typedef unsigned int uint;
typedef unsigned char uchar;

struct float2
{
	float2(float xu, float yu) : x(xu), y(yu) {}
	float2() {}
	float x;
	float y;
};

struct uchar3
{
	uchar3(uchar xu, uchar yu, uchar zu) : x(xu), y(yu), z(zu) {}
	uchar3() {}
	float x;
	float y;
	float z;
};


namespace Tool
{
	void drawHist(cv::Mat hist, int histSize);
	std::string type2str(int type);
	void MaskFromImage(const cv::Mat& input, cv::Mat& output);
	std::vector<std::pair<int, int>> generateArrays(int size);
}

class Timer
{
public:
	Timer(std::string str="");
	~Timer();
private:
	std::chrono::high_resolution_clock::time_point m_t;
	std::string m_str;
};