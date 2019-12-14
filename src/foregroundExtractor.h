#pragma once

#include <vector>
#include "opencv2/opencv.hpp"
#include "tools.h"



namespace FRGDExtractor
{
	void getHUEMask(const cv::Mat& M, cv::Mat& mask, float2 R, float T);
	void createSaturationThreshMap();
	void computeGrayConfidence(cv::Mat& GC, const cv::Mat& S, const cv::Mat& V);
	void extractColorlessForground(const cv::Mat& input, cv::Mat& output, const cv::Mat& GC, uchar T);
	void extractColorlessForgroundMask(cv::Mat& Mask, const cv::Mat& GC, uchar T);
	void greenSpillReduction(cv::Mat& mat);
};
