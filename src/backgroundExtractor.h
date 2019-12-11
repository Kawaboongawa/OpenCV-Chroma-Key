#pragma once

#include <vector>
#include "opencv2/opencv.hpp"
#include "tools.h"



namespace BGRDExtractor
{
	float2 computeRange(const cv::Mat& hist, float Vamp = 0.01f, float Tgrad = 0.1);
	float computeVariance(const cv::Mat& hist, int nit);
	void applyMask(cv::Mat& M, cv::Mat& mask);
	void getMask(cv::Mat& M, cv::Mat& mask, float2 T);
	void applyMaskInv(cv::Mat& M, cv::Mat& mask);
	void computeEntropyMask(const cv::Mat& M, cv::Mat& Mask, int kSize, float T);
	void backgroundPropagation(const cv::Mat& input, cv::Mat& output, const cv::Mat& bgrd);
};