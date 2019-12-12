#pragma once

#include "opencv2/opencv.hpp"
#include "tools.h"

namespace AlphaBlend
{
	void alphaBlend(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& outImage);
	void ExtractDominantColors(const cv::Mat& mat, cv::Mat& centers, int nClusters = 10);
	void computeAlpha(const cv::Mat& img, const cv::Mat& frgdSelectedColors, const cv::Mat& brgdSelectedColors,
		const cv::Mat& bgrdMask, const cv::Mat& fgrdMask, cv::Mat& alpha, cv::Mat& foregroundSamples);
	void knownRegionExpansion(const cv::Mat& img, cv::Mat& fgrdMask, cv::Mat& bgrdMask, int ki, int kc);
}