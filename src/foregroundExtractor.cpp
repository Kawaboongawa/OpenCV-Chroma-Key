#include <algorithm>
#include "foregroundExtractor.h"

cv::Mat m_St;
const std::vector<float> ThreshSat = { 1, 1, 1, 0.85, 0.7, 0.55, 0.4, 0.35, 0.30, 0.3, 0.3, 0.275, 0.25, 0.225, 0.2, 0.2, 0.2, 0.15, 0.1, 0.1, 0.1 };


void FRGDExtractor::getHUEMask(const cv::Mat& M, cv::Mat& mask, float2 R, float T)
{
	for (int j = 0; j < M.rows; j++)
	{
		const uchar* Mj = M.ptr<uchar>(j);
		uchar* maskj = mask.ptr<uchar>(j);
		for (int i = 0; i < M.cols; i++)
		{
			uchar val = Mj[i];
			if (R.x > T && R.y < 180 - T)
				maskj[i] = (val <= R.x - T) || (val >= R.y + T) ? 1 : 0;
			else if (R.y > 180 - T)
				maskj[i] = (val >= R.y + T - 180 && val <= R.y - T) ? 1 : 0;
			else if (R.x < T)
				maskj[i] = (val >= R.y + T && val <= 180 + R.y - T) ? 1 : 0;
		}
	}
}

void FRGDExtractor::computeGrayConfidence(cv::Mat& GC, const cv::Mat& S, const cv::Mat& V)
{
	for (int j = 0; j <GC.rows; j++)
	{
		uchar* MGC = GC.ptr<uchar>(j);
		const uchar* MS = S.ptr<uchar>(j);
		const uchar* MV = V.ptr<uchar>(j);
		for (int i = 0; i < GC.cols; i++)
			MGC[i] = std::max(0.1f, m_St.ptr<float>(MV[i])[0] - static_cast<float>(MS[i]));
	}
}

void FRGDExtractor::extractColorlessForground(const cv::Mat& input, cv::Mat& output, const cv::Mat& GC, uchar T)
{
	for (int j = 0; j < input.rows; j++)
	{
		const cv::Vec3b * MI = input.ptr<cv::Vec3b >(j);
		cv::Vec3b* MO = output.ptr<cv::Vec3b>(j);
		const uchar* MGC = GC.ptr<uchar>(j);
		for (int i = 0; i < input.cols; i++)
		{
			if (MGC[i] >= T)
				MO[i] = MI[i];		
		}
	}
}

void FRGDExtractor::greenSpillReduction(cv::Mat& mat)
{
	for (int j = 0; j < mat.rows; j++)
	{
		cv::Vec3b* M = mat.ptr<cv::Vec3b>(j);
		for (int i = 0; i < mat.cols; i++)
		{
			uchar max = std::max<uchar>(M[i][0], M[i][2]);
			if (M[i][1] > max)
				M[i][1] = max;
		}
	}
}


void FRGDExtractor::createSaturationThreshMap()
{
	cv::Mat St(21, 1, CV_32F);
	for (uint i = 0; i < 21; ++i)
		St.ptr<float>(i)[0] = ThreshSat[i] * 255;
	cv::resize(St, m_St, cv::Size(1, 256));
	/*cv::Mat hHist;
	int histSize = 256;
	float range[] = { 0, 256 }; //the upper boundary is exclusive
	const float* histRange = { range };
	cv::calcHist(&m_St, 1, 0, cv::Mat(), hHist, 1, &histSize, &histRange, true, false);
	Tool::drawHist(m_St, 256);*/
}
