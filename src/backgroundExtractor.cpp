#include "backgroundExtractor.h"


bool check_condition1(float Vamp, float v0, float vk)
{
	return vk < Vamp * v0;
}

bool check_condition2(float vk, float vkPrec, float Tgrad)
{
	return ((abs(vk - vkPrec) / vkPrec) < Tgrad) || vk == 0;
}

float2 BGRDExtractor::computeRange(const cv::Mat& hist, float Vamp, float Tgrad)
{
	int Tmin = 0;
	int Tmax = 1;
	int nit = 0;
	cv::Mat Ck = hist.clone();
	cv::Mat R = cv::Mat::zeros(cv::Size(hist.cols, hist.rows), hist.type());
	float v0 = computeVariance(hist, nit);
	float vk = v0;
	float vkPrec;
	cv::Point maxLoc, minLoc;
	do
	{
		nit++;
		double maxVal;
		cv::minMaxLoc(Ck, nullptr, &maxVal, nullptr, &maxLoc);
		R.ptr<float>(maxLoc.y)[maxLoc.x] = maxVal;
		Ck.ptr<float>(maxLoc.y)[maxLoc.x] = 0;
		vkPrec = vk;
		vk = computeVariance(Ck, nit);
	} while (!check_condition1(Vamp, v0, vk) || !check_condition2(vk, vkPrec, Tgrad));


	cv::minMaxLoc(R, nullptr, nullptr, &minLoc, &maxLoc);
	R.ptr<float>(minLoc.y)[minLoc.x] = 0;
	nit--;
	Tmin = maxLoc.y;
	Tmax = maxLoc.y;
	while (Tmin > 0)
	{
		if (R.ptr<float>(Tmin - 1)[0] == 0)
			break;
		Tmin--;
	}
	while (Tmax < hist.rows - 1)
	{
		if (R.ptr<float>(Tmax + 1)[0] == 0)
			break;
		Tmax++;
	}
	//drawHist2(hist, 180);
	//drawHist2(Ck, 180);
	//drawHist2(R, 180);
	return float2(Tmin, Tmax);
}

float BGRDExtractor::computeVariance(const cv::Mat& hist, int nit)
{
	float vk;
	float* data = (float*) hist.data;
	cv::Scalar sum = cv::sum(hist);
	float mean = sum[0] / hist.rows;
	float res = 0;
	for (int i = 0; i < hist.rows; ++i)
		res += pow((data[i] - mean), 2);
	float Nk = hist.rows - nit;
	if (Nk <= 0)
		Nk = 1;
	return res / Nk;
}


void BGRDExtractor::getMask(cv::Mat& M, cv::Mat& mask, float2 T)
{
	for (int j = 0; j < M.rows; j++)
	{
		uchar* Mj = M.ptr<uchar>(j);
		uchar* maskj = mask.ptr<uchar>(j);
		for (int i = 0; i < M.cols; i++)
			if (Mj[i] >= T.x && Mj[i] <= T.y)
				maskj[i] = 1;
	}
}

void BGRDExtractor::applyMask(cv::Mat& M, cv::Mat& mask)
{
	for (int j = 0; j < M.rows; j++)
	{
		uchar* Mj = M.ptr<uchar>(j);
		uchar* maskj = mask.ptr<uchar>(j);
		for (int i = 0; i < M.cols; i++)
			if (maskj[i] == 0)
				Mj[i] = 0;
	}
}

void BGRDExtractor::applyMaskInv(cv::Mat& M, cv::Mat& mask)
{
	for (int j = 0; j < M.rows; j++)
	{
		uchar* Mj = M.ptr<uchar>(j);
		uchar* maskj = mask.ptr<uchar>(j);
		for (int i = 0; i < M.cols; i++)
			if (maskj[i] != 0)
				Mj[i] = 0;
	}
}

void BGRDExtractor::computeEntropyMask(const cv::Mat& M, cv::Mat& Mask, int kSize, float T)
{
	int kHalf = kSize / 2;
	for (uint j = kHalf; j < M.rows - kSize; ++j)
	{
		for (uint i = kHalf; i < M.cols - kSize; ++i)
		{
			float res = 0;
			for (uint m = j - kHalf; m < j + kHalf; ++m)
			{
				const uchar* Brow = M.ptr<uchar>(m);
				for (uint n = i - kHalf; n < i + kHalf; ++n)
				{
					float val = static_cast<float>(Brow[n]) / 255;;
					res += val * log2f(val);
				}
			}
			res *= -1;
			Mask.ptr<uchar>(j)[i] = (res < T);
		}
	}
}

void createWMat(const cv::Mat& mat, cv::Mat& W, int kSize)
{
	uint kHalf = kSize / 2;
	size_t wsize = mat.rows * mat.cols;
	W = cv::Mat::zeros(wsize, wsize, CV_32F);
	for (int j = 0; j < wsize - kSize; ++j)
	{
		float * MW = W.ptr<float>(j);
		int posX = j / mat.cols;
		int posY = j % mat.cols;
		for (int i = 0; i < wsize; ++i)
		{
			if (j / mat.cols < kHalf || j / mat.cols > mat.rows - (kSize + 1)
				|| j % mat.rows < kHalf || j % mat.rows > mat.cols - (kSize + 1))
				continue;
			int tmpY = i / mat.cols;
			int tmpX = i % mat.cols;
			if (abs(posX - tmpX) <= kHalf && abs(posY - tmpY) <= kHalf)
			{
				if (tmpX == 0 && tmpY == 0)
					MW[i] = 1.f / 3.f;
				else if (tmpX == 0)
					MW[i] = 1.f / 5.f;
				else if (tmpX == 0)
					MW[i] = 1.f / 5.f;
				else
					MW[i] = 1.f / 8.f;
			}
		}

	}
}

void BGRDExtractor::backgroundPropagation(const cv::Mat& input, cv::Mat& output, const cv::Mat& bgrdMask)
{
	cv::inpaint(input, (1 - bgrdMask), output, 3, cv::INPAINT_TELEA);
}