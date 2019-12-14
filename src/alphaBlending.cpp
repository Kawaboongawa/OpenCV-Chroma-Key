#include "alphaBlending.h"
#include <limits>


void AlphaBlend::alphaBlend(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& outImage)
{
	auto f = [&](float F, float B, float A)
	{
		return (A * F) + (1 - A) * B;
	};

	for (int j = 0; j < foreground.rows; j++)
	{
		const cv::Vec3b * MF = foreground.ptr<cv::Vec3b>(j);
		const cv::Vec3b * MB = background.ptr<cv::Vec3b>(j);
		const cv::Vec3f * MA = alpha.ptr<cv::Vec3f>(j);
		cv::Vec3b * MO = outImage.ptr<cv::Vec3b>(j);
		for (int i = 0; i < foreground.cols; i++)
		{
			MO[i] = cv::Vec3b(f(MF[i][0], MB[i][0], MA[i][0]),
				f(MF[i][1], MB[i][1], MA[i][1]),
				f(MF[i][2], MB[i][2], MA[i][2]));
		}
	}
}


void AlphaBlend::ExtractDominantColors(const cv::Mat& mat, cv::Mat& centers, int nClusters)
{
	cv::Mat fltMat;
	mat.convertTo(fltMat, CV_32FC3);
	fltMat = fltMat.reshape(1, mat.rows * mat.cols);

	cv::Mat labels;
	double compactness = cv::kmeans(fltMat, nClusters, labels,
		cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0),
		3, cv::KMEANS_PP_CENTERS, centers);
	auto tmp = Tool::type2str(centers.type());
	centers.reshape(3, nClusters);
}

float mag(const cv::Vec3f& vec)
{
	return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}


cv::Vec3f computePixelAlpha(const cv::Vec3f& C, const cv::Vec3f& B, const cv::Vec3f& F)
{

	cv::Vec3f vec = F - B;
	float mag = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
	cv::Vec3f res;
	cv::multiply(C - B, vec, res);
	res = cv::Vec3f(abs(res[0]), abs(res[1]), abs(res[2]));
	res /= mag;
	return res;
}


float ComputeRd(const cv::Vec3f& C, const cv::Vec3f& B, const cv::Vec3f& F, float alpha)
{
	cv::Vec3f fgrd = alpha * F;
	cv::Vec3f bgrd = (1 - alpha) * B;
	cv::Vec3f res = C - (fgrd + bgrd);
	float num = mag(res);
	float div = mag(F - B);
	return num / div;
}

void AlphaBlend::computeAlpha(const cv::Mat& img, const cv::Mat& frgdSelectedColors, const cv::Mat& brgdSelectedColors,
	const cv::Mat& bgrdMask, const cv::Mat& fgrdMask, cv::Mat& alpha, cv::Mat& foregroundSamples)
{
	cv::Mat linCost(frgdSelectedColors.rows, 1, CV_32F);
	cv::Mat alphas(frgdSelectedColors.rows, 1, CV_32F);
	cv::Point minLoc;
	double minVal;
	for (int j = 0; j < img.rows; j++)
	{
		const cv::Vec3f* MI = img.ptr<cv::Vec3f>(j);
		cv::Vec3f * MA = alpha.ptr<cv::Vec3f>(j);
		const cv::Vec3f* MB = brgdSelectedColors.ptr<cv::Vec3f>(j);
		const uchar* MFk = fgrdMask.ptr<uchar>(j);
		const uchar* MBk = bgrdMask.ptr<uchar>(j);
		for (int i = 0; i < img.cols; i++)
		{
			if (MFk[i] == 1)
				MA[i] = cv::Vec3f(1.f, 1.f, 1.f);
			else if (MBk[i] == 1)
				MA[i] = cv::Vec3f(0.f, 0.f, 0.f);
			else
			{
				for (int m = 0; m < linCost.rows; ++m)
				{
					const cv::Vec3f MF = frgdSelectedColors.ptr<cv::Vec3f>(m)[0];
					cv::Vec3f tmpAlpha = computePixelAlpha(MI[i], MB[i], MF);
					float alpha = std::max(std::max(tmpAlpha[0], tmpAlpha[1]), tmpAlpha[2]);
					linCost.ptr<float>(m)[0] = ComputeRd(MI[i], MB[i], MF, alpha);;
					alphas.ptr<float>(m)[0] = alpha;
				}
				cv::minMaxLoc(linCost, &minVal, nullptr, &minLoc, nullptr);
				float finalAlpha = alphas.ptr<float>(minLoc.y)[minLoc.x];
				MA[i] = cv::Vec3f(finalAlpha, finalAlpha, finalAlpha);
				foregroundSamples.ptr<cv::Vec3f>(j)[i] = frgdSelectedColors.ptr<cv::Vec3f>(minLoc.y)[0];
			}
		}
	}
}


float euclideanDist(cv::Vec3f a, cv::Vec3f b)
{
	return sqrt(pow(a[0] - b[0], 2) + pow(a[1] - b[1], 2) + pow(a[2] - b[2], 2));
}

void AlphaBlend::knownRegionExpansion(const cv::Mat& img, const cv::Mat& unknowMask, cv::Mat& fgrdMask, cv::Mat& bgrdMask, float kc, const std::vector<std::pair<int, int>>& pos)
{
	for (int j = 0; j < img.rows; j++)
	{
		const cv::Vec3b* MI = img.ptr<cv::Vec3b>(j);
		uchar* MF = fgrdMask.ptr<uchar>(j);
		uchar* MB = bgrdMask.ptr<uchar>(j);
		const uchar* MU = unknowMask.ptr<uchar>(j);

		for (int i = 0; i < img.cols; i++)
		{
			if (!MF[i] && !MB[i])
			{
				for (auto e : pos)
				{
					int y = std::max(0, std::min(j + e.first, img.rows - 1));
					int x = std::max(0, std::min(i + e.second, img.cols - 1));
					if (unknowMask.ptr<uchar>(y)[x] == 1)
						continue;
					float dist = euclideanDist(MI[i], img.ptr<cv::Vec3b>(y)[x]) / 3;
					if (dist <= kc)
					{
						if (fgrdMask.ptr<uchar>(y)[x] == 1)
							MF[i] = 1;
						else
							MB[i] = 1;
					}

				}
			}
		}
	}
}