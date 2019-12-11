#include "alphaBlending.h"
#include <limits>

void AlphaBlend::alphaBlend(cv::Mat& foreground, cv::Mat& background, cv::Mat& alpha, cv::Mat& outImage)
{

	// Get floating point pointers to the data matrices
	/*float* fptr = reinterpret_cast<float*>(foreground.data);
	float* bptr = reinterpret_cast<float*>(background.data);
	float* aptr = reinterpret_cast<float*>(alpha.data);
	float* outImagePtr = reinterpret_cast<float*>(outImage.data);*/
	for (int j = 0; j < foreground.rows; j++)
	{
		const cv::Vec3b * MF = foreground.ptr<cv::Vec3b>(j);
		const cv::Vec3b * MB = background.ptr<cv::Vec3b>(j);
		const cv::Vec3f * MA = alpha.ptr<cv::Vec3f>(j);
		cv::Vec3b * MO = outImage.ptr<cv::Vec3b>(j);
		for (int i = 0; i < foreground.cols; i++)
		{
			auto f = [&](float F, float B, float A)
			{
				return (A * F) + (1 - A) * B;
			};
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
	/*int* tmp = labels.ptr<int>(0);
	auto str = Tool::type2str(centers.type());
	float* tmp2 = centers.ptr<float>(0);
	mat *= 0;
	cv::Point3f point = centers.ptr<cv::Point3f>(3)[0];
	cv::Vec3b result = cv::Vec3b(point.x, point.y, point.z);
	mat += result;*/
}

float computePixelAlpha(float C, float B, float F)
{
	if (F < 0.1f && B < 0.1f)
		return 0;
	return (abs((C - B) * (F - B))) / pow(F - B, 2);
}

float mag(const cv::Vec3f& vec)
{
	return sqrt(vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2]);
}


cv::Vec3f computePixelAlpha2(const cv::Vec3f& C, const cv::Vec3f& B, const cv::Vec3f& F)
{

	cv::Vec3f vec = F - B;
	float mag = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
	if (mag < 0.01f)
		return cv::Vec3f(0, 0, 0);
	cv::Vec3f res;
	cv::multiply(C - B, vec, res);
	res /= mag;
	return res;
}


float ComputeRd2(const cv::Vec3f& C, const cv::Vec3f& B, const cv::Vec3f& F, const cv::Vec3f& alpha)
{
	cv::Vec3f alphaDiff = cv::Vec3f(1 - alpha[0], 1 - alpha[1], 1 - alpha[2]);
	cv::Vec3f bgrd, fgrd;
	cv::multiply(alpha, F, fgrd);
	cv::multiply(alphaDiff, B, bgrd);
	cv::Vec3f res = C - (fgrd + bgrd);
	float num = mag(res);
	float div = mag(F - B);
	return num / div;
}

float ComputeRd(float C, float B, float F, float alpha)
{
	return abs(C - (alpha * F + (1 - alpha) * B)) / std::max<float>(0.01, abs(F - B));
}


void AlphaBlend::computeAlpha(const cv::Mat& img, const cv::Mat& frgdSelectedColors, const cv::Mat& brgdSelectedColors,
	const cv::Mat& bgrdMask, const cv::Mat& fgrdMask , cv::Mat& alpha, cv::Mat& foregroundSamples)
{
	cv::Mat linCost(frgdSelectedColors.rows, brgdSelectedColors.rows, CV_32F);
	cv::Mat alphas(frgdSelectedColors.rows, brgdSelectedColors.rows, CV_32FC3);
	cv::Point minLoc;
	double minVal;
	Timer t("for loop");
	for (int j = 0; j < img.rows; j++)
	{
		const cv::Vec3f* MI = img.ptr<cv::Vec3f>(j);
		cv::Vec3f * MA = alpha.ptr<cv::Vec3f>(j);
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
					for (int n = 0; n < linCost.cols; ++n)
					{
						const cv::Vec3f MF = frgdSelectedColors.ptr<cv::Vec3f>(m)[0];
						const cv::Vec3f MB = brgdSelectedColors.ptr<cv::Vec3f>(n)[0];
						/*cv::Vec3f alpha = cv::Vec3f(computePixelAlpha(MI[i][0], MB[0], MF[0]),
							computePixelAlpha(MI[i][1], MB[1], MF[1]),
							computePixelAlpha(MI[i][2], MB[2], MF[2]));*/
						float tmp = FLT_MAX;
						//auto test = computePixelAlpha2(cv::Vec3f(0.5, 0, 0), cv::Vec3f(0.9, 0, 0), cv::Vec3f(1, 0, 0));
						cv::Vec3f tmpAlpha = computePixelAlpha2(MI[i], MB, MF);
						if (tmpAlpha[0] >= 0 && tmpAlpha[1] >= 0 && tmpAlpha[2] >= 0)
						{
							float best = std::max(std::max(tmpAlpha[0], tmpAlpha[1]), tmpAlpha[2]);
							tmpAlpha = cv::Vec3f(best, best, best);
							tmp = ComputeRd2(MI[i], MB, MF, tmpAlpha);
						}
						linCost.ptr<float>(m)[n] = tmp;
						/*linCost.ptr<float>(m)[n] = (ComputeRd(MI[i][0], MB[0], MF[0], alpha[0])
							+ ComputeRd(MI[i][1], MB[1], MF[1], alpha[1])
							+ ComputeRd(MI[i][2], MB[2], MF[2], alpha[2]) / 3);*/
						alphas.ptr<cv::Vec3f>(m)[n] = tmpAlpha;
					}
				}
				cv::minMaxLoc(linCost, &minVal, nullptr, &minLoc, nullptr);
				cv::Vec3f finalAlpha = alphas.ptr<cv::Vec3f>(minLoc.y)[minLoc.x];
				MA[i] = finalAlpha;
				foregroundSamples.ptr<cv::Vec3f>(j)[i] = frgdSelectedColors.ptr<cv::Vec3f>(minLoc.y)[0];
			}
		}
	}
}
