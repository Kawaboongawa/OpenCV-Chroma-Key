#include "tools.h"

void Tool::drawHist(cv::Mat hist, int histSize)
{
	// Draw the histograms for R, G and B
	int hist_w = 512; int hist_h = 400;
	cv::Mat histImage(hist_h, hist_w, CV_8UC1, cv::Scalar(0, 0, 0));
	int bin_w = cvRound((double)hist_w / histSize);
	normalize(hist, hist, 0, histImage.rows, cv::NORM_MINMAX, -1, cv::Mat());
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, cv::Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			cv::Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			cv::Scalar(255, 0, 0), 2, 8, 0);
	}
	cv::namedWindow("calcHist Demo");
	cv::imshow("calcHist Demo", histImage);
	cv::waitKey(0);
}

std::string Tool::type2str(int type)
{
	std::string r;

	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);

	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}

	r += "C";
	r += (chans + '0');

	return r;
}

void Tool::MaskFromImage(const cv::Mat& input, cv::Mat& output)
{
	for (int j = 0; j < input.rows; j++)
	{
		const cv::Vec3b * MI = input.ptr<cv::Vec3b>(j);
		uchar* MO = output.ptr<uchar>(j);
		for (int i = 0; i < input.cols; i++)
			if (MI[i] != cv::Vec3b(0, 0, 0))
				MO[i] = 1;
	}
}

Timer::Timer(std::string str)
: m_str(str)
{
	m_t = std::chrono::high_resolution_clock::now();

}

Timer::~Timer()
{
	auto end = std::chrono::high_resolution_clock::now();
	auto res = end - m_t;
	std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(end - m_t);
	std::cout << m_str << ": " << time_span.count() * 1000 << "ms" << std::endl;
}