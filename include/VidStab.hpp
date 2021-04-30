
#ifndef VIDSTAB_H
#define VIDSTAB_H
#include <fftw3.h>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <fftw3.h>

static const uint16_t height = 480;
static const uint16_t width = 640;
static const uint32_t N = height * width;

void cross_power_spectrum(fftw_complex* in1, fftw_complex* in2, fftw_complex* pc);
void conj(fftw_complex* fft, fftw_complex* conj);
void fft(cv::Mat& img, double* data_in, fftw_complex* fft);
cv::Mat ifft(fftw_complex* fft, double* ifft);
void fft_shift(cv::Mat& img_ifft);

#endif