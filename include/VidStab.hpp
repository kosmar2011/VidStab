
#ifndef VIDSTAB_H
#define VIDSTAB_H

#include <fftw3.h>
#include <opencv2/core.hpp>

static const uint16_t height = 480;
static const uint16_t width = 640;
static const uint32_t N = height * width;

void cross_power_spectrum(fftw_complex* in1, fftw_complex* in2, fftw_complex* pc);
void conj(fftw_complex* fft, fftw_complex* conj);
void fft_shift(cv::Mat& img_ifft);
void circ_shift(cv::Mat out, const cv::Mat in, int xshift, int yshift);
#endif