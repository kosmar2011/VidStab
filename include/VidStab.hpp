#ifndef VIDSTAB_H
#define VIDSTAB_H
#include <fftw3.h>

static const uint16_t height = 480;
static const uint16_t width = 640;
static const uint32_t N = height * width;

void cross_power_spectrum(fftw_complex* in1, fftw_complex* in2, fftw_complex* pc) {
    for (uint32_t i = 0; i < 2 * N - 1; i++) {
        //complex multiplication
        double a = in1[i][0];
        double c = in2[i][0]; // read from (and write to!) each non-local variable exactly once in every function.
        double b = in1[i][1];
        double d = in2[i][1];
        double m0 = a * c - b * d;
        double m1 = a * d + b * c;
        //absolute
        auto abs = std::sqrt(m0 * m0 + m1 * m1);
        //regularization
        pc[i][0] = m0 / abs;
        pc[i][1] = m1 / abs;
    }
}

#endif