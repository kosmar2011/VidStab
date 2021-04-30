#include "VidStab.hpp"

void cross_power_spectrum(fftw_complex* in1, fftw_complex* in2, fftw_complex* pc) {
    for (uint32_t i = 0; i < N; i++) {
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

void conj(fftw_complex* fft, fftw_complex* conj) {
    for (uint32_t i = 0; i < N; i++) {
        conj[i][0] = static_cast<double>(fft[i][0]);
        conj[i][1] = static_cast<double>(-fft[i][1]);
    }
}

void fft_shift(cv::Mat& img_ifft) {
    int cx = img_ifft.cols / 2;
    int cy = img_ifft.rows / 2;
    cv::Mat tmp;
    cv::Mat q0(img_ifft, cv::Rect( 0,  0, cx, cy));
    cv::Mat q1(img_ifft, cv::Rect(cx,  0, cx, cy));
    cv::Mat q2(img_ifft, cv::Rect( 0, cy, cx, cy));
    cv::Mat q3(img_ifft, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
