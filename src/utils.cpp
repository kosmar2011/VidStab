#include "VidStab.hpp"

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

void conj(fftw_complex* fft, fftw_complex* conj) {
    for (uint32_t i = 0; i < 2 * N - 1; i++) {
        conj[i][0] = static_cast<double>(fft[i][0]);
        conj[i][1] = static_cast<double>(-fft[i][1]);
    }
}

void fft(cv::Mat& img, double* data_in, fftw_complex* fft) {
    /*
    Input:
    img = the image we want to apply fft
    data_in = pointer to empty fftw_complex type for the fftw_plan (we apply the contents of the 3-Channel image)
    fft = pointer to empty fftw_complex type for the fftw_plan
    Output:
    img_fft = cv::Mat type image for fft contents/visualization
    */

    fftw_plan plan_f = fftw_plan_dft_r2c_2d(height, width, data_in, fft, FFTW_ESTIMATE);
    //fftw_plan plan_f = fftw_plan_dft_2d(height, width, data_in, fft, FFTW_FORWARD, FFTW_ESTIMATE);

    //assign input image data to fftw_real* data_in
    for (uint32_t i = 0; i < 2 * N - 1; i++) {
        data_in[i] = (i >= N) ? 0 : static_cast<double>(img.data[i]);
    }

    fftw_execute(plan_f);

    fftw_destroy_plan(plan_f);
    fftw_cleanup();
}

cv::Mat ifft(fftw_complex* fft, double* ifft) {
    /*
    Input:
    img_fft = the fourier transformed image we want to apply inverse fft
    fft = pointer to empty fftw_complex type for the fftw_plan
    ifft = pointer to empty fftw_complex type for the fftw_plan
    Output:
    img_ifft = cv::Mat type image for inverse fft contents/visualization
    */

    cv::Mat img_ifft(height, width, CV_32FC1);

    fftw_plan  plan_b = fftw_plan_dft_c2r_2d(height, width, fft, ifft, FFTW_ESTIMATE);
    //fftw_plan  plan_b = fftw_plan_dft_2d(height, width, fft, ifft, FFTW_BACKWARD, FFTW_ESTIMATE);

    fftw_execute(plan_b);

    ////normalize
    //for (uint32_t i = 0; i < N; i++) {
    //    ifft[i] /= N;
    //}

    // convert ifft fftw_complex* to cv::Mat
    for (uint32_t i = 0; i < 2 * N - 1; i++) {
        img_ifft.data[i] = ifft[i];
    }

    //imshow("IFFT", img_ifft);
    //waitKey(0);
    //destroyWindow("IFFT");


    fftw_destroy_plan(plan_b);
    fftw_cleanup();


    return img_ifft;
}

void fft_shift(cv::Mat& img_ifft) {
    int cx = img_ifft.cols / 2;
    int cy = img_ifft.rows / 2;
    cv::Mat tmp;
    cv::Mat q0(img_ifft, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(img_ifft, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(img_ifft, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(img_ifft, cv::Rect(cx, cy, cx, cy));

    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
