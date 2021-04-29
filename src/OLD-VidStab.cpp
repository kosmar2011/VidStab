#include <iostream>
#include <math.h>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include <fftw3.h>
#include <chrono> 
#include <vector> 


//const uchar channels = 3;
const uint16_t height = 480;
const uint16_t width = 640;
const uint32_t N = height * width;


inline void cross_power_spectrum(fftw_complex* in1, fftw_complex* in2, fftw_complex* pc) {
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

inline void conj(fftw_complex* fft, fftw_complex* conj) {
    for (uint32_t i = 0; i < 2 * N - 1; i++) {
        conj[i][0] = static_cast<double>(fft[i][0]);
        conj[i][1] = static_cast<double>(-fft[i][1]);
    }
}

//inline void fft(cv::Mat& img, double* data_in, fftw_complex* fft) {
inline void fft(cv::Mat& img, double* data_in, fftw_complex* fft) {
    /*
    Input:
    img = the image we want to apply fft
    data_in = pointer to empty fftw_complex type for the fftw_plan (we apply the contents of the 3-Channel image)
    fft = pointer to empty fftw_complex type for the fftw_plan
    Output:
    img_fft = cv::Mat type image for fft contents/visualization
    */

    // prepei na ginei padding me midenika!!!!!!!!!!!!!!!!!!
    // prepei na ginei padding me midenika!!!!!!!!!!!!!!!!!!
    // prepei na ginei padding me midenika!!!!!!!!!!!!!!!!!!
    // prepei na ginei padding me midenika!!!!!!!!!!!!!!!!!!
    // prepei na ginei padding me midenika!!!!!!!!!!!!!!!!!!

    fftw_plan plan_f = fftw_plan_dft_r2c_2d(height, width, data_in, fft, FFTW_ESTIMATE);
    //fftw_plan plan_f = fftw_plan_dft_2d(height, width, data_in, fft, FFTW_FORWARD, FFTW_ESTIMATE);

    //assign input image data to fftw_real* data_in
    for (uint32_t i = 0; i < N; i++) {
        data_in[i] = static_cast<double>(img.data[i]);
    }

    fftw_execute(plan_f);

    fftw_destroy_plan(plan_f);
    fftw_cleanup();
}


//inline cv::Mat ifft(fftw_complex* fft, double* ifft) { 
inline cv::Mat ifft(fftw_complex* fft, double* ifft) {
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

int main() {

    cv::VideoCapture cap("../scene2.mpg");
    cv::Mat frame;
    cv::Mat g_frame0;

    cap >> frame;

    cv::VideoWriter writer;
    int codec = cv::VideoWriter::fourcc('M', 'P', '4', '2');  // MPEG4
    std::string filename = "../ifft.avi";
    writer.open(filename, codec, cap.get(cv::CAP_PROP_FPS), frame.size(), true);

    std::cout << cap.get(cv::CAP_PROP_FPS) << std::endl;

    if (!writer.isOpened()) {
        std::cout << "Could not open the output video for write.\n";
        return -1;
    }

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file.\n";
        return -1;
    }

    cvtColor(frame, g_frame0, cv::COLOR_BGR2GRAY);
    auto fft_in = fftw_alloc_real(2 * N-1);
    auto fft0 = fftw_alloc_complex(2 * N - 1);
    auto fft0_conj = fftw_alloc_complex(2 * N - 1);


    fft(g_frame0, fft_in, fft0);
    conj(fft0, fft0_conj);
    fftw_free(fft0);
    fftw_free(fft_in); //An thelo na to xrisimopoiiso prepei na to kano comment

    uint16_t i = 0;
    auto start_video = std::chrono::high_resolution_clock::now();

    //while (1) {
    while (1) {
        cap >> frame;
        if (frame.empty())
            break;
        cv::Mat grayscale;
        cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
        auto fft_in = fftw_alloc_real(2 * N - 1);
        auto fft_out = fftw_alloc_complex(2 * N - 1);
        auto ifft_out = fftw_alloc_real(2 * N - 1);
        auto pc = fftw_alloc_complex(2 * N - 1);


        //phase_correlation
        auto start = std::chrono::high_resolution_clock::now();
        cv::resize(grayscale, grayscale, grayscale.size() * 2, 0, 0, cv::INTER_LINEAR);
        fft(grayscale, fft_in, fft_out);
        auto p = fft0_conj;
        cross_power_spectrum(p, fft_out, pc);
        auto img_ifft = ifft(pc, ifft_out);


        //fft_shift(img_ifft);

        //end fftshift

        cv::Point maxLoc;
        minMaxLoc(img_ifft, NULL, NULL, NULL, &maxLoc);

        cv::Mat circ(height, width*2, CV_8UC3, cv::Scalar(0, 0, 0));

        auto dx = maxLoc.x;
        auto dy = maxLoc.y;
        std::cout << dx << "\n";
        std::cout << dy << "\n";
        //}


        for (auto i = 3 * width * dy + 3 * dx + 1, k=0; i < 2 * N - 1; i++, k++) {
            circ.data[i] = frame.data[k];
        }
        //writer << img_ifft;

        //fftw_free(fft0_conj);
        fftw_free(fft_in);
        fftw_free(fft_out);
        fftw_free(ifft_out);
        fftw_free(pc);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time taken for frame " << i << " is: "
            << duration.count() << "\n";

        cv::imshow("ifft", circ);
        cv::waitKey(1);

        i++;
    }


    auto stop_video = std::chrono::high_resolution_clock::now();
    auto duration_video = std::chrono::duration_cast<std::chrono::microseconds>(stop_video - start_video);
    std::cout << "Time taken for 50 frames: " << duration_video.count() << "ms.\n";

    cap.release();
    writer.release();
    cv::destroyAllWindows();
    return 0;
}