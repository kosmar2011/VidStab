#include <iostream>
// #include <math.h>
#include <cmath>
#include <chrono> 
#include <vector> 

#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "VidStab.hpp"

int main() {
    cv::VideoCapture cap("../scene2.mpg");
    cv::Mat frame, g_frame0;

    cap >> frame;
    cv::cvtColor(frame, g_frame0, cv::COLOR_BGR2GRAY);

    cv::VideoWriter writer;
    int codec = cv::VideoWriter::fourcc('M', 'P', '4', '2');  // MPEG4
    std::string filename = "../ifft.avi";
    writer.open(filename, codec, cap.get(cv::CAP_PROP_FPS), frame.size() * 4, true);
    std::cout << cap.get(cv::CAP_PROP_FPS) << std::endl;
    if (!writer.isOpened()) {
        std::cout << "Could not open the output video for write.\n";
        return -1;
    }

    if (!cap.isOpened()) {
        std::cout << "Error opening video stream or file.\n";
        return -1;
    }

    auto fft_in    = fftw_alloc_real(N);
    auto fft0      = fftw_alloc_complex(N);
    auto fft0_conj = fftw_alloc_complex(N);


    fftw_plan plan_f = fftw_plan_dft_r2c_2d(height, width, fft_in, fft0, FFTW_ESTIMATE);


    for (uint32_t i = 0; i < N; i++) {
        fft_in[i] = static_cast<double>(g_frame0.data[i]);
    }

    fftw_execute(plan_f);
    fftw_destroy_plan(plan_f);

    conj(fft0, fft0_conj);

    fftw_free(fft_in);
    fftw_free(fft0);

    int i = 0;
    auto start_video = std::chrono::high_resolution_clock::now();

    while(1){
        cap >> frame;
        if (frame.empty())
            break;
        cv::Mat grayscale;
        cv::cvtColor(frame, grayscale, cv::COLOR_BGR2GRAY);
        std::cout << frame.size() << "\n";      

        auto fft_in   = fftw_alloc_real(N);
        auto fft_out  = fftw_alloc_complex(N);
        auto ph_cor   = fftw_alloc_complex(N);
        auto ifft_out = fftw_alloc_real(N);
        
        auto start = std::chrono::high_resolution_clock::now();

        fftw_plan plan_f = fftw_plan_dft_r2c_2d(height, width, fft_in,  fft_out, FFTW_ESTIMATE);
        fftw_plan plan_b = fftw_plan_dft_c2r_2d(height, width, ph_cor, ifft_out, FFTW_ESTIMATE);

        for (uint32_t i = 0; i < N; i++) {
            fft_in[i] = static_cast<double>(grayscale.data[i]);
        }

        //RUN FORWARD FFT COMPUTATION
        fftw_execute(plan_f);   

        auto p = fft0_conj;
        cross_power_spectrum(p, fft_out, ph_cor);

        //RUN BACKWARD FFT COMPUTATION
        fftw_execute(plan_b);

        auto img_ifft = cv::Mat(height, width, CV_32FC1);
        //normalize
        for (uint32_t i = 0; i < N; i++) {
            ifft_out[i] /= static_cast<double>(N);
        }
        // convert ifft fftw_complex* to cv::Mat
        for (uint32_t i = 0; i < N; i++) {
            img_ifft.data[i] = static_cast<double>(ifft_out[i]);
        }

        fft_shift(img_ifft);

        auto circ = cv::Mat(height * 2, width * 2, CV_8UC3, cv::Scalar(0, 0, 0));

        double minVal; 
        double maxVal; 
        cv::Point minLoc; 
        cv::Point maxLoc;
        cv::minMaxLoc(img_ifft, &minVal, &maxVal, &minLoc, &maxLoc);
        

        auto dx = maxLoc.x;
        auto dy = maxLoc.y;
        std::cout << "(" << dx << ", " << dy << ")\n";
        std::cout << "With value: " << maxVal << "\n";
        //the distance of dx point from the center
        dx = std::abs(dx - (short int)height/2);
        dy = std::abs(dy - (short int) width/2);

        for (auto i = dx; i < height + dx; i++) {
            for (auto j = dy; j < width + dy; j++) {
                circ.at<cv::Vec3b>(i, j)[0] = (short int)frame.at<cv::Vec3b>(i - dx, j - dy)[0];
                circ.at<cv::Vec3b>(i, j)[1] = (short int)frame.at<cv::Vec3b>(i - dx, j - dy)[1];
                circ.at<cv::Vec3b>(i, j)[2] = (short int)frame.at<cv::Vec3b>(i - dx, j - dy)[2];
                // std::cout << "all good in: (" << i << ", " << j << ")\n";
            }
        }
        std::cout << "good2\n";
        writer << circ;

        fftw_free(fft_in);
        fftw_free(fft_out);
        fftw_free(ifft_out);
        fftw_free(ph_cor);

        // fftw_destroy_plan(plan_f);
        // fftw_destroy_plan(plan_b);

        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "Time taken for frame " << i << " is: "
            << duration.count() << "\n";

        cv::imshow("ifft", circ);
        cv::waitKey(1);

        i++;
    }

    fftw_cleanup();

}