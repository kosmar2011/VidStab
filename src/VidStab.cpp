#include <iostream>
#include <math.h>
#include <chrono> 
#include <vector> 

#include "VidStab.hpp"

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