cmake_minimum_required(VERSION 2.8)
project( VidStab )

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package( OpenCV REQUIRED )
#find_package( FFTW3 REQUIRED )

add_library(utils src/utils.cpp)
target_include_directories(utils PRIVATE include/)
target_link_libraries( utils 
	${OpenCV_LIBS} 
	fftw3
)

#message("OpenCV_INCLUDE_DIRS: " ${OpenCV_INCLUDE_DIRS})
#message("OpenCV_LIBS: " ${OpenCV_LIBS})

add_executable( VidStab src/VidStab.cpp)
target_include_directories( VidStab PUBLIC ${OpenCV_INCLUDE_DIRS})
target_include_directories( VidStab PUBLIC fftw3 )
target_include_directories( VidStab PRIVATE include/  )

#TESTING LIBRARY USE (one cpp at a time)
#add_executable( VidStab src/test_fftw.cpp)
#add_executable( VidStab src/test_opencv.cpp )

target_link_libraries( VidStab 
	${OpenCV_LIBS} 
	fftw3
	utils
)
