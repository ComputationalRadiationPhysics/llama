cmake_minimum_required (VERSION 3.15)
project(llama-hep_dimuon_analysis_cms)

set(CMAKE_CXX_STANDARD 17)

find_package(ROOT REQUIRED)
if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
add_executable(${PROJECT_NAME} hep_dimuon_analysis_cms.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ROOT::Hist ROOT::Graf ROOT::Gpad ROOT::ROOTNTuple llama::llama)
