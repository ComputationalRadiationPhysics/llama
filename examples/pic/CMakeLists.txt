cmake_minimum_required (VERSION 3.3)
project(llama-pic)

if (NOT TARGET llama::llama)
	find_package(llama REQUIRED)
endif()
find_package (Eigen3 CONFIG REQUIRED)

add_executable(${PROJECT_NAME} pic.cpp input.txt)
target_compile_features(${PROJECT_NAME} PRIVATE cxx_std_20)
target_link_libraries (${PROJECT_NAME} PRIVATE llama::llama Eigen3::Eigen)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
	target_compile_options(${PROJECT_NAME} PRIVATE
		-fno-math-errno # sqrt prevents vectorization otherwise
		-march=native
		-ffast-math
	)
endif()
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/input.txt" "${CMAKE_CURRENT_BINARY_DIR}/input.txt" COPYONLY)

install(FILES "${PROJECT_BINARY_DIR}/${PROJECT_NAME}" DESTINATION "bin")

