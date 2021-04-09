cmake_minimum_required (VERSION 3.16)
project (llama CXX)

# llama
find_package(Boost 1.70.0 REQUIRED)
find_package(fmt CONFIG REQUIRED)
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include> $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)
target_link_libraries(${PROJECT_NAME} INTERFACE Boost::headers fmt::fmt)

# llama::llama to make subdirectory projects work
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# llama IDE target to make source browsable/editable in IDEs
file(GLOB_RECURSE llamaSources "${CMAKE_CURRENT_SOURCE_DIR}/include/**")
add_custom_target("llamaIde" SOURCES ${llamaSources})
source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/include/llama" FILES ${llamaSources})

# tests
include(CTest)
if (BUILD_TESTING)
	find_package(Catch2 2.6.1 CONFIG QUIET)
	if (NOT ${Catch2_FOUND})
		message(FATAL_ERROR "Catch2 is required for building the tests. Either install Catch2 or disable tests by passing -DBUILD_TESTING=OFF to CMake.")
	endif()

	file(GLOB_RECURSE testSources "${CMAKE_CURRENT_SOURCE_DIR}/tests/**")
	add_executable(tests ${testSources})
	source_group(TREE "${CMAKE_CURRENT_LIST_DIR}/tests" FILES ${testSources})
	target_compile_features(tests PRIVATE cxx_std_20)
	if (MSVC)
		target_compile_options(tests PRIVATE /permissive- /constexpr:steps10000000 /diagnostics:caret)
	endif()
	target_link_libraries(tests PRIVATE Catch2::Catch2 llama::llama)

	option(ASAN_FOR_TESTS "Enables address sanitizer for tests" OFF)
	if (ASAN_FOR_TESTS)
		if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
			target_compile_options(tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
			target_link_options   (tests PRIVATE -fsanitize=address -fno-omit-frame-pointer)
		elseif(MSVC)
			target_compile_options(tests PRIVATE /fsanitize=address)
			target_link_options   (tests PRIVATE /wholearchive:clang_rt.asan_dynamic-x86_64.lib /wholearchive:clang_rt.asan_dynamic_runtime_thunk-x86_64.lib)
		endif()
	endif()
endif()

# examples
option(LLAMA_BUILD_EXAMPLES "Building (and installing) the examples" ON)
if (LLAMA_BUILD_EXAMPLES)
	# general examples
	add_subdirectory("examples/simpletest")
	add_subdirectory("examples/vectoradd")
	add_subdirectory("examples/nbody")
	add_subdirectory("examples/nbody_benchmark")
	add_subdirectory("examples/heatequation")
	add_subdirectory("examples/viewcopy")
	add_subdirectory("examples/bufferguard")
	add_subdirectory("examples/raycast")

	# alpaka examples
	find_package(alpaka 0.7.0 QUIET)
	if (_ALPAKA_FOUND)
		add_subdirectory("examples/alpaka/nbody")
		add_subdirectory("examples/alpaka/vectoradd")
		add_subdirectory("examples/alpaka/asyncblur")
	elseif()
		message(WARNING "Could not find alpaka. Alpaka examples are disabled.")
	endif()

	# ROOT examples
	find_package(ROOT QUIET)
	if (ROOT_FOUND)
		add_subdirectory("examples/hep_dimuon_analysis_cms")
	endif()

	# CUDA examples
	include(CheckLanguage)
	check_language(CUDA)
	if (CMAKE_CUDA_COMPILER)
		enable_language(CUDA)
		add_subdirectory("examples/cuda/nbody")
	elseif()
		message(WARNING "Could not find CUDA. Try setting CMAKE_CUDA_COMPILER. CUDA examples are disabled.")
	endif()
endif()

# install
include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

set(_llama_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/llama")

configure_package_config_file (
	"${PROJECT_SOURCE_DIR}/cmake/llama-config.cmake.in"
	"${PROJECT_BINARY_DIR}/cmake/llama-config.cmake"
	INSTALL_DESTINATION "${_llama_INSTALL_CMAKEDIR}")

configure_file (
	"${PROJECT_SOURCE_DIR}/cmake/llama-config-version.cmake.in"
	"${PROJECT_BINARY_DIR}/cmake/llama-config-version.cmake"
	@ONLY
)

install( DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/include/llama" DESTINATION "include" )
install(
	FILES
		"${PROJECT_BINARY_DIR}/cmake/llama-config.cmake"
		"${PROJECT_BINARY_DIR}/cmake/llama-config-version.cmake"
	DESTINATION
		"${_llama_INSTALL_CMAKEDIR}"
)
