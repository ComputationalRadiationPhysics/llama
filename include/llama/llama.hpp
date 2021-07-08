// Copyright 2018 Alexander Matthes
// SPDX-License-Identifier: GPL-3.0-or-later

#pragma once

/// \mainpage LLAMA API documentation
///
/// LLAMA is a C++17 template header-only library for the abstraction of memory access patterns. It distinguishes
/// between the view of the algorithm on the memory and the real layout in the background. This enables performance
/// portability for multicore, manycore and gpu applications with the very same code.
///
/// In contrast to many other solutions LLAMA can define nested data structures of arbitrary depths and is not limited
/// only to struct of array and array of struct data layouts. It is also capable to explicitly define padding, blocking,
/// striding and any other run time or compile time access pattern simultaneously.
///
/// To archieve this goal LLAMA is split into mostly independent, orthogonal parts completely written in modern C++17 to
/// run on as many architectures and with as many compilers as possible while still supporting extensions needed e.g. to
/// run on GPU or other many core hardware.
///
/// This page documents the API of LLAMA. The user documentation and an overview about the concepts and ideas can be
/// found here: https://llama-doc.rtfd.io
///
/// LLAMA is licensed under the LGPL3+.

#define LLAMA_VERSION_MAJOR 0
#define LLAMA_VERSION_MINOR 3
#define LLAMA_VERSION_PATCH 0

#ifdef __NVCC__
#    pragma push
#    pragma diag_suppress 940
#endif

#include "ArrayDimsIndexRange.hpp"
#include "BlobAllocators.hpp"
#include "Copy.hpp"
#include "Core.hpp"
#include "Meta.hpp"
#include "Vector.hpp"
#include "View.hpp"
#include "VirtualRecord.hpp"
#include "macros.hpp"
#include "mapping/AoS.hpp"
#include "mapping/AoSoA.hpp"
#include "mapping/Heatmap.hpp"
#include "mapping/OffsetTable.hpp"
#include "mapping/One.hpp"
#include "mapping/SoA.hpp"
#include "mapping/Split.hpp"
#include "mapping/Trace.hpp"
#include "mapping/tree/Mapping.hpp"

#ifdef __NVCC__
#    pragma pop
#endif
