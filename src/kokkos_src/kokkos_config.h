#ifndef __TYPEDEFS_FOR_KOKKOS__
#define __TYPEDEFS_FOR_KOKKOS__
/* Copyright for a lot of the type defs below
// It comes from the kokkos examples
// but we have lightly modified it to work with nice small matrices
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions Contact  H. Carter Edwards (hcedwar@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
#include <Kokkos_Core.hpp>


// typedef Kokkos::View<double*>   ViewVectorType;
// typedef Kokkos::View<double**>  ViewMatrixType;

#ifdef USE_GPU
// GPU options
typedef Kokkos::Cuda          ExecSpace;
typedef Kokkos::CudaUVMSpace  MemSpace_CB;
typedef Kokkos::LayoutLeft    Layout;
#else
// CPU options
typedef Kokkos::OpenMP   ExecSpace;
typedef Kokkos::OpenMP   MemSpace_CB;
typedef Kokkos::LayoutRight  Layout; // faster for CPUs
#endif

// All options
// typedef Kokkos::Serial   ExecSpace;
// typedef Kokkos::Threads  ExecSpace;
// typedef Kokkos::OpenMP   ExecSpace;
// typedef Kokkos::Cuda     ExecSpace;

// typedef Kokkos::HostSpace     MemSpace;
// typedef Kokkos::OpenMP        MemSpace;
// typedef Kokkos::CudaSpace     MemSpace;
// typedef Kokkos::CudaUVMSpace  MemSpace;

// typedef Kokkos::LayoutLeft   Layout;
// typedef Kokkos::LayoutRight  Layout; // faster for CPUs

typedef Kokkos::RangePolicy<ExecSpace>  range_policy;

// Allocate y, x vectors and Matrix A on device.
// typedef Kokkos::View<double*, Layout, MemSpace>   ViewVectorType;
// typedef Kokkos::View<double**, Layout, MemSpace>  ViewMatrixType;

// typedef Kokkos::View<double**, Kokkos::LayoutLeft,  MemSpace> ViewMatrixTypeL;
// typedef Kokkos::View<double**, Kokkos::LayoutRight, MemSpace> ViewMatrixTypeR;

// typedef Kokkos::View<double****, Kokkos::LayoutLeft,  MemSpace> ViewMatrixBatch;

#endif