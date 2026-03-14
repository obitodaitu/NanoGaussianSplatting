// Copyright Epic Games, Inc. All Rights Reserved.

#include "GaussianSplatShaders.h"
#include "ShaderParameterUtils.h"
#include "RHIStaticStates.h"

// Implement global shaders
IMPLEMENT_GLOBAL_SHADER(FGaussianSplatCalcViewDataCS, "/Plugin/GaussianSplatting/Private/CalcViewData.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FGaussianSplatCalcDistancesCS, "/Plugin/GaussianSplatting/Private/CalcDistances.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FGaussianSplatVS, "/Plugin/GaussianSplatting/Private/GaussianSplatRendering.usf", "MainVS", SF_Vertex);
IMPLEMENT_GLOBAL_SHADER(FGaussianSplatPS, "/Plugin/GaussianSplatting/Private/GaussianSplatRendering.usf", "MainPS", SF_Pixel);
IMPLEMENT_GLOBAL_SHADER(FRadixSortCountCS, "/Plugin/GaussianSplatting/Private/RadixSort.usf", "CountCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FRadixSortPrefixSumCS, "/Plugin/GaussianSplatting/Private/RadixSort.usf", "PrefixSumCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FRadixSortDigitPrefixSumCS, "/Plugin/GaussianSplatting/Private/RadixSort.usf", "DigitPrefixSumCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FRadixSortScatterCS, "/Plugin/GaussianSplatting/Private/RadixSort.usf", "ScatterCS", SF_Compute);

// Cluster culling shaders
IMPLEMENT_GLOBAL_SHADER(FClusterCullingResetCS, "/Plugin/GaussianSplatting/Private/ClusterCulling.usf", "ResetCounterCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FClusterCullingCS, "/Plugin/GaussianSplatting/Private/ClusterCulling.usf", "MainCS", SF_Compute);

// Global cluster culling
IMPLEMENT_GLOBAL_SHADER(FGlobalClusterCullingResetCS, "/Plugin/GaussianSplatting/Private/GlobalClusterCulling.usf", "ResetCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FGlobalClusterCullingCS, "/Plugin/GaussianSplatting/Private/GlobalClusterCulling.usf", "MainCS", SF_Compute);

// Global compact splats
IMPLEMENT_GLOBAL_SHADER(FGlobalCompactSplatsResetCS, "/Plugin/GaussianSplatting/Private/GlobalCompactSplats.usf", "ResetCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FGlobalCompactSplatsCS, "/Plugin/GaussianSplatting/Private/GlobalCompactSplats.usf", "MainCS", SF_Compute);

// Repack + Global CalcViewData
IMPLEMENT_GLOBAL_SHADER(FRepackCompactedIndicesCS, "/Plugin/GaussianSplatting/Private/RepackCompactedIndices.usf", "MainCS", SF_Compute);
IMPLEMENT_GLOBAL_SHADER(FGlobalCalcViewDataCS, "/Plugin/GaussianSplatting/Private/GlobalCalcViewData.usf", "MainCS", SF_Compute);

// NOTE: GPU-driven LOD shaders removed in unified approach
// LOD splats are now processed by CalcViewData.usf using the same buffers as original splats

// Gather visible counts global
IMPLEMENT_GLOBAL_SHADER(FGatherVisibleCountsGlobalCS, "/Plugin/GaussianSplatting/Private/GatherVisibleCountsGlobal.usf", "MainCS", SF_Compute);

// Compaction prefix-sum shaders
IMPLEMENT_GLOBAL_SHADER(FPrefixSumVisibleCountsCS,  "/Plugin/GaussianSplatting/Private/GlobalAccumulatorPrefixSum.usf", "MainCS", SF_Compute);
