// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "RHI.h"
#include "RHIResources.h"
#include "RHIGPUReadback.h"

class FGaussianSplatSceneProxy;
class FSceneView;
struct FGaussianGlobalAccumulator;

/**
 * Work item produced by BuildVisibleClusterWorkList: one per visible cluster.
 * Stored in VisibleClusterWorkList as uint4 (16 bytes).
 */
struct FVisibleClusterWorkItem
{
	uint32 GlobalClusterIndex;   // Index into GlobalClusterBuffer
	uint32 MetadataIndex;        // Proxy index into ProxyMetadataBuffer
	uint32 GlobalSplatStart;     // First splat in GlobalPackedSplatBuffer
	uint32 SplatCount;           // Number of splats to process
};

static_assert(sizeof(FVisibleClusterWorkItem) == 16, "FVisibleClusterWorkItem must be 16 bytes (uint4)");

/**
 * Per-proxy GPU metadata uploaded to ProxyMetadataBuffer each frame.
 * Must be exactly 128 bytes to match the HLSL struct in global shaders (Stages 2-5).
 *
 * Layout (row-major, float32):
 *   [0..63]   float LocalToWorld[16]           4x4 transform matrix
 *   [64..127] uint32 fields x 16               offsets, counts, cumulative totals
 */
struct FProxyGPUMetadata
{
	// Transform: row-major float4x4 (64 bytes)
	float LocalToWorld[16];

	// Splat buffer offsets
	uint32 GlobalSplatStartIndex;        // Byte offset / PackedSplatStride into GlobalPackedSplatBuffer
	uint32 TotalSplatCount;              // Total splats for this proxy (original + LOD)
	uint32 OriginalSplatCount;           // Original splats only (excludes LOD splats)

	// Cluster buffer offsets
	uint32 GlobalClusterStartIndex;      // Element offset into GlobalClusterBuffer
	uint32 ClusterCount;                 // Total clusters for this proxy
	uint32 LeafClusterCount;             // Leaf clusters only (used by culling dispatch size)

	// Dynamic bitmap offsets (uint32 units) — populated in Stage 2
	uint32 GlobalVisibilityBitmapStart;  // uint32 offset into GlobalClusterVisibilityBitmap
	uint32 GlobalSelectedClusterStart;   // Element offset into GlobalSelectedClusterBuffer
	uint32 GlobalLODBitmapStart;         // uint32 offset into GlobalLODClusterSelectedBitmap

	// Compaction
	uint32 GlobalSplatClusterIndexStart; // Element offset into GlobalSplatClusterIndexBuffer
	uint32 CompactionCounterIndex;       // Index into GlobalVisibleCountArray (equals proxy index)
	uint32 CompactionOutputStart;        // Element offset into GlobalCompactedSplatIndices (worst-case slot)

	// SH
	uint32 GlobalSHByteOffset;           // Byte offset into GlobalSHBuffer for this proxy's SH data

	// Cumulative totals for binary search in global shaders
	uint32 GlobalLeafClusterEnd;         // Exclusive end of this proxy's leaf clusters in global space
	uint32 GlobalSplatEnd;               // Exclusive end of this proxy's splats in global space

	// Per-proxy LOD parameters (must match per-proxy culling for Stage 3 validation)
	float ErrorThreshold;                // LOD error threshold from proxy
};

static_assert(sizeof(FProxyGPUMetadata) == 128, "FProxyGPUMetadata must be 128 bytes — HLSL struct must match");

/**
 * Manages global GPU buffers that concatenate data across all registered proxies.
 *
 * Stage 1: CPU offset table, ProxyMetadataBuffer, and static buffer
 *   concatenation (GlobalPackedSplatBuffer, GlobalClusterBuffer,
 *   GlobalSplatClusterIndexBuffer). No rendering changes.
 *
 * Stage 2: Global cluster culling using shadow bitmaps.
 *   DispatchGlobalClusterCulling runs and writes to shadow buffers.
 *
 * Owned by FGaussianSplattingModule (render-thread lifetime), like FGaussianGlobalAccumulator.
 */
struct FGlobalSplatBufferManager
{
	//------------------------------------------------------------------------
	// Global static GPU buffers (rebuilt when the registered proxy set changes)
	//------------------------------------------------------------------------

	/** All proxy packed splat data concatenated. ByteAddressBuffer, 16 bytes/splat. */
	FBufferRHIRef GlobalPackedSplatBuffer;
	FShaderResourceViewRHIRef GlobalPackedSplatBufferSRV;

	/** All proxy cluster data concatenated. StructuredBuffer<FGaussianGPUCluster>, 80 bytes/cluster. */
	FBufferRHIRef GlobalClusterBuffer;
	FShaderResourceViewRHIRef GlobalClusterBufferSRV;

	/** All proxy splat-to-cluster index data concatenated. StructuredBuffer<uint>, 4 bytes/entry. */
	FBufferRHIRef GlobalSplatClusterIndexBuffer;
	FShaderResourceViewRHIRef GlobalSplatClusterIndexBufferSRV;

	//------------------------------------------------------------------------
	// Per-frame dynamic metadata (rebuilt every frame for updated transforms)
	//------------------------------------------------------------------------

	/** One FProxyGPUMetadata per proxy. StructuredBuffer, 128 bytes/entry. */
	FBufferRHIRef ProxyMetadataBuffer;
	FShaderResourceViewRHIRef ProxyMetadataBufferSRV;

	//------------------------------------------------------------------------
	// Stage 2: Shadow bitmaps (global cluster culling)
	//------------------------------------------------------------------------

	/** Shadow visibility bitmap — 1 bit per cluster, concatenated across proxies. */
	FBufferRHIRef ShadowClusterVisibilityBitmap;
	FUnorderedAccessViewRHIRef ShadowClusterVisibilityBitmapUAV;

	/** Shadow selected cluster buffer — one entry per leaf cluster across all proxies. */
	FBufferRHIRef ShadowSelectedClusterBuffer;
	FUnorderedAccessViewRHIRef ShadowSelectedClusterBufferUAV;

	/** Shadow LOD cluster selected bitmap. */
	FBufferRHIRef ShadowLODClusterSelectedBitmap;
	FUnorderedAccessViewRHIRef ShadowLODClusterSelectedBitmapUAV;

	/** Shadow visible cluster count (single uint32). */
	FBufferRHIRef ShadowVisibleClusterCountBuffer;
	FUnorderedAccessViewRHIRef ShadowVisibleClusterCountBufferUAV;

	/** Shadow LOD cluster list. */
	FBufferRHIRef ShadowLODClusterBuffer;
	FUnorderedAccessViewRHIRef ShadowLODClusterBufferUAV;

	/** Shadow LOD cluster count (single uint32). */
	FBufferRHIRef ShadowLODClusterCountBuffer;
	FUnorderedAccessViewRHIRef ShadowLODClusterCountBufferUAV;

	/** Shadow LOD splat total (single uint32). */
	FBufferRHIRef ShadowLODSplatTotalBuffer;
	FUnorderedAccessViewRHIRef ShadowLODSplatTotalBufferUAV;

	bool bShadowBuffersAllocated = false;

	//------------------------------------------------------------------------
	// Stage 3: Shadow compaction buffers (global compact splats)
	//------------------------------------------------------------------------

	/** Shadow compacted splat indices — worst case TotalSplatCount entries. */
	FBufferRHIRef ShadowCompactedSplatIndices;
	FUnorderedAccessViewRHIRef ShadowCompactedSplatIndicesUAV;

	/** Shadow visible count array — one uint32 per proxy. */
	FBufferRHIRef ShadowVisibleCountArray;
	FUnorderedAccessViewRHIRef ShadowVisibleCountArrayUAV;
	FShaderResourceViewRHIRef ShadowVisibleCountArraySRV;

	bool bShadowCompactBuffersAllocated = false;

	//------------------------------------------------------------------------
	// Stage 4: Shadow ViewData buffers (Repack + GlobalCalcViewData)
	//------------------------------------------------------------------------

	/** Shadow repacked splat indices — dense contiguous layout (TotalSplatCount worst case). */
	FBufferRHIRef ShadowRepackedSplatIndices;
	FUnorderedAccessViewRHIRef ShadowRepackedSplatIndicesUAV;
	FShaderResourceViewRHIRef ShadowRepackedSplatIndicesSRV;

	/** Per-proxy render parameters (OpacityScale, SplatScale, SH settings). */
	FBufferRHIRef ProxyRenderParamsBuffer;
	FShaderResourceViewRHIRef ProxyRenderParamsBufferSRV;
	uint32 AllocatedRenderParamsCount = 0;

	/** Mapping from processed-proxy index to metadata index (uploaded per frame). */
	FBufferRHIRef ProcessedToMetadataIndexBuffer;
	FShaderResourceViewRHIRef ProcessedToMetadataIndexBufferSRV;
	uint32 AllocatedMappingCount = 0;

	bool bShadowViewDataBuffersAllocated = false;

	//------------------------------------------------------------------------
	// Cluster-based CalcViewData buffers (new pipeline)
	//------------------------------------------------------------------------

	/** Work item: one per visible cluster. uint4 (16 bytes) each. */
	FBufferRHIRef VisibleClusterWorkList;
	FUnorderedAccessViewRHIRef VisibleClusterWorkListUAV;
	FShaderResourceViewRHIRef VisibleClusterWorkListSRV;

	/** Atomic counter: number of visible clusters appended. Single uint32. */
	FBufferRHIRef VisibleClusterCountBuffer;
	FUnorderedAccessViewRHIRef VisibleClusterCountBufferUAV;
	FShaderResourceViewRHIRef VisibleClusterCountBufferSRV;

	/** Per visible cluster splat count (parallel to work list). */
	FBufferRHIRef ClusterSplatCountsBuffer;
	FUnorderedAccessViewRHIRef ClusterSplatCountsBufferUAV;
	FShaderResourceViewRHIRef ClusterSplatCountsBufferSRV;

	/** Prefix sum of ClusterSplatCounts. Size: TotalClusterCount + 1. */
	FBufferRHIRef ClusterOutputOffsetsBuffer;
	FUnorderedAccessViewRHIRef ClusterOutputOffsetsBufferUAV;
	FShaderResourceViewRHIRef ClusterOutputOffsetsBufferSRV;

	/** Indirect dispatch args for ClusterCalcViewData. 3 uint32s. */
	FBufferRHIRef ClusterCalcViewDataIndirectArgs;
	FUnorderedAccessViewRHIRef ClusterCalcViewDataIndirectArgsUAV;

	/** Test indirect args buffers (Stage 2 validation only, not used by real pipeline). */
	FBufferRHIRef TestCalcDistIndirectArgs;
	FUnorderedAccessViewRHIRef TestCalcDistIndirectArgsUAV;
	FBufferRHIRef TestSortIndirectArgs;
	FUnorderedAccessViewRHIRef TestSortIndirectArgsUAV;
	FBufferRHIRef TestSortParams;
	FUnorderedAccessViewRHIRef TestSortParamsUAV;
	FBufferRHIRef TestDrawIndirectArgs;
	FUnorderedAccessViewRHIRef TestDrawIndirectArgsUAV;

	/** Shadow ViewData buffer for cluster CalcViewData test mode. */
	FBufferRHIRef ShadowClusterViewDataBuffer;
	FUnorderedAccessViewRHIRef ShadowClusterViewDataBufferUAV;

	bool bClusterWorkListBuffersAllocated = false;

	/** GPU readback objects for cluster work list test (multi-frame). */
	TUniquePtr<FRHIGPUBufferReadback> ClusterTest_CountReadback;
	TUniquePtr<FRHIGPUBufferReadback> ClusterTest_SplatCountsReadback;
	TUniquePtr<FRHIGPUBufferReadback> ClusterTest_OldVisibleReadback;
	TUniquePtr<FRHIGPUBufferReadback> ClusterTest_WorkItemsReadback;
	uint32 ClusterTest_MaxReadbackClusters = 0;
	uint32 ClusterTest_ProxyCount = 0;
	bool bClusterTestWaitingForReadback = false;

	/** GPU readback objects for cluster prefix sum test (multi-frame). */
	TUniquePtr<FRHIGPUBufferReadback> PrefixTest_NewTotalReadback;
	TUniquePtr<FRHIGPUBufferReadback> PrefixTest_OldDrawArgsReadback;
	TUniquePtr<FRHIGPUBufferReadback> PrefixTest_NewDrawArgsReadback;
	bool bPrefixTestWaitingForReadback = false;

	/** GPU readback objects for cluster CalcViewData test (multi-frame). */
	TUniquePtr<FRHIGPUBufferReadback> CalcViewTest_NewReadback;
	TUniquePtr<FRHIGPUBufferReadback> CalcViewTest_OldReadback;
	TUniquePtr<FRHIGPUBufferReadback> CalcViewTest_CountReadback;
	uint32 CalcViewTest_TotalVisible = 0;
	bool bCalcViewTestWaitingForReadback = false;

	//------------------------------------------------------------------------
	// Global SH buffer (static, concatenated raw bytes from all proxies)
	//------------------------------------------------------------------------

	FBufferRHIRef GlobalSHBuffer;
	FShaderResourceViewRHIRef GlobalSHBufferSRV;
	uint32 TotalSHBytes = 0;

	//------------------------------------------------------------------------
	// CPU-side state
	//------------------------------------------------------------------------

	/** Last known valid proxy set — used to detect changes requiring static buffer rebuild. */
	TArray<FGaussianSplatSceneProxy*> LastProxySet;

	uint32 TotalSplatCount = 0;
	uint32 TotalClusterCount = 0;
	uint32 TotalLeafClusterCount = 0;

	/** Total uint32s needed for all per-proxy visibility bitmaps combined (for Stage 2). */
	uint32 TotalVisibilityBitmapUints = 0;

	/** Allocated capacity of ProxyMetadataBuffer (in proxy count). Recreated when proxy count grows. */
	uint32 AllocatedMetadataCount = 0;

	bool bStaticBuffersBuilt = false;

	//------------------------------------------------------------------------
	// Interface
	//------------------------------------------------------------------------

	/**
	 * Call once per frame from the render loop.
	 * - Filters AllRegisteredProxies to valid Nanite proxies.
	 * - Rebuilds static GPU buffers if the proxy set changed.
	 * - Uploads ProxyMetadataBuffer (transforms + offsets) every frame.
	 *
	 * Returns true if static buffers were rebuilt this frame.
	 */
	bool UpdateIfNeeded(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& AllRegisteredProxies);

	/** Release all GPU buffers. Must be called from the render thread before destruction. */
	void Release();

	//------------------------------------------------------------------------
	// Stage 2: Global Cluster Culling
	//------------------------------------------------------------------------

	/**
	 * Allocate shadow bitmaps and counters for global culling.
	 * Called lazily on first dispatch. Resized when totals change.
	 */
	void EnsureShadowBuffers(FRHICommandListImmediate& RHICmdList);

	/**
	 * Dispatch the global cluster culling shader.
	 * Writes results to shadow culling buffers.
	 */
	void DispatchGlobalClusterCulling(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies);

	//------------------------------------------------------------------------
	// Stage 3: Global Compact Splats
	//------------------------------------------------------------------------

	/**
	 * Allocate shadow compaction buffers.
	 * Called lazily on first dispatch. Resized when totals change.
	 */
	void EnsureShadowCompactBuffers(FRHICommandListImmediate& RHICmdList);

	/**
	 * Dispatch the global compact splats shader.
	 * Uses shadow bitmaps from Stage 2 (DispatchGlobalClusterCulling must run first).
	 */
	void DispatchGlobalCompactSplats(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies);

	//------------------------------------------------------------------------
	// Stage 4: Repack + Global CalcViewData
	//------------------------------------------------------------------------

	/**
	 * Allocate shadow ViewData buffers for Stage 4.
	 * Called lazily on first dispatch. Resized when totals change.
	 */
	void EnsureShadowViewDataBuffers(FRHICommandListImmediate& RHICmdList);

	/**
	 * Upload per-proxy render parameters (OpacityScale, SplatScale, SH settings).
	 * Called each frame alongside UploadProxyMetadata.
	 */
	void UploadProxyRenderParams(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies);

	/**
	 * Dispatch Stage 4: Repack + GlobalCalcViewData.
	 * Called AFTER PrefixSum so GlobalBaseOffsetsBuffer is available.
	 * Uses ShadowCompactedSplatIndices from Stage 3 as input.
	 * Reuses GlobalAccumulator's prefix sum results (validated in Stage 3).
	 */
	void DispatchRepackAndGlobalCalcViewData(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		uint32 MaxRenderBudget);

	//------------------------------------------------------------------------
	// Stage 5: Gather Visible Counts Global (reorder shadow → processed order)
	//------------------------------------------------------------------------

	/**
	 * Upload ProcessedToMetadataIndexBuffer and dispatch the GatherVisibleCountsGlobal
	 * shader to reorder ShadowVisibleCountArray (metadata order) into
	 * GlobalVisibleCountArray (processed order) so PrefixSum works unchanged.
	 *
	 * Must be called AFTER DispatchGlobalCompactSplats (Stage 3) and BEFORE PrefixSum.
	 */
	void DispatchGatherVisibleCountsGlobal(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies,
		FGaussianGlobalAccumulator* GlobalAccumulator);

	//------------------------------------------------------------------------
	// Cluster-based CalcViewData: Build visible cluster work list
	//------------------------------------------------------------------------

	/**
	 * Allocate cluster work list buffers.
	 * Called lazily on first dispatch. Resized when totals change.
	 */
	void EnsureClusterWorkListBuffers(FRHICommandListImmediate& RHICmdList);

	/**
	 * Dispatch BuildVisibleClusterWorkList shaders (Reset + Leaf + LOD).
	 * Runs AFTER DispatchGlobalClusterCulling, IN PARALLEL with old compact path.
	 * Produces VisibleClusterWorkList + VisibleClusterCount + ClusterSplatCounts.
	 */
	void DispatchBuildVisibleClusterWorkList(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies);

	/**
	 * Dispatch ClusterCalcViewData shader.
	 * One thread group per visible cluster, groupshared proxy data.
	 * Writes to ShadowClusterViewDataBuffer (test mode) or GlobalViewDataBuffer.
	 */
	void DispatchClusterCalcViewData(
		FRHICommandListImmediate& RHICmdList,
		const FSceneView& View,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		uint32 MaxRenderBudget);

	/**
	 * Dispatch ClusterPrefixSum shader.
	 * Computes prefix sum of ClusterSplatCounts → ClusterOutputOffsets.
	 * Writes indirect args for ClusterCalcViewData.
	 * Runs IN PARALLEL with old PrefixSum (test buffers only).
	 */
	void DispatchClusterPrefixSum(
		FRHICommandListImmediate& RHICmdList,
		FGaussianGlobalAccumulator* GlobalAccumulator,
		uint32 MaxRenderBudget);

	bool IsReady() const { return bStaticBuffersBuilt && ProxyMetadataBuffer.IsValid(); }
	uint32 GetProxyCount() const { return (uint32)LastProxySet.Num(); }

private:
	/** GPU-to-GPU copy of per-proxy static buffers into global concatenated buffers. */
	void RebuildStaticBuffers(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies);

	/** Build FProxyGPUMetadata array and upload to ProxyMetadataBuffer. */
	void UploadProxyMetadata(
		FRHICommandListImmediate& RHICmdList,
		const TArray<FGaussianSplatSceneProxy*>& ValidProxies,
		bool bLogDetails);
};
