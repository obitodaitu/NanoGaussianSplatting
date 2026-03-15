// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "GaussianDataTypes.h"
#include "GaussianClusterTypes.h"

class UGaussianSplatAsset;

/**
 * Shared render data for a Gaussian Splat asset.
 * Holds CPU-side cached data that is loaded once per asset and shared
 * across all FGaussianSplatGPUResources instances referencing the same asset.
 * This avoids repeated bulk data reads and splat packing when duplicating actors.
 */
class FGaussianSplatRenderData
{
public:
	FGaussianSplatRenderData();
	~FGaussianSplatRenderData();

	/** Initialize from asset data. Only runs once (guarded by bIsInitialized). */
	void Initialize(UGaussianSplatAsset* Asset);

	/** Whether this render data has been initialized */
	bool IsInitialized() const { return bIsInitialized; }

	/** Get asset name for logging */
	const FString& GetAssetName() const { return AssetName; }

public:
	// ---- Cached CPU-side data (loaded once from asset) ----

	/** Packed splat data (16 bytes/splat: float16 pos + octahedral quat + log scale + RGBA) */
	TArray<uint8> PackedSplatData;

	/** Spherical harmonics data */
	TArray<uint8> SHData;

	/** Chunk quantization info */
	TArray<FGaussianChunkInfo> ChunkData;

	/** Cluster hierarchy data (GPU format) */
	TArray<FGaussianGPUCluster> ClusterData;

	/** Splat-to-cluster index mapping */
	TArray<uint32> SplatClusterIndices;

	/** LOD splat-to-cluster index mapping */
	TArray<uint32> LODSplatClusterIndices;

	// ---- Metadata ----

	int32 SplatCount = 0;
	int32 SHBands = 0;
	int32 ClusterCount = 0;
	int32 LeafClusterCount = 0;
	int32 LODSplatCount = 0;
	bool bEnableNanite = false;
	bool bHasClusterData = false;
	bool bHasLODSplats = false;
	EGaussianPositionFormat PositionFormat = EGaussianPositionFormat::Float32;

private:
	bool bIsInitialized = false;
	FString AssetName;
	FCriticalSection InitLock;
};
