const NUM_BINS: u32 = 8;
const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;

var<workgroup> giNumIntermediateNodes: atomic<u32>;

struct IntermediateNode
{
    mCentroid: vec4<f32>,
    mMinBounds: vec4<f32>,
    mMaxBounds: vec4<f32>,

    miLeftNodeIndex: u32,
    miRightNodeIndex: u32,
    miLeafNode: u32,
    miMortonCode: u32,
};

struct ShortestDistanceResult
{
    mCentroid: vec4<f32>,
    mShortestDistanceCentroid: vec4<f32>,
    mfShortestDistance: f32,
    miShortestDistanceNode: u32,
};

struct MergeInfo
{
    miNumTotalNodes: u32,
    miPadding0: u32,
    miPadding1: u32,
    miPadding2: u32,
};

struct BVHProcessInfo
{
    miStep: u32,
    miStartNodeIndex: u32,
    miEndNodeIndex: u32, 
    miNumMeshes: u32,
};

@group(0) @binding(0)
var<storage, read_write> aTempIntermediateNodes: array<IntermediateNode>;

@group(0) @binding(1)
var<storage, read> aIntermediateNodes: array<IntermediateNode>;

@group(0) @binding(2)
var<storage, read> aShortestDistanceResults: array<ShortestDistanceResult>;

@group(0) @binding(3)
var<storage, read> bvhProcessInfo: BVHProcessInfo;

const iNumLocalThreads: u32 = 256u;

//////
@compute
@workgroup_size(iNumLocalThreads)
fn cs_main(
    @builtin(local_invocation_id) localIndex: vec3<u32>,
    @builtin(workgroup_id) workGroupIndex: vec3<u32>,
    @builtin(num_workgroups) numWorkGroups: vec3<u32>) 
{
    if(bvhProcessInfo.miEndNodeIndex - bvhProcessInfo.miStartNodeIndex <= bvhProcessInfo.miNumMeshes)
    {
        return;
    }

    let iLocalThreadID: u32 = localIndex.x;
    let iTotalThreadID: u32 = localIndex.x + workGroupIndex.x * iNumLocalThreads;
    let iNumTotalThreads: u32 = iNumLocalThreads * numWorkGroups.x;

    let iNodeIndex: u32 = iTotalThreadID + bvhProcessInfo.miStartNodeIndex;
    let iShortestDistanceNodeIndex: u32 = aShortestDistanceResults[iTotalThreadID].miShortestDistanceNode + bvhProcessInfo.miStartNodeIndex;

    // merge the boundaries and compute node centroid
    var iNextLevelNodeIndex: u32 = iTotalThreadID;
    aTempIntermediateNodes[iNextLevelNodeIndex].mMinBounds = getMinBounds(
        aIntermediateNodes[iNodeIndex], 
        aIntermediateNodes[iShortestDistanceNodeIndex]);
    aTempIntermediateNodes[iNextLevelNodeIndex].mMaxBounds = getMaxBounds(
        aIntermediateNodes[iNodeIndex], 
        aIntermediateNodes[iShortestDistanceNodeIndex]);
    aTempIntermediateNodes[iNextLevelNodeIndex].miLeftNodeIndex = iNodeIndex;
    aTempIntermediateNodes[iNextLevelNodeIndex].miRightNodeIndex = iShortestDistanceNodeIndex;

    aTempIntermediateNodes[iNextLevelNodeIndex].mCentroid = (
        aTempIntermediateNodes[iNextLevelNodeIndex].mMinBounds +
        aTempIntermediateNodes[iNextLevelNodeIndex].mMaxBounds
    ) * 0.5f;

    aTempIntermediateNodes[iNextLevelNodeIndex].mCentroid.w = aIntermediateNodes[iTotalThreadID + bvhProcessInfo.miStartNodeIndex].mCentroid.w;
    // aTempIntermediateNodes[iNextLevelNodeIndex].mCentroid.w = f32(iTotalThreadID);

    aTempIntermediateNodes[iNextLevelNodeIndex].miMortonCode = 
        m3D_e_magicbits(aTempIntermediateNodes[iNextLevelNodeIndex].mCentroid.xyz);
    
}

/////
fn getMinBounds(
    node0: IntermediateNode, 
    node1: IntermediateNode) -> vec4<f32>
{
    var ret: vec4<f32> = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, 0.0f);
    ret.x = min(node0.mMinBounds.x, node1.mMinBounds.x);
    ret.y = min(node0.mMinBounds.y, node1.mMinBounds.y);
    ret.z = min(node0.mMinBounds.z, node1.mMinBounds.z);

    return ret;
}

/////
fn getMaxBounds(
    node0: IntermediateNode, 
    node1: IntermediateNode) -> vec4<f32>
{
    var ret: vec4<f32> = vec4<f32>(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0f);
    ret.x = max(node0.mMaxBounds.x, node1.mMaxBounds.x);
    ret.y = max(node0.mMaxBounds.y, node1.mMaxBounds.y);
    ret.z = max(node0.mMaxBounds.z, node1.mMaxBounds.z);

    return ret;
};

// from https://github.com/Forceflow/libmorton/blob/main/include/libmorton/morton3D.h
//uint32_t magicbit3D_masks32_encode[6] = { 0x000003ff, 0, 0x30000ff, 0x0300f00f, 0x30c30c3, 0x9249249 }; // we add a 0 on position 1 in this array to use same code for 32-bit and 64-bit cases
//uint64_t magicbit3D_masks64_encode[6] = { 0x1fffff, 0x1f00000000ffff, 0x1f0000ff0000ff, 0x100f00f00f00f00f, 0x10c30c30c30c30c3, 0x1249249249249249 };

/////
fn morton3D_SplitBy3bits(
    a: u32) -> u32
{
    var masks: array<u32, 6> = array<u32, 6>(
        0x000003ff, 
        0, 
        0x30000ff, 
        0x0300f00f, 
        0x30c30c3, 
        0x9249249
    );

    var x: u32 = u32(a) & masks[0];
    x = (x | x << 16) & masks[2];
    x = (x | x << 8) & masks[3];
    x = (x | x << 4) & masks[4];
    x = (x | x << 2) & masks[5];
    return x;
};

/////
fn m3D_e_magicbits(
    v: vec3<f32>) -> u32
{
    var vInt: vec3<u32> = vec3<u32>(
        u32(v.x * 100.0f + 100.0f),
        u32(v.y * 100.0f + 100.0f),
        u32(v.z * 100.0f + 100.0f)
    );
    return morton3D_SplitBy3bits(vInt.x) | (morton3D_SplitBy3bits(vInt.y) << 1) | (morton3D_SplitBy3bits(vInt.z) << 2);
};