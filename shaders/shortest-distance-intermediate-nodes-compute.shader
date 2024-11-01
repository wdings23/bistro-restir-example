const NUM_BINS: u32 = 8;
const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;

var<workgroup> giNumIntermediateNodes: atomic<u32>;
var<workgroup> giNumPrunedIntermediateNodes: atomic<u32>;

struct Triangle
{
    miV0 : u32,
    miV1 : u32,
    miV2 : u32,
    mPadding : u32,

    mCentroid : vec4<f32>
};

struct LeafNode
{
    miTriangleIndex: u32,
};

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

struct MeshInfo
{
    miPositionOffsetStart: u32,
    miNumPositions: u32,

    miVertexIndexOffsetStart: u32,
    miNumTriangles: u32
};

struct ShortestDistanceResult
{
    mCentroid: vec4<f32>,
    mShortestDistanceCentroid: vec4<f32>,
    mfShortestDistance: f32,
    miShortestDistanceNode: u32,
};

struct BVHProcessInfo
{
    miStep: u32,
    miStartNodeIndex: u32,
    miEndNodeIndex: u32, 
    miNumMeshes: u32,
};

@group(0) @binding(0)
var<storage, read_write> aIntermediateNodes: array<IntermediateNode>;

@group(0) @binding(1)
var<storage, read_write> aLeafNodes: array<LeafNode>;

@group(0) @binding(2)
var<storage, read_write> aShortestDistanceResults: array<ShortestDistanceResult>;

@group(0) @binding(3)
var<storage, read_write> bvhProcessInfo: BVHProcessInfo;

@group(1) @binding(0)
var<storage, read> aVertexPositions: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read> aTriangles: array<Triangle>;

var<workgroup> aLocalCentroids: array<vec4<f32>, 256>;

const iNumLocalThreads: u32 = 256u;

@compute
@workgroup_size(iNumLocalThreads)
fn cs_main(
    @builtin(local_invocation_id) localIndex: vec3<u32>,
    @builtin(workgroup_id) workGroupIndex: vec3<u32>,
    @builtin(num_workgroups) numWorkGroups: vec3<u32>) 
{
    let iNumNodes: u32 = bvhProcessInfo.miEndNodeIndex - bvhProcessInfo.miStartNodeIndex;
    if(iNumNodes <= bvhProcessInfo.miNumMeshes)
    {
        return;
    }

    let iLocalThreadID: u32 = localIndex.x;
    let iTotalThreadID: u32 = localIndex.x + workGroupIndex.x * iNumLocalThreads;
    let iNumTotalThreads: u32 = iNumLocalThreads * numWorkGroups.x;

    aShortestDistanceResults[iTotalThreadID].mfShortestDistance = FLT_MAX;
    aShortestDistanceResults[iTotalThreadID].miShortestDistanceNode = UINT32_MAX;

    // for debugging
    aShortestDistanceResults[iTotalThreadID].mCentroid = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);     
    aShortestDistanceResults[iTotalThreadID].mShortestDistanceCentroid = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);     

    workgroupBarrier();

    var nodeCentroid: vec4<f32> = aIntermediateNodes[iTotalThreadID + bvhProcessInfo.miStartNodeIndex].mCentroid;

    // load node centroids into LDS, then find the shortest distance of the current centroid to the loaded ones in the LDS
    // repeat until gone through all the intermediate nodes
    var iNumLoops: u32 = u32(ceil(f32(iNumNodes) / f32(iNumLocalThreads)));
    for(var iLoop: u32 = 0; iLoop < iNumLoops; iLoop++)
    {
        workgroupBarrier();
        // load node centroids in to LDS
        var iNodeIndex: u32 = iLoop * iNumLocalThreads + iLocalThreadID + bvhProcessInfo.miStartNodeIndex;
        if(iNodeIndex >= bvhProcessInfo.miEndNodeIndex)
        {
            break;
        }
        aLocalCentroids[iLocalThreadID] = aIntermediateNodes[iNodeIndex].mCentroid;
        workgroupBarrier();

        if(iTotalThreadID >= iNumNodes)
        {
            // for debugging
            aShortestDistanceResults[iTotalThreadID].mShortestDistanceCentroid = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
            aShortestDistanceResults[iTotalThreadID].mfShortestDistance = 0.0f;
            aShortestDistanceResults[iTotalThreadID].miShortestDistanceNode = 0u;

            break;
        }

        // get shortest distance
        let result: ShortestDistanceResult = getShortestDistance2(
            nodeCentroid,
            iTotalThreadID,
            iLoop * iNumLocalThreads);

        aShortestDistanceResults[iTotalThreadID] = result;
        aShortestDistanceResults[iTotalThreadID].mCentroid = nodeCentroid;
    }
}

/////
fn getShortestDistance2(
    nodeCentroid: vec4<f32>,
    iTotalThreadID: u32,
    iStartTotalNodeIndex: u32,
) -> ShortestDistanceResult
{
    var ret: ShortestDistanceResult = aShortestDistanceResults[iTotalThreadID];
    ret.mCentroid = nodeCentroid;
    for(var i: u32 = 0; i < iNumLocalThreads; i++)
    {
        var iCheckNodeIndex: u32 = iStartTotalNodeIndex + i; 
        var fDistance: f32 = length(nodeCentroid - aLocalCentroids[i]);
        

        var bShorterDistance: bool = (ret.mfShortestDistance > fDistance && (iCheckNodeIndex != iTotalThreadID) && (nodeCentroid.w == aLocalCentroids[i].w));
        ret.mfShortestDistance = select(ret.mfShortestDistance, fDistance, bShorterDistance);
        ret.miShortestDistanceNode = select(ret.miShortestDistanceNode, iCheckNodeIndex, bShorterDistance);
        ret.mShortestDistanceCentroid = select(ret.mShortestDistanceCentroid, aLocalCentroids[i], bShorterDistance);
    }

    return ret;
}

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
}

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
}