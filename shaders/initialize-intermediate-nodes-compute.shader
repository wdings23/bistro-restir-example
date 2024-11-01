const NUM_BINS: u32 = 8;
const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;


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
var<storage, read_write> bvhProcessInfo: BVHProcessInfo;

@group(1) @binding(0)
var<storage, read> aVertexPositions: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read> aTriangles: array<Triangle>;

@group(1) @binding(2)
var<storage, read> aMeshInfo: array<MeshInfo, 128>;

const iNumLocalThreads: u32 = 256u;

@compute
@workgroup_size(iNumLocalThreads)
fn cs_main(
    @builtin(local_invocation_id) localIndex: vec3<u32>,
    @builtin(workgroup_id) workGroupIndex: vec3<u32>,
    @builtin(num_workgroups) numWorkGroups: vec3<u32>) 
{
    let iLocalThreadID: u32 = localIndex.x;
    let iTotalThreadID: u32 = localIndex.x + workGroupIndex.x * iNumLocalThreads;
    let iNumTotalThreads: u32 = iNumLocalThreads * numWorkGroups.x;

    for(var i: u32 = 0; i < 10; i++)
    {
        var iThreadID: u32 = iTotalThreadID + i * iNumLocalThreads * numWorkGroups.x;

        aIntermediateNodes[iThreadID].mCentroid = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        aIntermediateNodes[iThreadID].mMinBounds = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        aIntermediateNodes[iThreadID].mMaxBounds = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

        aIntermediateNodes[iThreadID].miLeftNodeIndex = 0u;
        aIntermediateNodes[iThreadID].miRightNodeIndex = 0u;
        aIntermediateNodes[iThreadID].miLeafNode = 0u;
        aIntermediateNodes[iThreadID].miMortonCode = 0u;
    }

    // intermediate nodes from triangles with leaf node containing the triangle index
    let meshInfo: MeshInfo = aMeshInfo[0];
    for(var i: u32 = iTotalThreadID; i < meshInfo.miNumTriangles; i += iNumTotalThreads)
    {
        if(i >= meshInfo.miNumTriangles)
        {
            break;
        }

        var iTriangleIndex: u32 = i + meshInfo.miNumTriangles;

        let triangle: Triangle = aTriangles[i];
        let pos0: vec4<f32> = aVertexPositions[triangle.miV0];
        let pos1: vec4<f32> = aVertexPositions[triangle.miV1];
        let pos2: vec4<f32> = aVertexPositions[triangle.miV2];

        aIntermediateNodes[i].mCentroid = aTriangles[i].mCentroid;
        aIntermediateNodes[i].mMinBounds = min(min(pos0, pos1), pos2);
        aIntermediateNodes[i].mMaxBounds = max(max(pos0, pos1), pos2);

        // aIntermediateNodes[i].mCentroid.w = f32(iTotalThreadID);

        aIntermediateNodes[i].miLeftNodeIndex = UINT32_MAX;
        aIntermediateNodes[i].miRightNodeIndex = UINT32_MAX;
        aIntermediateNodes[i].miLeafNode = i;
        aIntermediateNodes[i].miMortonCode = m3D_e_magicbits(aIntermediateNodes[i].mCentroid .xyz);

        aLeafNodes[i].miTriangleIndex = i;
    }

    if(workGroupIndex.x == 0 && iLocalThreadID == 0)
    {
        bvhProcessInfo.miStartNodeIndex = 0u;
    }

    bvhProcessInfo.miStep = 0u;
    bvhProcessInfo.miStartNodeIndex = 0u;
    bvhProcessInfo.miEndNodeIndex = meshInfo.miNumTriangles;
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