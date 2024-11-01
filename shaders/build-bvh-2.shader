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

@group(0) @binding(0)
var<storage, read_write> aIntermediateNodes: array<IntermediateNode>;

@group(0) @binding(1)
var<storage, read_write> aLeafNodes: array<LeafNode>;

@group(0) @binding(2)
var<storage, read_write> aTempIntermediateNodes: array<IntermediateNode>;

@group(1) @binding(0)
var<storage, read> aVertexPositions: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read> aTriangles: array<Triangle>;

@group(1) @binding(2)
var<storage, read> aMeshInfo: array<MeshInfo, 128>;

//var<workgroup> aTempIntermediateNodes2: array<IntermediateNode, 4096>;

const iNumThreads: u32 = 256u;

@compute
@workgroup_size(iNumThreads)
fn cs_main(
    @builtin(global_invocation_id) index: vec3<u32>,
    @builtin(workgroup_id) workGroupIndex: vec3<u32>) 
{
    let iThreadID: u32 = u32(index.x) + workGroupIndex.x * iNumThreads;

    atomicStore(&giNumIntermediateNodes, 0u);
    atomicStore(&giNumPrunedIntermediateNodes, 0u);

    workgroupBarrier();
    
    var iTotalNumTriangles: u32 = 0;
    for(var iMesh: u32 = 0; iMesh < 1; iMesh++)
    {
        workgroupBarrier();

        // build intermediate nodes with triangles
        let meshInfo: MeshInfo = aMeshInfo[iMesh];
        for(var i: u32 = iThreadID; i < meshInfo.miNumTriangles; i += iNumThreads)
        {
            var iTriangleIndex: u32 = i + iTotalNumTriangles;

            let triangle: Triangle = aTriangles[i];
            let pos0: vec4<f32> = aVertexPositions[triangle.miV0];
            let pos1: vec4<f32> = aVertexPositions[triangle.miV1];
            let pos2: vec4<f32> = aVertexPositions[triangle.miV2];

            aIntermediateNodes[i].mCentroid = aTriangles[i].mCentroid;
            aIntermediateNodes[i].mMinBounds = min(min(pos0, pos1), pos2);
            aIntermediateNodes[i].mMaxBounds = max(max(pos0, pos1), pos2);

            aIntermediateNodes[i].miLeftNodeIndex = UINT32_MAX;
            aIntermediateNodes[i].miRightNodeIndex = UINT32_MAX;
            aIntermediateNodes[i].miLeafNode = i;
            aIntermediateNodes[i].miMortonCode = m3D_e_magicbits(aIntermediateNodes[i].mCentroid .xyz);

            aLeafNodes[i].miTriangleIndex = i;
        }

        workgroupBarrier();

        // build bvh
        buildBVH2(
            iThreadID,
            iNumThreads,
            meshInfo.miVertexIndexOffsetStart,
            meshInfo.miNumTriangles);

        if(iThreadID == 0)
        {
            iTotalNumTriangles += meshInfo.miNumTriangles;
        }
    }

}

/////
fn buildBVH2(
    iThreadIndex: u32,
    iNumThreads: u32,
    iStartTriangleIndex: u32,
    iNumNodes: u32,
)
{
    // start and end node index
    var iNumIntermediateNodes: u32 = iNumNodes;
    var iNodeIndex: u32 = iThreadIndex;
    var iStartNodeIndex: u32 = 0u;
    var iEndNodeIndex: u32 = iNumNodes;

    var iRootNodeIndex: u32 = UINT32_MAX;

    var bDone: bool = false;
    for(var iStep: u32 = 0; iStep < 1000u; iStep++)
    {
        if(iEndNodeIndex - iStartNodeIndex <= 1u)
        {
            iRootNodeIndex = iEndNodeIndex - 1;
            break;
        }

        // use thread index as starting node index 
        var iNodeIndex: u32 = iThreadIndex + iStartNodeIndex;
        for(;;)
        {
            if(iNodeIndex >= iEndNodeIndex)
            {
                break;
            }

            // shortest distance node index from this current node
            let iShortestDistanceNodeIndex: u32 = findShortestDistance(
                iNodeIndex,
                iStartNodeIndex,
                iEndNodeIndex);

            if(iShortestDistanceNodeIndex == UINT32_MAX)
            {
                bDone = true;
                break;
            }

            // merge the boundaries and compute node centroid
            let iNextLevelNodeIndex: u32 = atomicAdd(&giNumIntermediateNodes, 1u);
            aTempIntermediateNodes[iNextLevelNodeIndex].mMinBounds = getMinBounds(iNodeIndex, iShortestDistanceNodeIndex);
            aTempIntermediateNodes[iNextLevelNodeIndex].mMaxBounds = getMaxBounds(iNodeIndex, iShortestDistanceNodeIndex);
            aTempIntermediateNodes[iNextLevelNodeIndex].miLeftNodeIndex = iNodeIndex;
            aTempIntermediateNodes[iNextLevelNodeIndex].miRightNodeIndex = iShortestDistanceNodeIndex;

            aTempIntermediateNodes[iNextLevelNodeIndex].mCentroid = (
                aTempIntermediateNodes[iNextLevelNodeIndex].mMinBounds +
                aTempIntermediateNodes[iNextLevelNodeIndex].mMaxBounds
            ) * 0.5f;

            aTempIntermediateNodes[iNextLevelNodeIndex].miMortonCode = 
                m3D_e_magicbits(aTempIntermediateNodes[iNextLevelNodeIndex].mCentroid.xyz);
        
            // update node index
            iNodeIndex += iNumThreads;
        }

        // barrier
        workgroupBarrier();
        
        // prune next level nodes
        let iNumTempNodes: u32 = atomicLoad(&giNumIntermediateNodes);
        for(var iNodeIndex: u32 = iThreadIndex; iNodeIndex < iNumTempNodes; iNodeIndex += iNumThreads)
        {
            var node: IntermediateNode = aTempIntermediateNodes[iNodeIndex];
            //if(node.miLeftNodeIndex == UINT32_MAX)
            //{
            //    continue;
            //}

            var bInserted: bool = false;
            for(var iCheckNodeIndex: u32 = 0; iCheckNodeIndex < iNumTempNodes; iCheckNodeIndex++)
            {
                if(iNodeIndex == iCheckNodeIndex)
                {
                    continue;
                }

                var checkNode: IntermediateNode = aTempIntermediateNodes[iCheckNodeIndex];
                if((node.miLeftNodeIndex == checkNode.miLeftNodeIndex || node.miLeftNodeIndex == checkNode.miRightNodeIndex) &&
                    (node.miRightNodeIndex == checkNode.miLeftNodeIndex || node.miRightNodeIndex == checkNode.miRightNodeIndex))
                {
                    // add node with smaller left index
                    if(node.miLeftNodeIndex < node.miRightNodeIndex)
                    {
                        let iAddIndex: u32 = atomicAdd(&giNumPrunedIntermediateNodes, 1u);
                        aIntermediateNodes[iEndNodeIndex + iAddIndex] = node;
                    }

                    bInserted = true;
                    break;
                }
            }

            // didn't find a matching node, just add
            if(!bInserted)
            {
                let iAddIndex: u32 = atomicAdd(&giNumPrunedIntermediateNodes, 1u);
                aIntermediateNodes[iEndNodeIndex + iAddIndex] = node;
            }

        }

        if(bDone)
        {
            break;
        }

        // barrier
        workgroupBarrier();

        iStartNodeIndex = iEndNodeIndex;
        iEndNodeIndex += giNumPrunedIntermediateNodes;
        iNumIntermediateNodes += giNumPrunedIntermediateNodes;
        atomicStore(&giNumIntermediateNodes, 0u);

        // barrier
        workgroupBarrier();

        atomicStore(&giNumPrunedIntermediateNodes, 0u);
    }
}

/////
fn findShortestDistance(
    iNodeIndex: u32, 
    iStartNodeIndex: u32, 
    iEndNodeIndex: u32) -> u32
{
    var fShortestDistance: f32 = FLT_MAX;
    var iShortestNodeIndex: u32 = UINT32_MAX;
    let centroid: vec4<f32> = aIntermediateNodes[iNodeIndex].mCentroid;
    for(var i: u32 = iStartNodeIndex; i < iEndNodeIndex; i++)
    {
        if(i == iNodeIndex)
        {
            continue;
        }

        let fDistance: f32 = length(centroid.xyz - aIntermediateNodes[i].mCentroid.xyz);
        let bShorterDistance: bool = fShortestDistance > fDistance;
        fShortestDistance = select(fShortestDistance, fDistance, bShorterDistance);
        iShortestNodeIndex = select(iShortestNodeIndex, i, bShorterDistance);
        //if(fShortestDistance > fDistance)
        //{
        //    fShortestDistance = fDistance;
        //    iShortestNodeIndex = i;
        //}
    }

    return iShortestNodeIndex;
}

/////
fn getMinBounds(
    iNodeIndex0: u32, 
    iNodeIndex1: u32) -> vec4<f32>
{
     let node0: IntermediateNode = aIntermediateNodes[iNodeIndex0];
     let node1: IntermediateNode = aIntermediateNodes[iNodeIndex1];

    var ret: vec4<f32> = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, 0.0f);
    ret.x = min(node0.mMinBounds.x, node1.mMinBounds.x);
    ret.y = min(node0.mMinBounds.y, node1.mMinBounds.y);
    ret.z = min(node0.mMinBounds.z, node1.mMinBounds.z);

    return ret;
}

/////
fn getMaxBounds(
    iNodeIndex0: u32, 
    iNodeIndex1: u32) -> vec4<f32>
{
    let node0: IntermediateNode = aIntermediateNodes[iNodeIndex0];
    let node1: IntermediateNode = aIntermediateNodes[iNodeIndex1];

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
