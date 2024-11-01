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

struct TruncationInfo
{
    maiNumValidNodes: array<u32, 256>,
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
var<storage, read> aWorkGroupIntermediateNodes: array<IntermediateNode>;

@group(0) @binding(2)
var<storage, read> truncationInfo: TruncationInfo;

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
    let iNumNodes: u32 = bvhProcessInfo.miEndNodeIndex - bvhProcessInfo.miStartNodeIndex;
    if(iNumNodes <= bvhProcessInfo.miNumMeshes)
    {
        return;
    }

    let iLocalThreadID: u32 = localIndex.x;
    let iTotalThreadID: u32 = localIndex.x + workGroupIndex.x * iNumLocalThreads;
    let iNumTotalThreads: u32 = iNumLocalThreads * numWorkGroups.x;

    let iNumWorkgroups = u32(ceil(f32(iNumNodes) / f32(iNumLocalThreads)));

    let iCurrWorkgroup: u32 = workGroupIndex.x;

    var iStartIndex: u32 = bvhProcessInfo.miEndNodeIndex;
    for(var iWorkgroup: u32 = 0; iWorkgroup < iCurrWorkgroup; iWorkgroup++)
    {
        iStartIndex += truncationInfo.maiNumValidNodes[iWorkgroup];
    }

    // for debugging
    aIntermediateNodes[iStartIndex + iLocalThreadID].mCentroid = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    if(iLocalThreadID < truncationInfo.maiNumValidNodes[iCurrWorkgroup])
    {
        aIntermediateNodes[iStartIndex + iLocalThreadID] = aWorkGroupIntermediateNodes[iTotalThreadID]; 
    }
}