const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;

var<workgroup> giNumTruncatedNodes: atomic<u32>;

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
    maiNumMeshBVH: array<u32, 256>
};

struct BVHProcessInfo
{
    miStep: u32,
    miStartNodeIndex: u32,
    miEndNodeIndex: u32, 
    miNumMeshes: u32,
};

struct LocalNodeInfo
{
    mCentroid: vec3<f32>,
    miLeftNodeIndex: u32,
    miRightNodeIndex: u32,
    miMortonCode: u32,
};

var<workgroup> aLocalNodeInfo: array<LocalNodeInfo, 256>;

const iNumLocalThreads: u32 = 256u;

@group(0) @binding(0)
var<storage, read_write> aWorkgroupIntermediateNodes: array<IntermediateNode>;

@group(0) @binding(1)
var<storage, read_write> truncationInfo: TruncationInfo;

@group(0) @binding(2)
var<storage, read> aMergedIntermediateNodes: array<IntermediateNode>;

@group(0) @binding(3)
var<storage, read> bvhProcessInfo: BVHProcessInfo;

//////
@compute
@workgroup_size(iNumLocalThreads)
fn cs_main(
    @builtin(local_invocation_id) localIndex: vec3<u32>,
    @builtin(workgroup_id) workGroupIndex: vec3<u32>,
    @builtin(num_workgroups) numWorkGroups: vec3<u32>) 
{
    let iNumTotalNodes: u32 = bvhProcessInfo.miEndNodeIndex - bvhProcessInfo.miStartNodeIndex;
    if(iNumTotalNodes <= bvhProcessInfo.miNumMeshes)
    {
        return;
    }

    let iLocalThreadID: u32 = localIndex.x;
    let iTotalThreadID: u32 = localIndex.x + workGroupIndex.x * iNumLocalThreads;
    let iNumTotalThreads: u32 = iNumLocalThreads * numWorkGroups.x;

    let iNumLoops: u32 = u32(ceil(f32(iNumTotalNodes) / f32(iNumLocalThreads)));
    
    if(iLocalThreadID == 0)
    {
        atomicStore(&giNumTruncatedNodes, 0u);
        truncationInfo.maiNumValidNodes[workGroupIndex.x] = 0u;
    }

    workgroupBarrier();

    let iStartNodeIndex: u32 = workGroupIndex.x * iNumLocalThreads;
    if(iTotalThreadID >= iNumTotalNodes)
    {
        return;
    }

    // for debugging
    //aWorkgroupIntermediateNodes[iTotalThreadID].mCentroid = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    //aWorkgroupIntermediateNodes[iTotalThreadID].mMinBounds = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    //aWorkgroupIntermediateNodes[iTotalThreadID].mMaxBounds = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

    // check all the nodes
    let node: IntermediateNode = aMergedIntermediateNodes[iTotalThreadID];
    var bInserted: bool = false;
    var aiSameNodes: array<u32, 16>;
    var iNumSameNodes: u32 = 0;
    var iMinSameNodeIndex: u32 = UINT32_MAX;
    var iStartingWorkingGroup: u32 = UINT32_MAX;
    for(var iWorkgroupLoop: u32 = 0; iWorkgroupLoop < iNumLoops; iWorkgroupLoop++)
    {
        //workgroupBarrier();

        // fill out local data store array
        let iThreadNodeIndex: u32 = iLocalThreadID + iWorkgroupLoop * iNumLocalThreads; 
        aLocalNodeInfo[iLocalThreadID].miLeftNodeIndex = aMergedIntermediateNodes[iThreadNodeIndex].miLeftNodeIndex;
        aLocalNodeInfo[iLocalThreadID].miRightNodeIndex = aMergedIntermediateNodes[iThreadNodeIndex].miRightNodeIndex;
        aLocalNodeInfo[iLocalThreadID].miMortonCode = aMergedIntermediateNodes[iThreadNodeIndex].miMortonCode;
        aLocalNodeInfo[iLocalThreadID].mCentroid = aMergedIntermediateNodes[iThreadNodeIndex].mCentroid.xyz;

        workgroupBarrier();

        // check local nodes
        for(var iCheckNodeIndex: u32 = 0; iCheckNodeIndex < iNumLocalThreads; iCheckNodeIndex++)
        {
            let iCheckThreadNodeIndex: u32 = iCheckNodeIndex + iWorkgroupLoop * iNumLocalThreads;
            if(iCheckThreadNodeIndex >= iNumTotalNodes)
            {
                break;
            } 

            // same node
            if(iTotalThreadID == iCheckThreadNodeIndex)
            {
                continue;
            }

            // count node if same centroid or left or right node index is the same 
            var checkLocalNode: LocalNodeInfo = aLocalNodeInfo[iCheckNodeIndex];
            var diff: vec3<f32> = node.mCentroid.xyz - checkLocalNode.mCentroid.xyz;
            var bCond0: bool = ((node.miLeftNodeIndex == checkLocalNode.miLeftNodeIndex || node.miLeftNodeIndex == checkLocalNode.miRightNodeIndex) &&
                (node.miRightNodeIndex == checkLocalNode.miLeftNodeIndex || node.miRightNodeIndex == checkLocalNode.miRightNodeIndex));
            var bCond1: bool = (dot(diff, diff) <= 0.0f); 
            var bTotalCond: bool = (bCond0 || bCond1);

            if(bTotalCond)
            {
                aiSameNodes[iNumSameNodes] = iCheckThreadNodeIndex;
                iNumSameNodes += 1u;
                iStartingWorkingGroup = min(iWorkgroupLoop, iStartingWorkingGroup);
                iMinSameNodeIndex = min(iCheckThreadNodeIndex, iMinSameNodeIndex);

                bInserted = true;
            }
        }
    }
    
    // didn't find a matching node, just add
    let iMesh: u32 = u32(floor(node.mCentroid.w + 0.5f) - 1.0f);
    if(iNumSameNodes == 0 || (iNumSameNodes == 1 && node.miLeftNodeIndex < node.miRightNodeIndex))
    {
        let iAddIndex: u32 = atomicAdd(&giNumTruncatedNodes, 1u);
        aWorkgroupIntermediateNodes[iAddIndex + iStartNodeIndex] = node;
    }
    else if(iNumSameNodes >= 2) 
    {
        // multiple nodes, just merge them into one
        if(iStartingWorkingGroup == workGroupIndex.x && iTotalThreadID <= iMinSameNodeIndex)
        {
            var minBounds: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
            var maxBounds: vec3<f32> = vec3<f32>(-FLT_MAX, -FLT_MAX, -FLT_MAX);
            for(var i: u32 = 0; i < iNumSameNodes; i++)
            {
                var iSameNodeIndex: u32 = aiSameNodes[i];

                minBounds = min(minBounds, aMergedIntermediateNodes[iSameNodeIndex].mMinBounds.xyz);
                maxBounds = max(maxBounds, aMergedIntermediateNodes[iSameNodeIndex].mMaxBounds.xyz);
            }

            var mergeNode: IntermediateNode;
            mergeNode.mCentroid = vec4<f32>((minBounds + maxBounds) * 0.5f, node.mCentroid.w);
            mergeNode.mMinBounds = vec4<f32>(minBounds, 1.0f);
            mergeNode.mMaxBounds = vec4<f32>(maxBounds, 1.0f);
            mergeNode.miLeftNodeIndex = iTotalThreadID + bvhProcessInfo.miStartNodeIndex;
            mergeNode.miRightNodeIndex = iTotalThreadID + bvhProcessInfo.miStartNodeIndex;

            let iAddIndex: u32 = atomicAdd(&giNumTruncatedNodes, 1u);
            aWorkgroupIntermediateNodes[iAddIndex + iStartNodeIndex] = mergeNode;
        }
    }

    workgroupBarrier();

    // save the number of nodes added
    if(iLocalThreadID == 0)
    {
        truncationInfo.maiNumValidNodes[workGroupIndex.x] = atomicLoad(&giNumTruncatedNodes);
    }
}