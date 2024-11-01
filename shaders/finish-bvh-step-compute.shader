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

struct NodeLevelRange
{
    miStartNodeIndex: u32,
    miEndNodeIndex: u32,
    miLevel: u32,
};

@group(0) @binding(0)
var<storage, read_write> bvhProcessInfo: BVHProcessInfo;

@group(0) @binding(1)
var<storage, read> truncationInfo: TruncationInfo;

@group(0) @binding(2)
var<storage, read_write> nodeLevelRange: array<NodeLevelRange, 128>;

//////
@compute
@workgroup_size(1)
fn cs_main(
    @builtin(local_invocation_id) localIndex: vec3<u32>,
    @builtin(workgroup_id) workGroupIndex: vec3<u32>,
    @builtin(num_workgroups) numWorkGroups: vec3<u32>) 
{
    if(workGroupIndex.x == 0 && localIndex.x == 0)
    {
        var iNumWorkGroups: u32 = u32(ceil(f32(bvhProcessInfo.miEndNodeIndex - bvhProcessInfo.miStartNodeIndex) / 256.0f));
        var iNumNewNodes: u32 = 0u;
        for(var i: u32 = 0; i < iNumWorkGroups; i++)
        {
            iNumNewNodes += truncationInfo.maiNumValidNodes[i];
        }

        // for debugging
        nodeLevelRange[bvhProcessInfo.miStep].miLevel = bvhProcessInfo.miStep;
        nodeLevelRange[bvhProcessInfo.miStep].miStartNodeIndex = bvhProcessInfo.miStartNodeIndex;
        nodeLevelRange[bvhProcessInfo.miStep].miEndNodeIndex = bvhProcessInfo.miEndNodeIndex;

        bvhProcessInfo.miStep += 1u;
        bvhProcessInfo.miStartNodeIndex = bvhProcessInfo.miEndNodeIndex;
        bvhProcessInfo.miEndNodeIndex += iNumNewNodes;

        // for debugging
        nodeLevelRange[bvhProcessInfo.miStep].miLevel = bvhProcessInfo.miStep;
        nodeLevelRange[bvhProcessInfo.miStep].miStartNodeIndex = bvhProcessInfo.miStartNodeIndex;
        nodeLevelRange[bvhProcessInfo.miStep].miEndNodeIndex = bvhProcessInfo.miEndNodeIndex;
    }
}