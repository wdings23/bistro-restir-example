var<workgroup> giNodeWorkIndex: atomic<i32>;
var<workgroup> giNumNodes: atomic<i32>;

var<workgroup> abWorking: array<atomic<i32>, 256>;
var<workgroup> aiWorkNodeIndex: array<u32, 64>;

const NUM_BINS: u32 = 8;
const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;

struct BVHNode
{
    mBoundingBoxMin : vec4<f32>,
    mBoundingBoxMax : vec4<f32>,
    miLeftFirst : u32,
    miNumTriangles : u32,
    miLevel : u32,
    mPadding : u32
};

struct Tri
{
    miV0 : u32,
    miV1 : u32,
    miV2 : u32,
    mPadding : u32,

    mCentroid : vec4<f32>
};

struct AxisAlignedBoundingBox
{
    mMin : vec4<f32>,
    mMax : vec4<f32>
};

struct Bin
{
    mBoundingBox : AxisAlignedBoundingBox,
    miNumTriangles : u32,
    
    miPadding0 : u32,
    miPadding1 : u32,
    miPadding2 : u32
};

struct SubDivisionInfo
{
    mfBestCost: f32,
    miAxis: u32,
    miSplitBinIndex: u32
};

struct MeshInfo
{
    miPositionOffsetStart: u32,
    miNumPositions: u32,

    miVertexIndexOffsetStart: u32,
    miNumTriangles: u32
};

struct MeshInfoList
{
    miNumMeshes: u32,
    maMeshInfo: array<MeshInfo, 4>
};

struct OutputData
{
    miNumNodes: u32,
}

struct DebugData
{
    maBoundingBox: array<AxisAlignedBoundingBox, 24>,
    maiNumTriangles: array<u32, 24>,
    
    maiLeftNumTriangles: array<u32, 24>,
    mafLeftAreas: array<f32, 24>,
    maiRightNumTriangles: array<u32, 24>,
    mafRightAreas: array<f32, 24>,
    
    maLeftBBox: array<AxisAlignedBoundingBox, 24>,
    maRightBBox: array<AxisAlignedBoundingBox, 24>,

    mafLeftCosts: array<f32, 24>,
    mafRightCosts: array<f32, 24>,

    maNodeBBox: array<AxisAlignedBoundingBox, 8>,

    mWritten: vec4<u32>
};

@group(0) @binding(0)
var<storage, read_write> nodeList: array<BVHNode>;

@group(1) @binding(0)
var<storage, read> positions: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read> triangles: array<Tri>;

@group(1) @binding(2)
var<storage, read_write> triangleIndices: array<u32>;

@group(1) @binding(3)
var<storage, read> meshInfoList: array<MeshInfo, 32>;

@group(1) @binding(4)
var<storage, read_write> outputData: OutputData;

@group(1) @binding(5)
var<storage, read_write> debugData: DebugData;

const iNumThreads: u32 = 16u;

@compute
@workgroup_size(iNumThreads)
fn cs_main(@builtin(global_invocation_id) index: vec3<u32>) 
{
    let iThreadID: u32 = u32(index.x);

    var bFirstTimeBuild: bool = false;
    if(outputData.miNumNodes == 0)
    {
        bFirstTimeBuild = true;
    }

if(!bFirstTimeBuild)
{
    return;
}

    for(var i: u32 = 0; i < 100u; i++)
    {
        var index: u32 = iThreadID + i * iNumThreads;
        nodeList[i].miLevel = 0u;
    }
    workgroupBarrier();


    if(iThreadID == 0)
    {
        giNodeWorkIndex = 0;
        giNumNodes = 0;

        let iNodeIndex: i32 = atomicAdd(&giNumNodes, 1);
        
        nodeList[iNodeIndex].miLeftFirst = u32(0);
        nodeList[iNodeIndex].miNumTriangles = meshInfoList[0].miNumTriangles;
        nodeList[iNodeIndex] = updateNodeBound(nodeList[iNodeIndex]);
        nodeList[iNodeIndex].miLevel = 1u;

        debugData.maNodeBBox[0].mMin = nodeList[iNodeIndex].mBoundingBoxMin;
        debugData.maNodeBBox[0].mMax = nodeList[iNodeIndex].mBoundingBoxMax;

        debugData.mWritten = vec4<u32>(0u, 0u, 0u, 0u);
    }
    abWorking[iThreadID] = 0;
    aiWorkNodeIndex[iThreadID] = iThreadID;
    workgroupBarrier();

    //for(var i: u32 = 0; i < 100u; i++)
    for(;;)
    {   
        let iCurrNodeWorkIndex: u32 = aiWorkNodeIndex[iThreadID];
        if(nodeList[iCurrNodeWorkIndex].miLevel > 0)
        {
            atomicStore(&abWorking[iThreadID], 1);
            var bHasSplitNode: bool = subdivideNode(
                u32(iCurrNodeWorkIndex),
                true);
            nodeList[iCurrNodeWorkIndex].mPadding = iThreadID;
            aiWorkNodeIndex[iThreadID] += iNumThreads;
            atomicStore(&abWorking[iThreadID], 0);
        }

        if(!bFirstTimeBuild && aiWorkNodeIndex[iThreadID] >= outputData.miNumNodes)
        {
            break;
        }

        var bWorking: bool = false;
        var iSmallestNodeToWork: u32 = UINT32_MAX;
        for(var j: u32 = 0; j < iNumThreads; j++)
        {
            var iThreadWorking: i32 = atomicLoad(&abWorking[j]);
            if(iThreadWorking > 0)
            {
                bWorking = true;
            }

            if(iSmallestNodeToWork > aiWorkNodeIndex[j])
            {
                iSmallestNodeToWork = aiWorkNodeIndex[j];
            }
        }

        if(!bWorking && giNumNodes <= i32(iSmallestNodeToWork))
        {
            break;
        }

    }
    workgroupBarrier();

    outputData.miNumNodes = max(u32(atomicLoad(&giNumNodes)), outputData.miNumNodes);
}

/////
fn updateNodeBound(
    inputNode : BVHNode) -> BVHNode
{
    var node : BVHNode = inputNode;
    node.mBoundingBoxMin = vec4<f32>(1.0e+10, 1.0e+10, 1.0e+10, 0.0);
    node.mBoundingBoxMax = vec4<f32>(-1.0e+10, -1.0e+10, -1.0e+10, 0.0);

    for(var i: u32 = 0; i < node.miNumTriangles; i++)
    {
        var iTriangleIndex: u32 = triangleIndices[node.miLeftFirst + i];
        let triangle: Tri = triangles[iTriangleIndex];

        let pos0 : vec4<f32> = positions[triangle.miV0];
        let pos1 : vec4<f32> = positions[triangle.miV1];
        let pos2 : vec4<f32> = positions[triangle.miV2];

        node.mBoundingBoxMin = min(pos0, node.mBoundingBoxMin);
        node.mBoundingBoxMin = min(pos1, node.mBoundingBoxMin);
        node.mBoundingBoxMin = min(pos2, node.mBoundingBoxMin);

        node.mBoundingBoxMax = max(pos0, node.mBoundingBoxMax);
        node.mBoundingBoxMax = max(pos1, node.mBoundingBoxMax);
        node.mBoundingBoxMax = max(pos2, node.mBoundingBoxMax);
    }

    return node;
};

/////
fn addBVHNode(
    iNodeIndex : u32,
    iLeftFirst : u32,
    iNumTriangles : u32,
    iLevel : u32)
{
    var node: BVHNode;
    node.miLeftFirst = iLeftFirst;
    node.miNumTriangles = iNumTriangles;
    node.miLevel = iLevel;
    updateNodeBound(node);

    nodeList[iNodeIndex] = node;
}

/////
fn computeArea(bbox : AxisAlignedBoundingBox) -> f32
{
    let diff: vec4<f32> = bbox.mMax - bbox.mMin; 
    return diff.x * diff.y + diff.x * diff.z + diff.y * diff.z;
}


/////
fn growBoundingBoxSize(
    bboxInput: AxisAlignedBoundingBox,
    v: vec4<f32>) -> AxisAlignedBoundingBox
{
    var bbox: AxisAlignedBoundingBox = bboxInput;

    bbox.mMax.x = max(v.x, bbox.mMax.x);
    bbox.mMax.y = max(v.y, bbox.mMax.y);
    bbox.mMax.z = max(v.z, bbox.mMax.z);

    bbox.mMin.x = min(v.x, bbox.mMin.x);
    bbox.mMin.y = min(v.y, bbox.mMin.y);
    bbox.mMin.z = min(v.z, bbox.mMin.z);

    return bbox;
}

/////
fn growBoundingBoxSize2(
    bboxInput: AxisAlignedBoundingBox,
    minVal: vec4<f32>,
    maxVal: vec4<f32>) -> AxisAlignedBoundingBox
{
    var bbox: AxisAlignedBoundingBox = bboxInput;

    bbox.mMax.x = max(maxVal.x, bbox.mMax.x);
    bbox.mMax.y = max(maxVal.y, bbox.mMax.y);
    bbox.mMax.z = max(maxVal.z, bbox.mMax.z);

    bbox.mMin.x = min(minVal.x, bbox.mMin.x);
    bbox.mMin.y = min(minVal.y, bbox.mMin.y);
    bbox.mMin.z = min(minVal.z, bbox.mMin.z);

    return bbox;
}

/////
fn growBinSize(
    binInput: Bin,
    triangle: Tri) -> Bin
{
    var bin: Bin = binInput;

    let pos0: vec4<f32> = positions[triangle.miV0];
    let pos1: vec4<f32> = positions[triangle.miV1];
    let pos2: vec4<f32> = positions[triangle.miV2];

    bin.mBoundingBox = growBoundingBoxSize(
        bin.mBoundingBox,
        pos0);

    bin.mBoundingBox = growBoundingBoxSize(
        bin.mBoundingBox,
        pos1);

    bin.mBoundingBox = growBoundingBoxSize(
        bin.mBoundingBox,
        pos2);

    bin.miNumTriangles = binInput.miNumTriangles + 1u;

    return bin;
}

/////
fn findBestSubDivision(
    node: BVHNode,
    centroidMin: vec4<f32>,
    centroidMax: vec4<f32>) -> SubDivisionInfo
{
    var ret: SubDivisionInfo;

    var fBestCost: f32 = 1.0e+10;
    var fBestValue: f32 = 1.0e+10;
    
    var iRetAxis: u32 = UINT32_MAX;
    var iRetSplit: u32 = UINT32_MAX;

    for(var iAxis: u32 = 0; iAxis < 3u; iAxis++)
    {
        var fBoundsMin: f32 = centroidMin.x;
        var fBoundsMax: f32 = centroidMax.x;
        if(iAxis == 1u)
        {
            fBoundsMin = centroidMin.y;
            fBoundsMax = centroidMax.y;
        }
        else if(iAxis == 2u)
        {
            fBoundsMin = centroidMin.z;
            fBoundsMax = centroidMax.z;
        }

        var fScale: f32 = f32(NUM_BINS) / (fBoundsMax - fBoundsMin);

        // initialize bounds
        var aBins: array<Bin, NUM_BINS>;
        for(var i: u32 = 0u; i < NUM_BINS; i++)
        {
            aBins[i].mBoundingBox.mMin = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, 0.0);
            aBins[i].mBoundingBox.mMax = vec4<f32>(-FLT_MAX, -FLT_MAX, -FLT_MAX, 0.0);
            aBins[i].miNumTriangles = 0u;
        }
    
        // partition triangles to all the bins
        for(var iTri: u32 = 0u; iTri < node.miNumTriangles; iTri++)
        {
            let triangle: Tri = triangles[iTri + node.miLeftFirst];

            // choose the bin
            var fCentroidComponent: f32 = triangle.mCentroid.x;
            if(iAxis == 1u)
            {
                fCentroidComponent = triangle.mCentroid.y;
            }
            else if(iAxis == 2u)
            {
                fCentroidComponent = triangle.mCentroid.z;
            }
            
            var fTriangleToNodeCentroid: f32 = max(fCentroidComponent - fBoundsMin, 0.0);
            let iBinIndex: u32 = min(
                u32(NUM_BINS - 1),
                u32((fTriangleToNodeCentroid * fScale))
            );

            // grow bin size
            aBins[iBinIndex] = growBinSize(
                aBins[iBinIndex],
                triangle);
        }

        // compute areas for left and right paritions of the bins
        var afLeftCountArea: array<f32, 7> = array<f32, 7>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        var afRightCountArea: array<f32, 7> = array<f32, 7>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f);
        var iNumTriangleLeft: u32 = 0;
        var iNumTriangleRight: u32 = 0;
        var aiLeftCount: array<u32, 7> =  array<u32, 7>(0u, 0u, 0u, 0u, 0u, 0u, 0u);
        var aiRightCount: array<u32, 7> = array<u32, 7>(0u, 0u, 0u, 0u, 0u, 0u, 0u);

        var leftBBox: AxisAlignedBoundingBox;
        var rightBBox: AxisAlignedBoundingBox;

        // initialize bounding boxes
        leftBBox.mMin = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
        leftBBox.mMax = vec4<f32>(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
        rightBBox.mMin = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);
        rightBBox.mMax = vec4<f32>(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);

        var aLeftBBox: array<AxisAlignedBoundingBox, 8>;
        var aRightBBox: array<AxisAlignedBoundingBox, 8>; 
        for(var i: u32 = 0u; i < NUM_BINS - 1; i++)
        {
            // left
            let iLeftBinIndex: u32 = i;
            let iLeftIndex: u32 = i;
            iNumTriangleLeft += aBins[iLeftBinIndex].miNumTriangles;
            aiLeftCount[iLeftIndex] = iNumTriangleLeft;

            // update left bounding box with bin's bounding box
            leftBBox.mMax = max(aBins[iLeftBinIndex].mBoundingBox.mMax, leftBBox.mMax);
            leftBBox.mMin = min(aBins[iLeftBinIndex].mBoundingBox.mMin, leftBBox.mMin);
            aLeftBBox[iLeftIndex] = leftBBox;

            // area with number of triangles in the split
            afLeftCountArea[iLeftIndex] = f32(iNumTriangleLeft) * computeArea(leftBBox);

            // right
            let iRightBinIndex: u32 = NUM_BINS - 1u - i;
            let iRightIndex: u32 = NUM_BINS - 2u - i;
            iNumTriangleRight += aBins[iRightBinIndex].miNumTriangles;
            aiRightCount[iRightIndex] = iNumTriangleRight;

            // update right bounding box with bin's bounding box
            rightBBox.mMax = max(aBins[iRightBinIndex].mBoundingBox.mMax, rightBBox.mMax);
            rightBBox.mMin = min(aBins[iRightBinIndex].mBoundingBox.mMin, rightBBox.mMin);
            aRightBBox[iRightIndex] = rightBBox;

            // area with number of triangles in the split
            afRightCountArea[iRightIndex] = f32(iNumTriangleRight) * computeArea(rightBBox);
        }

        // compute cost and store the lowest one
        fScale = (fBoundsMax - fBoundsMin) / f32(NUM_BINS);
        for(var i: u32 = 0; i < NUM_BINS - 1; i++)
        {
            let fPlaneCost: f32 = afLeftCountArea[i] + afRightCountArea[i];
            if(fPlaneCost < fBestCost)
            {
                iRetAxis = iAxis;
                iRetSplit = i + 1;
                fBestValue = fBoundsMin + fScale * f32(i + 1);
                fBestCost = fPlaneCost;
            }
        }
        
    }   // for axis = 0 to 3

    ret.miAxis = iRetAxis;
    ret.miSplitBinIndex = iRetSplit;
    ret.mfBestCost = fBestCost;

    return ret;
}

/////
fn computeNodeCost(node: BVHNode) -> f32
{
    let diff: vec3<f32> = 
        vec3<f32>(node.mBoundingBoxMax.x, node.mBoundingBoxMax.y, node.mBoundingBoxMax.z) - 
        vec3<f32>(node.mBoundingBoxMin.x, node.mBoundingBoxMin.y, node.mBoundingBoxMin.z);

    return (diff.x * diff.y + diff.y * diff.z + diff.z * diff.x) * f32(node.miNumTriangles);
}

/////
fn subdivideNode(
    iNode: u32,
    bSplitDownToOneTriangle: bool) -> bool
{
    var node : BVHNode = nodeList[iNode];

    var fBestCost: f32 = FLT_MAX;
    var fSplitComponentValue: f32 = FLT_MAX;
    var iAxis: u32 = UINT32_MAX;
    var iSplitBinIndex: u32 = UINT32_MAX;

    // get the best subdivision axis and partition index
    let subDivisionInfo: SubDivisionInfo = findBestSubDivision(
        node,
        node.mBoundingBoxMin,
        node.mBoundingBoxMax);
    iAxis = subDivisionInfo.miAxis;
    iSplitBinIndex = subDivisionInfo.miSplitBinIndex; 
    fBestCost = subDivisionInfo.mfBestCost;

    if(bSplitDownToOneTriangle)
    {
        if(node.miNumTriangles == 1u)
        {
            return false;
        }
    }
    else
    {
        let fNoSplitCost: f32 = computeNodeCost(node);
        if(fNoSplitCost < fBestCost)
        {
            return false;
        }
    }

    // partition
    var i: i32 = i32(node.miLeftFirst);
    var j: i32 = i32(i + i32(node.miNumTriangles - 1));
    var fCentroidMax: f32 = node.mBoundingBoxMax.x;
    var fCentroidMin: f32 = node.mBoundingBoxMin.x;
    if(iAxis == 1u)
    {
        fCentroidMax = node.mBoundingBoxMax.y;
        fCentroidMin = node.mBoundingBoxMin.y;
    }
    else if(iAxis == 2u)
    {
        fCentroidMax = node.mBoundingBoxMax.z;
        fCentroidMin = node.mBoundingBoxMin.z;
    }
    
    var fScale: f32 = f32(NUM_BINS) / (fCentroidMax - fCentroidMin);

    // sort the triangle based on centroid
    // NOTE: large triangles can expand into left and right bins
    var iAxisAddition: u32 = 0;
    for(var iTest: i32 = 0; iTest < i32(NUM_BINS) * 3; iTest++)
    {
        while(i <= j)
        {
            let iTriangleIndex: u32 = triangleIndices[i];
            let triangle: Tri = triangles[iTriangleIndex];
            var fTriangleCentroid: f32 = triangle.mCentroid.x;
            let iCheckAxis: u32 = (iAxis + iAxisAddition) % 3;
            if(iCheckAxis == 1u)
            {
                fTriangleCentroid = triangle.mCentroid.y;
            }
            else if(iCheckAxis == 2u)
            {
                fTriangleCentroid = triangle.mCentroid.z;
            }

            var fTriangleToNodeCentroid: f32 = max(fTriangleCentroid - fCentroidMin, 0.0f);
            let iBinIndex: u32 = min(
                u32(NUM_BINS - 1),
                u32((fTriangleToNodeCentroid * fScale))
            );

            if(iBinIndex < iSplitBinIndex)
            {
                i += 1;
            }
            else
            {
                // swap triangle indices, adding triangle i to the right node and j to the left node
                let iTemp: u32 = triangleIndices[i];
                triangleIndices[i] = triangleIndices[j];
                triangleIndices[j] = iTemp;
                j -= 1;
            }
        }

        var iLeftCount: u32 = u32(i - i32(node.miLeftFirst));
        if(iLeftCount == 0 || iLeftCount >= node.miNumTriangles)
        {
            // this split is lop-sided with all the triangles at one side
            // keep checking for better split index
            iSplitBinIndex = (iSplitBinIndex + 1u) % NUM_BINS;
            i = i32(node.miLeftFirst);
            j = i + i32(node.miNumTriangles) - 1;
            iAxisAddition = u32(f32(iTest) / f32(NUM_BINS));
        }
        else
        {
            break;
        }
    }

    // don't split if one side is empty
    let iLeftCount: u32 = u32(i - i32(node.miLeftFirst));
    if(iLeftCount == 0 || iLeftCount == node.miNumTriangles)
    {
        return false;
    }

    // create left and right node
    let iChildNodeIndex: u32 = u32(atomicAdd(&giNumNodes, 2));

    nodeList[iChildNodeIndex].miLeftFirst = node.miLeftFirst;
    nodeList[iChildNodeIndex].miNumTriangles = iLeftCount;
    nodeList[iChildNodeIndex].miLevel = node.miLevel + 1;
    nodeList[iChildNodeIndex] = updateNodeBound(nodeList[iChildNodeIndex]);

    nodeList[iChildNodeIndex + 1].miLeftFirst = node.miLeftFirst + iLeftCount;
    nodeList[iChildNodeIndex + 1].miNumTriangles = node.miNumTriangles - iLeftCount;
    nodeList[iChildNodeIndex + 1].miLevel = node.miLevel + 1;
    nodeList[iChildNodeIndex + 1] = updateNodeBound(nodeList[iChildNodeIndex + 1]);

    // write child node index and clear out the triangle count
    nodeList[iNode].miLeftFirst = iChildNodeIndex;
    nodeList[iNode].miNumTriangles = u32(0);

    debugData.maNodeBBox[iChildNodeIndex].mMin = nodeList[iChildNodeIndex].mBoundingBoxMin;
    debugData.maNodeBBox[iChildNodeIndex].mMax = nodeList[iChildNodeIndex].mBoundingBoxMax;
    debugData.maNodeBBox[iChildNodeIndex + 1].mMin = nodeList[iChildNodeIndex + 1].mBoundingBoxMin;
    debugData.maNodeBBox[iChildNodeIndex + 1].mMax = nodeList[iChildNodeIndex + 1].mBoundingBoxMax;

    return true;
};
