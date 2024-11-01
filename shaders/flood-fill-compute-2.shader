const FLT_MAX: f32 = 1.0e+10;
const UINT_MAX: u32 = 0xffffffff;

struct BrickInfo
{
    mPosition: vec3<f32>,
    miBrixelIndex: u32,
};

struct BrixelInfo
{
    maiBrixelDistances: array<u32, 512>,
    maBrixelBarycentricCoord: array<vec3<f32>, 512>
};

@group(0) @binding(0)
var<storage, read_write> aBrixelDistances: array<f32>;

@group(0) @binding(1)
var<storage, read_write> aBrixelBarycentricCoords: array<vec3<f32>>;

@group(0) @binding(2)
var<storage, read> aBricks: array<BrickInfo>;

@group(0) @binding(3)
var<storage, read> aBrixels: array<BrixelInfo>;

@group(0) @binding(4)
var<storage, read> aBBox: array<i32>;

@group(0) @binding(5)
var<storage, read_write> aiBrickToBrixelMapping: array<i32>;

@group(0) @binding(6)
var<storage, read_write> aiCounters: array<atomic<i32>, 256>;

struct UniformData
{
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    mfPositionScale: f32, 
    miNumBrixelsPerRow: u32,
};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

var<workgroup> gaiWorkgroupBrixelPositions: array<u32, 512>;
var<workgroup> gaiWorkgroupBrixelDistances: array<u32, 512>;
var<workgroup> gaWorkgroupBrixelBarycentricCoordinates: array<vec3<f32>, 512>;

var<workgroup> giNumValidBrixels: atomic<u32>;
var<workgroup> giSaveIndex: i32;

const iNumThreads: u32 = 256u;

/////
@compute
@workgroup_size(iNumThreads)
fn cs_main(
    @builtin(num_workgroups) numWorkGroups: vec3<u32>,
    @builtin(local_invocation_index) iLocalIndex: u32,
    @builtin(workgroup_id) workGroup: vec3<u32>)
{
    //if(uniformData.miFrameIndex != 1u)
    //{
    //    return;
    //}

    let iTotalThreadIndex: u32 = workGroup.x * iNumThreads + workGroup.y * iNumThreads + workGroup.z * iNumThreads + iLocalIndex;
    
    let meshBBox0: vec3<f32> = vec3<f32>(
        f32(aBBox[3]) * 0.001f,
        f32(aBBox[4]) * 0.001f,
        f32(aBBox[5]) * 0.001f
    ) * uniformData.mfPositionScale;

    let meshBBox1: vec3<f32> = vec3<f32>(
        f32(aBBox[0]) * 0.001f,
        f32(aBBox[1]) * 0.001f,
        f32(aBBox[2]) * 0.001f
    ) * uniformData.mfPositionScale;

    let bbox0: vec3<f32> = vec3<f32>(
        floor(meshBBox0.x),
        floor(meshBBox0.y),
        floor(meshBBox0.z)
    );

    let bbox1: vec3<f32> = vec3<f32>(
        ceil(meshBBox1.x),
        ceil(meshBBox1.y),
        ceil(meshBBox1.z)
    );

    let bboxDimension: vec3<u32> = vec3<u32>(
        u32(bbox1.x - bbox0.x),
        u32(bbox1.y - bbox0.y),
        u32(bbox1.z - bbox0.z)
    );

    let iTotalDimension: u32 = uniformData.miNumBrixelsPerRow * uniformData.miNumBrixelsPerRow * uniformData.miNumBrixelsPerRow;
    let iDimensionSquared: u32 = uniformData.miNumBrixelsPerRow * uniformData.miNumBrixelsPerRow;
    let iNumTotalBricks: u32 = bboxDimension.x * bboxDimension.y * bboxDimension.z;

    giSaveIndex = -1;

    // individual brick per workgroup
    var iCurrBrick: u32 = workGroup.x + workGroup.y * bboxDimension.x + workGroup.z * bboxDimension.x * bboxDimension.y;
    giNumValidBrixels = 0u;
    
    workgroupBarrier();

    var iBrixelIndex: u32 = iLocalIndex * 2u;
    gaiWorkgroupBrixelPositions[iBrixelIndex] = 0u;
    gaiWorkgroupBrixelPositions[iBrixelIndex + 1u] = 0u;

    // load and count number of valid brixels, even brixel index
    gaiWorkgroupBrixelDistances[iBrixelIndex] = UINT_MAX; 
    gaiWorkgroupBrixelDistances[iBrixelIndex] = aBrixels[aBricks[iCurrBrick].miBrixelIndex].maiBrixelDistances[iBrixelIndex];
    gaWorkgroupBrixelBarycentricCoordinates[iBrixelIndex] = aBrixels[aBricks[iCurrBrick].miBrixelIndex].maBrixelBarycentricCoord[iBrixelIndex];
    if(gaiWorkgroupBrixelDistances[iBrixelIndex] > 0u)
    {
        atomicAdd(&giNumValidBrixels, gaiWorkgroupBrixelDistances[iBrixelIndex]);

        let iZ: u32 = iBrixelIndex / iDimensionSquared;
        let iY: u32 = (iBrixelIndex - iZ * iDimensionSquared) / uniformData.miNumBrixelsPerRow;
        let iX: u32 = iBrixelIndex % uniformData.miNumBrixelsPerRow;

        let iSeedPosition: u32 = computeSeedIndex(iX, iY, iZ);
        gaiWorkgroupBrixelPositions[iBrixelIndex] = iSeedPosition;
    }

    // load and count number of valid brixels, odd brixel index 
    gaiWorkgroupBrixelDistances[iBrixelIndex + 1u] = UINT_MAX;
    gaiWorkgroupBrixelDistances[iBrixelIndex + 1u] = aBrixels[aBricks[iCurrBrick].miBrixelIndex].maiBrixelDistances[iBrixelIndex + 1u];
    gaWorkgroupBrixelBarycentricCoordinates[iBrixelIndex + 1u] = aBrixels[aBricks[iCurrBrick].miBrixelIndex].maBrixelBarycentricCoord[iBrixelIndex + 1u];
    if(gaiWorkgroupBrixelDistances[iBrixelIndex + 1u] > 0u)
    {
        atomicAdd(&giNumValidBrixels, gaiWorkgroupBrixelDistances[iBrixelIndex + 1u]);

        let iZ: u32 = (iBrixelIndex + 1u) / iDimensionSquared;
        let iY: u32 = ((iBrixelIndex + 1u) - iZ * iDimensionSquared) / uniformData.miNumBrixelsPerRow;
        let iX: u32 = (iBrixelIndex + 1u) % uniformData.miNumBrixelsPerRow;

        let iSeedPosition: u32 = computeSeedIndex(iX, iY, iZ);
        gaiWorkgroupBrixelPositions[iBrixelIndex + 1u] = iSeedPosition;
    }
    
    workgroupBarrier();

    // brick has brixels, allocate space for the brixel distances in the brick
    if(iLocalIndex == 0u && giNumValidBrixels > 0u)
    {
        if(aiBrickToBrixelMapping[iCurrBrick] == 0)
        {
            giSaveIndex = atomicAdd(&aiCounters[0], 1);
            aiBrickToBrixelMapping[iCurrBrick] = giSaveIndex;
        }
        else 
        {
            giSaveIndex = aiBrickToBrixelMapping[iCurrBrick];
        }
    }
    
    workgroupBarrier();

    // initialize brixel distance
    aBrixelDistances[u32(giSaveIndex * 512) + iLocalIndex * 2u] = 999999.0f;
    aBrixelDistances[u32(giSaveIndex * 512) + iLocalIndex * 2u + 1u] = 999999.0f;

    aBrixelBarycentricCoords[u32(giSaveIndex * 512) + iLocalIndex * 2u] = vec3<f32>(0.0f, 0.0f, 0.0f);
    aBrixelBarycentricCoords[u32(giSaveIndex * 512) + iLocalIndex * 2u + 1u] = vec3<f32>(0.0f, 0.0f, 0.0f);

    workgroupBarrier();

    // jump flood
    for(var iPass: u32 = 0u; iPass < 4u; iPass++)
    {
        // no valid brixels
        if(giSaveIndex < 0)
        {
            break;
        }

        for(var i: u32 = 0; i < 2u; i++)
        {
            iBrixelIndex = iLocalIndex * 2u + i;
            let iZ: u32 = iBrixelIndex / iDimensionSquared;
            let iY: u32 = (iBrixelIndex - iZ * iDimensionSquared) / uniformData.miNumBrixelsPerRow;
            let iX: u32 = iBrixelIndex % uniformData.miNumBrixelsPerRow;

            jumpFlood3D(
                iX,
                iY,
                iZ,
                uniformData.miNumBrixelsPerRow,
                uniformData.miNumBrixelsPerRow,
                uniformData.miNumBrixelsPerRow,
                4u,
                uniformData.mfBrixelDimension,
                u32(giSaveIndex * 512),
                u32(iCurrBrick),
                0u,
                u32(giSaveIndex)
            );

            workgroupBarrier();
            jumpFlood3D(
                iX,
                iY,
                iZ,
                uniformData.miNumBrixelsPerRow,
                uniformData.miNumBrixelsPerRow,
                uniformData.miNumBrixelsPerRow,
                2u,
                uniformData.mfBrixelDimension,
                u32(giSaveIndex * 512),
                u32(iCurrBrick),
                1u,
                u32(giSaveIndex)
            );

            workgroupBarrier();
            jumpFlood3D(
                iX,
                iY,
                iZ,
                uniformData.miNumBrixelsPerRow,
                uniformData.miNumBrixelsPerRow,
                uniformData.miNumBrixelsPerRow,
                1u,
                uniformData.mfBrixelDimension,
                u32(giSaveIndex * 512),
                u32(iCurrBrick),
                2u,
                u32(giSaveIndex)
            );

        }
    }
        
}

/////
fn jumpFlood3D(
    iOrigX: u32,
    iOrigY: u32,
    iOrigZ: u32,
    iImageWidth: u32,
    iImageHeight: u32,
    iImageDepth: u32,
    iStepSize: u32,
    fBrixelSize: f32,
    iStartingBrixelIndex: u32,
    iBrickIndex: u32,
    iPass: u32,
    iDebugIndex: u32)
{
    // save with the best distance to seed
    let iIndex: u32 = iOrigZ * iImageWidth * iImageHeight + iOrigY * iImageWidth + iOrigX;

    var iBestSeedPosition: u32 = gaiWorkgroupBrixelPositions[iIndex];
    let iDecodedSeedPosition: u32 = iBestSeedPosition - 1u;
    let iSeedZ: u32 = (iDecodedSeedPosition >> 20u);
    let iSeedY: u32 = ((iDecodedSeedPosition & ((1u << 20u) - 1u)) >> 10u);
    let iSeedX: u32 = (iDecodedSeedPosition & ((1u << 10u) - 1u));
    let diff: vec3<f32> = vec3<f32>(f32(iSeedX - iOrigX), f32(iSeedY - iOrigY), f32(iSeedZ - iOrigZ));
    var fBestDistance: f32 = sqrt(dot(diff, diff));
    if(iBestSeedPosition == 0u)
    {
        fBestDistance = FLT_MAX;
    }
    var bestBarycentricCoord: vec3<f32> = gaWorkgroupBrixelBarycentricCoordinates[iIndex];

    for(var iZ: i32 = -1; iZ <= 1; iZ++)
    {
        if(i32(iOrigZ) + iZ * i32(iStepSize) < 0 || i32(iOrigZ) + iZ * i32(iStepSize) >= i32(iImageDepth))
        {
            continue;
        }

        for(var iY: i32 = -1; iY <= 1; iY++)
        {
            if(i32(iOrigY) + iY * i32(iStepSize) < 0 || i32(iOrigY) + iY * i32(iStepSize) >= i32(iImageHeight))
            {
                continue;
            }

            for(var iX: i32 = -1; iX <= 1; iX++)
            {
                if(i32(iOrigX) + iX * i32(iStepSize) < 0 || i32(iOrigX) * i32(iStepSize) + iX >= i32(iImageWidth))
                {
                    continue;
                }

                let iSampleX: i32 = min(i32(max(i32(iOrigX) + iX * i32(iStepSize), 0)), i32(iImageWidth));
                let iSampleY: i32 = min(i32(max(i32(iOrigY) + iY * i32(iStepSize), 0)), i32(iImageHeight));
                let iSampleZ: i32 = min(i32(max(i32(iOrigZ) + iZ * i32(iStepSize), 0)), i32(iImageDepth));
                let iSampleIndex: i32 = iSampleZ * i32(iImageWidth) * i32(iImageHeight) + iSampleY * i32(iImageWidth) + iSampleX;

                let iSeedPosition: u32 = gaiWorkgroupBrixelPositions[iSampleIndex];
                if(iSeedPosition > 0u)
                {
                    let iDecodedSeedPosition: u32 = iSeedPosition - 1u;

                    // seed coordinate
                    let iSeedZ: u32 = (iDecodedSeedPosition >> 20u);
                    let iSeedY: u32 = ((iDecodedSeedPosition & ((1u << 20u) - 1u)) >> 10u);
                    let iSeedX: u32 = (iDecodedSeedPosition & ((1u << 10u) - 1u));

                    // distance to the seed
                    let diff: vec3<f32> = vec3<f32>(f32(iSeedX - iOrigX), f32(iSeedY - iOrigY), f32(iSeedZ - iOrigZ));
                    let fDistance: f32 = sqrt(dot(diff, diff));
                    if(fDistance < fBestDistance)
                    {
                        // best distance 
                        fBestDistance = fDistance;
                        iBestSeedPosition = iSeedPosition;
                        bestBarycentricCoord = gaWorkgroupBrixelBarycentricCoordinates[iSampleIndex];
                    }
                }
            
            }   // for x
        
        }   // for y

    }   // for z 

    if(fBestDistance != FLT_MAX && iBestSeedPosition > 0u)
    { 
        let iDecodedSeedPosition: u32 = iBestSeedPosition - 1u;

        let iSeedZ: u32 = (iDecodedSeedPosition >> 20u);
        let iSeedY: u32 = ((iDecodedSeedPosition & ((1u << 20u) - 1u)) >> 10u);
        let iSeedX: u32 = (iDecodedSeedPosition & ((1u << 10u) - 1u));

        let iDiffX: u32 = iOrigX - iSeedX;
        let iDiffY: u32 = iOrigY - iSeedY;
        let iDiffZ: u32 = iOrigZ - iSeedZ;

        let fDistance: f32 = sqrt(f32(iDiffX * iDiffX) + f32(iDiffY * iDiffY) + f32(iDiffZ * iDiffZ));
        aBrixelDistances[iStartingBrixelIndex + iIndex] = fDistance;
        aBrixelBarycentricCoords[iStartingBrixelIndex + iIndex] = bestBarycentricCoord;

        gaiWorkgroupBrixelPositions[iIndex] = iBestSeedPosition;
        gaWorkgroupBrixelBarycentricCoordinates[iIndex] = bestBarycentricCoord;
    }

}

/////
fn computeSeedIndex(
    iX: u32,
    iY: u32,
    iZ: u32) -> u32
{
    return ((iZ << 20) | (iY << 10) | iX) + 1u;
}
