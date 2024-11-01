const FLT_MAX: f32 = 9999999.0f;

struct BrickInfo
{
    mPosition: vec3<f32>,
    miBrixelIndex: u32,
};

struct BrixelInfo
{
    maiBrixelDistances: array<u32, 512>,
    maBrixelBarycentricCoord: array<vec3<f32>, 512>,
};

struct UniformData
{
    mfPositionScale: f32, 
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    miFrameIndex: u32,
};

@group(0) @binding(0)
var<storage, read_write> aBrixels: array<BrixelInfo>;

@group(0) @binding(1)
var<storage, read> aBricks: array<BrickInfo>;

@group(0) @binding(2)
var<storage, read> aBBox: array<i32>;

@group(0) @binding(3)
var<storage, read> aTrianglePositions: array<vec4<f32>>;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> aiTriangleIndices: array<u32>;

@group(0) @binding(4)
var<storage, read_write> debugBuffer: array<vec4<f32>>;

const iNumThreads: u32 = 256u;

/////
@compute
@workgroup_size(iNumThreads)
fn cs_main(
    @builtin(num_workgroups) numWorkGroups: vec3<u32>,
    @builtin(local_invocation_index) iLocalIndex: u32,
    @builtin(workgroup_id) workGroup: vec3<u32>)
{
    let iNumTotalThreads: u32 = numWorkGroups.x * iNumThreads;
    let iTotalThreadIndex: u32 = workGroup.x * iNumThreads + iLocalIndex;

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

    let brickSize: vec3<f32> = vec3<f32>(
        uniformData.mfBrickDimension, 
        uniformData.mfBrickDimension, 
        uniformData.mfBrickDimension);

    let bboxBrixelDimension = (bbox1 - bbox0) / uniformData.mfBrixelDimension;
    let bboxBrickDimension = (bbox1 - bbox0) / uniformData.mfBrickDimension;

    var iIndex: u32 = iTotalThreadIndex;
    for(var iLoop: u32 = 0u; iLoop < 10000u; iLoop++)
    {
        var fY: f32 = bbox0.y + floor(f32(iIndex) / bboxBrixelDimension.x) * uniformData.mfBrixelDimension;
        var fX: f32 = bbox0.x + f32(iIndex % u32(bboxBrixelDimension.x)) * uniformData.mfBrixelDimension;

        if(fY > bbox1.y)
        {
            break;
        }

        debugBuffer[iLoop * iNumTotalThreads + iTotalThreadIndex] = vec4<f32>(
            f32(iLoop * iNumTotalThreads + iTotalThreadIndex),
            fX,
            fY,
            f32(iTotalThreadIndex)
        );
        
        var fLastZ: f32 = FLT_MAX;
        for(var fZ: f32 = bbox0.z; fZ <= bbox1.z; fZ += uniformData.mfBrixelDimension)
        {
            let position: vec3<f32> = vec3<f32>(fX, fY, fZ);

            let brickIndex: vec3<f32> = (position - bbox0) / brickSize;
            let iBrickArrayIndex: u32 = 
                u32(floor(brickIndex.x)) + 
                u32(floor(brickIndex.y)) * u32(bboxBrickDimension.x) +
                u32(floor(brickIndex.z)) * u32(bboxBrickDimension.x) * u32(bboxBrickDimension.y);
            let brixelIndex: vec3<f32> = fract(brickIndex) * 8.0f;
            let iBrixelArrayIndex: u32 = u32(brixelIndex.z) * 64u + u32(brixelIndex.y) * 8u + u32(brixelIndex.x);

            let iCurrBrixelDistance: u32 = aBrixels[iBrickArrayIndex].maiBrixelDistances[iBrixelArrayIndex];
            if(iCurrBrixelDistance > 0u)
            {
                if(fLastZ != FLT_MAX)
                {
                    // fill in between
                    let iNumInbetweenBrixels: u32 = u32(ceil((fZ - fLastZ) / uniformData.mfBrixelDimension));
                    for(var iZ: u32 = 1; iZ <= iNumInbetweenBrixels; iZ++)
                    {
                        let currScanPosition: vec3<f32> = vec3<f32>(fX, fY, fLastZ + f32(iZ) * uniformData.mfBrixelDimension);
                        let scanBrickIndex: vec3<f32> = (currScanPosition - bbox0) / brickSize;
                        let iScanBrickArrayIndex: u32 = 
                            u32(floor(scanBrickIndex.x)) + 
                            u32(floor(scanBrickIndex.y)) * u32(bboxBrickDimension.x) +
                            u32(floor(scanBrickIndex.z)) * u32(bboxBrickDimension.x) * u32(bboxBrickDimension.y);
                        let scanBrixelIndex: vec3<f32> = fract(scanBrickIndex) * 8.0f;
                        let iScanBrixelArrayIndex: u32 = u32(scanBrixelIndex.z) * 64u + u32(scanBrixelIndex.y) * 8u + u32(scanBrixelIndex.x);

                        if(aBrixels[iScanBrickArrayIndex].maiBrixelDistances[iScanBrixelArrayIndex] == 0u)
                        {
                            aBrixels[iScanBrickArrayIndex].maiBrixelDistances[iScanBrixelArrayIndex] = 1u;
                        }
                    }
                }
                fLastZ = fZ;
            }
            
        }

        iIndex += iNumTotalThreads;
    }


}