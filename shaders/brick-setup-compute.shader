const UINT_MAX: u32 = 9999u;

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

struct DefaultUniformData
{
    miScreenWidth: i32,
    miScreenHeight: i32,
    miFrame: i32,
    miNumMeshes: u32,

    mfRand0: f32,
    mfRand1: f32,
    mfRand2: f32,
    mfRand3: f32,

    mViewProjectionMatrix: mat4x4<f32>,
    mPrevViewProjectionMatrix: mat4x4<f32>,
    mViewMatrix: mat4x4<f32>,
    mProjectionMatrix: mat4x4<f32>,

    mJitteredViewProjectionMatrix: mat4x4<f32>,
    mPrevJitteredViewProjectionMatrix: mat4x4<f32>,

    mCameraPosition: vec4<f32>,
    mCameraLookDir: vec4<f32>,

    mLightRadiance: vec4<f32>,
    mLightDirection: vec4<f32>,

    mfAmbientOcclusionDistanceThreshold: f32,
};

@group(0) @binding(0)
var<storage, read_write> aBricks: array<BrickInfo>;

@group(0) @binding(1)
var<storage, read_write> aBrixels: array<BrixelInfo>;

@group(0) @binding(2)
var<storage, read_write> aiBrickToBrixelMapping: array<i32>;

@group(0) @binding(3)
var<storage, read> aBBox: array<i32>;

@group(0) @binding(4)
var<storage, read_write> aBrixelDistances: array<f32>;

@group(0) @binding(5)
var<storage, read_write> aBrixelBarycentricCoords: array<vec3<f32>>;


struct UniformData
{
    mfPositionScale: f32, 
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    miFrameIndex: u32,
};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> aiCurrTriangleRange: array<u32>;

@group(1) @binding(2)
var<uniform> defaultUniformData: DefaultUniformData;


const iNumThreads = 256u;

/////
@compute
@workgroup_size(iNumThreads)
fn cs_main(
    @builtin(num_workgroups) numWorkGroups: vec3<u32>,
    @builtin(local_invocation_index) iLocalThreadIndex: u32,
    @builtin(workgroup_id) workGroup: vec3<u32>)
{
    
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

    let meshBBoxDiff: vec3<f32> = meshBBox1 - meshBBox0;

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

    let bboxDimension: vec3<f32> = bbox1 - bbox0;
    for(var iZ: i32 = 0; iZ < i32(bboxDimension.z); iZ++)
    {
        for(var iY: i32 = 0; iY < i32(bboxDimension.y); iY++)
        {
            for(var iX: i32 = 0; iX < i32(bboxDimension.x); iX++)
            {
                let brickPosition: vec3<f32> = bbox0 + vec3<f32>(
                    f32(iX) * uniformData.mfBrickDimension,
                    f32(iY) * uniformData.mfBrickDimension,
                    f32(iZ) * uniformData.mfBrickDimension
                );

                let iBrickIndex: i32 = iZ * i32(bboxDimension.y) * i32(bboxDimension.x) + iY * i32(bboxDimension.x) + iX;
                if(iLocalThreadIndex == 0u)
                {
                    aBricks[iBrickIndex].mPosition = brickPosition;
                }

                aBricks[iBrickIndex].miBrixelIndex = u32(iBrickIndex);

                aBrixels[iBrickIndex].maiBrixelDistances[iLocalThreadIndex * 2u] = 0u;
                aBrixels[iBrickIndex].maBrixelBarycentricCoord[iLocalThreadIndex * 2u] = vec3<f32>(0.0f, 0.0f, 0.0f);

                aBrixels[iBrickIndex].maiBrixelDistances[iLocalThreadIndex * 2u + 1u] = 0u;
                aBrixels[iBrickIndex].maBrixelBarycentricCoord[iLocalThreadIndex * 2u + 1u] = vec3<f32>(0.0f, 0.0f, 0.0f);
                
            }
        }
    }

    if(defaultUniformData.miFrame == 0)
    {
        for(var iLoop: u32 = 0; iLoop < 10u; iLoop++)
        {
            aiBrickToBrixelMapping[iLoop * iNumThreads + iLocalThreadIndex] = 0;
        }
    }

    for(var iLoop: u32 = 0u; iLoop < 1000u; iLoop++)
    {
        aBrixelDistances[iLoop * iNumThreads + iLocalThreadIndex * 2u] = 99999.0f;
        aBrixelDistances[iLoop * iNumThreads + iLocalThreadIndex * 2u + 1u] = 99999.0f;
        aBrixelBarycentricCoords[iLoop * iNumThreads + iLocalThreadIndex * 2u] = vec3<f32>(0.0f, 0.0f, 0.0f);
        aBrixelBarycentricCoords[iLoop * iNumThreads + iLocalThreadIndex * 2u + 1u] = vec3<f32>(0.0f, 0.0f, 0.0f);
    }
}