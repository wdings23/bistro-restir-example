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

struct VertexRange
{
    miStartVertex: u32,
    miEndVertex: u32,
};

struct BrickRange
{
    miStartBrickRange: u32,
    miEndBrickRange: u32,
};

@group(0) @binding(0)
var<storage, read_write> aBricks: array<BrickInfo>;

@group(0) @binding(1)
var<storage, read_write> aBrixels: array<BrixelInfo>;

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
var<storage, read> aMeshBrickStartPositions: array<vec3<f32>>;

@group(1) @binding(2)
var<storage, read> aMeshBBoxDimensions: array<vec3<f32>>;

@group(1) @binding(3)
var<storage, read> aBrickRanges: array<BrickRange>;


const iNumThreads = 256u;

/////
@compute
@workgroup_size(iNumThreads)
fn cs_main(
    @builtin(num_workgroups) numWorkGroups: vec3<u32>,
    @builtin(local_invocation_index) iLocalThreadIndex: u32,
    @builtin(workgroup_id) workGroup: vec3<u32>)
{
    let iMesh: u32 = workGroup.x;

    let iStartBrickRange: i32 = i32(aBrickRanges[iMesh].miStartBrickRange);
    let bboxDimension: vec3<f32> = aMeshBBoxDimensions[iMesh];
    for(var iZ: i32 = 0; iZ < i32(bboxDimension.z); iZ++)
    {
        for(var iY: i32 = 0; iY < i32(bboxDimension.y); iY++)
        {
            for(var iX: i32 = 0; iX < i32(bboxDimension.x); iX++)
            {
                let brickPosition: vec3<f32> = aMeshBrickStartPositions[iMesh] + vec3<f32>(
                    f32(iX) * uniformData.mfBrickDimension,
                    f32(iY) * uniformData.mfBrickDimension,
                    f32(iZ) * uniformData.mfBrickDimension
                );

                let iBrickIndex: i32 = iStartBrickRange + iZ * i32(bboxDimension.y) * i32(bboxDimension.x) + iY * i32(bboxDimension.x) + iX;
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

}