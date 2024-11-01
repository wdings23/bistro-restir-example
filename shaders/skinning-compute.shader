const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;

struct Vertex
{
    mPosition: vec4<f32>,
    mTexCoord: vec4<f32>,
    mNormal: vec4<f32>,
};

struct VertexWeight
{
    maJointWeights: vec4<f32>,
    maiJointIndices: vec4<u32>, 
};

struct VertexRange
{
    miStartVertex: u32,
    miEndVertex: u32,
};

struct UniformData
{
    mMeshTranslation: vec3<f32>,
    mfMeshScale: f32,
};

@group(0) @binding(0)
var<storage, read_write> aXFormVertices: array<Vertex>;

@group(0) @binding(1)
var<storage, read_write> aXFormPositions: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read_write> aBBox: array<atomic<i32>>;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> aVertices: array<Vertex>;

@group(1) @binding(2)
var<storage, read> aVertexWeights: array<VertexWeight>;

@group(1) @binding(3)
var<storage, read> aSkinMatrices: array<mat4x4<f32>>;

@group(1) @binding(4)
var<storage, read> aMeshVertexRanges: array<VertexRange>;

@group(1) @binding(5)
var<storage, read> aNormalMatrices: array<mat4x4<f32>>;

const iNumThreads: u32 = 256u;

@compute
@workgroup_size(iNumThreads)
fn cs_main(
    @builtin(num_workgroups) numWorkGroups: vec3<u32>,
    @builtin(local_invocation_index) iLocalIndex: u32,
    @builtin(workgroup_id) workGroup: vec3<u32>) 
{
    let iNumTotalThreads: u32 = numWorkGroups.x * iNumThreads;
    let iNumMeshes = aMeshVertexRanges[0].miEndVertex;

    let iThreadID: u32 = iLocalIndex + workGroup.x * iNumThreads;
    
    for(var iMesh: u32 = 0u; iMesh < iNumMeshes; iMesh++)
    {
        let iNumVertices: u32 = aMeshVertexRanges[iMesh + 1u].miEndVertex - aMeshVertexRanges[iMesh + 1u].miStartVertex;
        let iNumSteps: u32 = u32(ceil(f32(iNumVertices) / f32(iNumTotalThreads)));
        for(var i: u32 = 0; i < iNumSteps; i++)
        {
            let iVertex: u32 = aMeshVertexRanges[iMesh].miStartVertex + (i * iNumTotalThreads + iThreadID);
            if(iVertex >= iNumVertices)
            {
                break;
            }

            aXFormVertices[iVertex] = aVertices[iVertex];

            let vertexPosition: vec4<f32> = aXFormVertices[iVertex].mPosition;
            let iMatrix0: u32 = aVertexWeights[iVertex].maiJointIndices[0];
            let iMatrix1: u32 = aVertexWeights[iVertex].maiJointIndices[1];
            let iMatrix2: u32 = aVertexWeights[iVertex].maiJointIndices[2];
            let iMatrix3: u32 = aVertexWeights[iVertex].maiJointIndices[3];
            let xformPos0: vec4<f32> = (vertexPosition * aSkinMatrices[iMatrix0]) * aVertexWeights[iVertex].maJointWeights[0];
            let xformPos1: vec4<f32> = (vertexPosition * aSkinMatrices[iMatrix1]) * aVertexWeights[iVertex].maJointWeights[1];
            let xformPos2: vec4<f32> = (vertexPosition * aSkinMatrices[iMatrix2]) * aVertexWeights[iVertex].maJointWeights[2];
            let xformPos3: vec4<f32> = (vertexPosition * aSkinMatrices[iMatrix3]) * aVertexWeights[iVertex].maJointWeights[3]; 

            var normalMatrix0: mat4x4<f32> = aNormalMatrices[iMatrix0];
            var normalMatrix1: mat4x4<f32> = aNormalMatrices[iMatrix1];
            var normalMatrix2: mat4x4<f32> = aNormalMatrices[iMatrix2];
            var normalMatrix3: mat4x4<f32> = aNormalMatrices[iMatrix3];

            let vertexNormal: vec4<f32> = aXFormVertices[iVertex].mNormal;
            let xformNorm0 = (vertexNormal * normalMatrix0) * aVertexWeights[iVertex].maJointWeights[0];
            let xformNorm1 = (vertexNormal * normalMatrix1) * aVertexWeights[iVertex].maJointWeights[1];
            let xformNorm2 = (vertexNormal * normalMatrix2) * aVertexWeights[iVertex].maJointWeights[2];
            let xformNorm3 = (vertexNormal * normalMatrix3) * aVertexWeights[iVertex].maJointWeights[3];

            aXFormVertices[iVertex].mPosition = vec4<f32>(
                xformPos0.xyz +
                xformPos1.xyz +
                xformPos2.xyz +
                xformPos3.xyz,
                1.0f) * vec4<f32>(uniformData.mfMeshScale, uniformData.mfMeshScale, uniformData.mfMeshScale, 1.0f) + 
                vec4<f32>(uniformData.mMeshTranslation.x, uniformData.mMeshTranslation.y, uniformData.mMeshTranslation.z, 0.0f);


            let xformNormal: vec3<f32> = normalize(xformNorm0.xyz + xformNorm1.xyz + xformNorm2.xyz + xformNorm3.xyz);
            aXFormVertices[iVertex].mNormal = vec4<f32>(
                xformNormal,
                1.0f);

            aXFormPositions[iVertex] = aXFormVertices[iVertex].mPosition;
            let iX: i32 = i32(aXFormVertices[iVertex].mPosition.x * 1000.0f);
            let iY: i32 = i32(aXFormVertices[iVertex].mPosition.y * 1000.0f);
            let iZ: i32 = i32(aXFormVertices[iVertex].mPosition.z * 1000.0f);
            
            atomicMax(&aBBox[0], iX);
            atomicMax(&aBBox[1], iY);
            atomicMax(&aBBox[2], iZ);

            atomicMin(&aBBox[3], iX);
            atomicMin(&aBBox[4], iY);
            atomicMin(&aBBox[5], iZ);
        }
    }
}