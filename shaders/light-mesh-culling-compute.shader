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

struct DrawIndexParam
{
    miIndexCount: i32,
    miInstanceCount: i32,
    miFirstIndex: i32,
    miBaseVertex: i32,
    miFirstInstance: i32,
};

struct MeshBBox
{
    mMinPosition: vec4<f32>,
    mMaxPosition: vec4<f32>,
};

struct Range
{
    miStart: i32,
    miEnd: i32,
};

struct UniformData
{
    mLightViewMatrix: mat4x4<f32>,
    mLightProjectionMatrix: mat4x4<f32>,
    mLightViewProjectionMatrix: mat4x4<f32>,

    miNumMeshes: u32,
};

struct DebugInfo
{
    mXFormCenter: vec4<f32>,
    mLeftRightUpDown: vec4<f32>,  
    mNear: vec4<f32>,

    miNumIndices: u32,
    miIndexAddressOffset: u32,
    miIndirectDrawIndex: u32,
    miDraw: u32, 

};

@group(0) @binding(0)
var<storage, read_write> aDrawCalls: array<DrawIndexParam>;

@group(0) @binding(1)
var<storage, read_write> aNumDrawCalls: array<atomic<u32>>;

@group(0) @binding(2)
var<storage, read_write> aDebug: array<DebugInfo>;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> aMeshTriangleIndexRanges: array<Range>;

@group(1) @binding(2)
var<storage, read> aMeshBBox: array<MeshBBox>;

@group(1) @binding(3)
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
    let iMesh: u32 = iLocalThreadIndex + workGroup.x * iNumThreads;
    if(iMesh >= uniformData.miNumMeshes)
    {
        return;
    }

    let bInside: bool = cullBBox(
        aMeshBBox[iMesh].mMinPosition.xyz,
        aMeshBBox[iMesh].mMaxPosition.xyz,
        iMesh);    

    let center: vec3<f32> = (aMeshBBox[iMesh].mMaxPosition.xyz + aMeshBBox[iMesh].mMinPosition.xyz) * 0.5f;
    
    var xformCenter: vec4<f32> = vec4<f32>(center.xyz, 1.0f) * uniformData.mLightViewProjectionMatrix;
    xformCenter.x /= xformCenter.w;
    xformCenter.y /= xformCenter.w;
    xformCenter.z /= xformCenter.w;

    let iNumIndices: i32 = aMeshTriangleIndexRanges[iMesh].miEnd - aMeshTriangleIndexRanges[iMesh].miStart;
    let iIndexAddressOffset: i32 = aMeshTriangleIndexRanges[iMesh].miStart;
    var iDrawCommandIndex: u32 = 0u;
    if(bInside)
    {
        iDrawCommandIndex = atomicAdd(&aNumDrawCalls[0], 1u);

        aDrawCalls[iDrawCommandIndex].miIndexCount = iNumIndices;
        aDrawCalls[iDrawCommandIndex].miInstanceCount = 1;
        aDrawCalls[iDrawCommandIndex].miFirstIndex = iIndexAddressOffset;
        aDrawCalls[iDrawCommandIndex].miBaseVertex = 0;
        aDrawCalls[iDrawCommandIndex].miFirstInstance = 0;

        aDebug[iDrawCommandIndex].mXFormCenter = xformCenter;
    }

    atomicAdd(&aNumDrawCalls[1], 1u);
}

/////
fn sortDrawCalls()
{
    let iNumDrawCalls: u32 = aNumDrawCalls[0];
    for(var i: u32 = 0u; i < iNumDrawCalls; i++)
    {
        for(var j: u32 = i + 1; j < iNumDrawCalls; j++)
        {
            if(aDebug[i].mXFormCenter.z > aDebug[i].mXFormCenter.z)
            {
                let drawCall: DrawIndexParam = aDrawCalls[i];
                aDrawCalls[i] = aDrawCalls[j];
                aDrawCalls[j] = drawCall;

                let temp: vec4<f32> = aDebug[i].mXFormCenter;
                aDebug[i].mXFormCenter = aDebug[j].mXFormCenter;
                aDebug[j].mXFormCenter = temp;
            }
        }
    }
}


/////
fn getFrustumPlane(
    iColumn: u32,
    fMult: f32) -> vec4<f32>
{
    var plane: vec3<f32> = vec3<f32>(
        uniformData.mLightViewProjectionMatrix[iColumn][0] * fMult + uniformData.mLightViewProjectionMatrix[3][0],
        uniformData.mLightViewProjectionMatrix[iColumn][1] * fMult + uniformData.mLightViewProjectionMatrix[3][1],
        uniformData.mLightViewProjectionMatrix[iColumn][2] * fMult + uniformData.mLightViewProjectionMatrix[3][2]);
    let fPlaneW: f32 = uniformData.mLightViewProjectionMatrix[iColumn][3] * fMult + uniformData.mLightViewProjectionMatrix[3][3];
    let fLength: f32 = length(plane);
    plane = normalize(plane);
    
    let ret: vec4<f32> = vec4<f32>(
        plane.xyz,
        fPlaneW / fLength
    );
    
    return ret;
}

//////
fn cullBBox(
    minPosition: vec3<f32>,
    maxPosition: vec3<f32>,
    iMesh: u32) -> bool
{
    // frustum planes
    let leftPlane: vec4<f32> = getFrustumPlane(0u, 1.0f);
    let rightPlane: vec4<f32> = getFrustumPlane(0u, -1.0f);
    let bottomPlane: vec4<f32> = getFrustumPlane(1u, 1.0f);
    let upperPlane: vec4<f32> = getFrustumPlane(1u, -1.0f);
    let nearPlane: vec4<f32> = getFrustumPlane(2u, 1.0f);
    let farPlane: vec4<f32> = getFrustumPlane(2u, -1.0f);

    let v0: vec3<f32> = vec3<f32>(minPosition.x, minPosition.y, minPosition.z);
    let v1: vec3<f32> = vec3<f32>(maxPosition.x, minPosition.y, minPosition.z);
    let v2: vec3<f32> = vec3<f32>(minPosition.x, minPosition.y, maxPosition.z);
    let v3: vec3<f32> = vec3<f32>(maxPosition.x, minPosition.y, maxPosition.z);

    let v4: vec3<f32> = vec3<f32>(minPosition.x, maxPosition.y, minPosition.z);
    let v5: vec3<f32> = vec3<f32>(maxPosition.x, maxPosition.y, minPosition.z);
    let v6: vec3<f32> = vec3<f32>(minPosition.x, maxPosition.y, maxPosition.z);
    let v7: vec3<f32> = vec3<f32>(maxPosition.x, maxPosition.y, maxPosition.z);

    var fCount0: f32 = 0.0f;
    fCount0 += sign(dot(leftPlane.xyz, v0) + leftPlane.w);
    fCount0 += sign(dot(leftPlane.xyz, v1) + leftPlane.w);
    fCount0 += sign(dot(leftPlane.xyz, v2) + leftPlane.w);
    fCount0 += sign(dot(leftPlane.xyz, v3) + leftPlane.w);
    fCount0 += sign(dot(leftPlane.xyz, v4) + leftPlane.w);
    fCount0 += sign(dot(leftPlane.xyz, v5) + leftPlane.w);
    fCount0 += sign(dot(leftPlane.xyz, v6) + leftPlane.w);
    fCount0 += sign(dot(leftPlane.xyz, v7) + leftPlane.w);

    var fCount1: f32 = 0.0f;
    fCount1 += sign(dot(rightPlane.xyz, v0) + rightPlane.w);
    fCount1 += sign(dot(rightPlane.xyz, v1) + rightPlane.w);
    fCount1 += sign(dot(rightPlane.xyz, v2) + rightPlane.w);
    fCount1 += sign(dot(rightPlane.xyz, v3) + rightPlane.w);
    fCount1 += sign(dot(rightPlane.xyz, v4) + rightPlane.w);
    fCount1 += sign(dot(rightPlane.xyz, v5) + rightPlane.w);
    fCount1 += sign(dot(rightPlane.xyz, v6) + rightPlane.w);
    fCount1 += sign(dot(rightPlane.xyz, v7) + rightPlane.w);

    var fCount2: f32 = 0.0f;
    fCount2 += sign(dot(upperPlane.xyz, v0) + upperPlane.w);
    fCount2 += sign(dot(upperPlane.xyz, v1) + upperPlane.w);
    fCount2 += sign(dot(upperPlane.xyz, v2) + upperPlane.w);
    fCount2 += sign(dot(upperPlane.xyz, v3) + upperPlane.w);
    fCount2 += sign(dot(upperPlane.xyz, v4) + upperPlane.w);
    fCount2 += sign(dot(upperPlane.xyz, v5) + upperPlane.w);
    fCount2 += sign(dot(upperPlane.xyz, v6) + upperPlane.w);
    fCount2 += sign(dot(upperPlane.xyz, v7) + upperPlane.w);

    var fCount3: f32 = 0.0f;
    fCount3 += sign(dot(bottomPlane.xyz, v0) + bottomPlane.w);
    fCount3 += sign(dot(bottomPlane.xyz, v1) + bottomPlane.w);
    fCount3 += sign(dot(bottomPlane.xyz, v2) + bottomPlane.w);
    fCount3 += sign(dot(bottomPlane.xyz, v3) + bottomPlane.w);
    fCount3 += sign(dot(bottomPlane.xyz, v4) + bottomPlane.w);
    fCount3 += sign(dot(bottomPlane.xyz, v5) + bottomPlane.w);
    fCount3 += sign(dot(bottomPlane.xyz, v6) + bottomPlane.w);
    fCount3 += sign(dot(bottomPlane.xyz, v7) + bottomPlane.w);

    var fCount4: f32 = 0.0f;
    fCount4 += sign(dot(nearPlane.xyz, v0) + nearPlane.w);
    fCount4 += sign(dot(nearPlane.xyz, v1) + nearPlane.w);
    fCount4 += sign(dot(nearPlane.xyz, v2) + nearPlane.w);
    fCount4 += sign(dot(nearPlane.xyz, v3) + nearPlane.w);
    fCount4 += sign(dot(nearPlane.xyz, v4) + nearPlane.w);
    fCount4 += sign(dot(nearPlane.xyz, v5) + nearPlane.w);
    fCount4 += sign(dot(nearPlane.xyz, v6) + nearPlane.w);
    fCount4 += sign(dot(nearPlane.xyz, v7) + nearPlane.w);

    aDebug[iMesh].mLeftRightUpDown = vec4<f32>(fCount0, fCount1, fCount2, fCount3);
    aDebug[iMesh].mNear = vec4<f32>(fCount4, 0.0f, 0.0f, 0.0f);

    return (fCount0 > -8.0f && fCount1 > -8.0f && fCount2 > -8.0f && fCount3 > -8.0f && fCount4 > -8.0f);  
}