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
var<storage, read_write> aiVisibleMeshID: array<u32>;

@group(0) @binding(3)
var<storage, read_write> aDebug: array<DebugInfo>;

@group(0) @binding(4)
var depthTexture0: texture_2d<f32>;

@group(0) @binding(5)
var depthTexture1: texture_2d<f32>;

@group(0) @binding(6)
var depthTexture2: texture_2d<f32>;

@group(0) @binding(7)
var depthTexture3: texture_2d<f32>;

@group(0) @binding(8)
var depthTexture4: texture_2d<f32>;

@group(0) @binding(9)
var depthTexture5: texture_2d<f32>;

@group(0) @binding(10)
var depthTexture6: texture_2d<f32>;

@group(0) @binding(11)
var depthTexture7: texture_2d<f32>;

@group(0) @binding(12)
var depthTexture8: texture_2d<f32>;

@group(0) @binding(13)
var depthTexture9: texture_2d<f32>;

@group(0) @binding(14)
var textureSampler: sampler;

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

    // already drawn, just want to check mesh not drawn already
    let iVisible: u32 = aiVisibleMeshID[iMesh];
    if(iVisible > 0u)
    {
        atomicAdd(&aNumDrawCalls[3], 1u);
        return;
    }

    // check bbox against the updated depth pyramid textures from the first culling pass
    let bOccluded: bool = cullBBoxDepth(
        aMeshBBox[iMesh].mMinPosition.xyz,
        aMeshBBox[iMesh].mMaxPosition.xyz,
        iMesh
    );

    // check inside of frustum
    let bInside: bool = cullBBox(
        aMeshBBox[iMesh].mMinPosition.xyz,
        aMeshBBox[iMesh].mMaxPosition.xyz,
        iMesh);    

    let center: vec3<f32> = (aMeshBBox[iMesh].mMaxPosition.xyz + aMeshBBox[iMesh].mMinPosition.xyz) * 0.5f;
    var xformCenter: vec4<f32> = vec4<f32>(center.xyz, 1.0f) * defaultUniformData.mViewProjectionMatrix;
    xformCenter.x /= xformCenter.w;
    xformCenter.y /= xformCenter.w;
    xformCenter.z /= xformCenter.w;

    aiVisibleMeshID[iMesh] = 0u;

    let iNumIndices: i32 = aMeshTriangleIndexRanges[iMesh].miEnd - aMeshTriangleIndexRanges[iMesh].miStart;
    let iIndexAddressOffset: i32 = aMeshTriangleIndexRanges[iMesh].miStart;
    var iDrawCommandIndex: u32 = 0u;
    if(bInside && !bOccluded)
    {
        // update draw count
        atomicAdd(&aNumDrawCalls[0], 1u);
        iDrawCommandIndex = atomicAdd(&aNumDrawCalls[2], 1u);

        aDrawCalls[iDrawCommandIndex].miIndexCount = iNumIndices;
        aDrawCalls[iDrawCommandIndex].miInstanceCount = 1;
        aDrawCalls[iDrawCommandIndex].miFirstIndex = iIndexAddressOffset;
        aDrawCalls[iDrawCommandIndex].miBaseVertex = 0;
        aDrawCalls[iDrawCommandIndex].miFirstInstance = 0;

        aDebug[iDrawCommandIndex].mXFormCenter = xformCenter;

        // mark as visible
        aiVisibleMeshID[iMesh] = 1u;
    }

    atomicAdd(&aNumDrawCalls[3], 1u);
}

/////
fn getFrustumPlane(
    iColumn: u32,
    fMult: f32) -> vec4<f32>
{
    var plane: vec3<f32> = vec3<f32>(
        defaultUniformData.mViewProjectionMatrix[iColumn][0] * fMult + defaultUniformData.mViewProjectionMatrix[3][0],
        defaultUniformData.mViewProjectionMatrix[iColumn][1] * fMult + defaultUniformData.mViewProjectionMatrix[3][1],
        defaultUniformData.mViewProjectionMatrix[iColumn][2] * fMult + defaultUniformData.mViewProjectionMatrix[3][2]);
    let fPlaneW: f32 = defaultUniformData.mViewProjectionMatrix[iColumn][3] * fMult + defaultUniformData.mViewProjectionMatrix[3][3];
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

//////
fn cullBBoxDepth(
    minPosition: vec3<f32>,
    maxPosition: vec3<f32>,
    iMesh: u32) -> bool
{
    let v0: vec3<f32> = vec3<f32>(minPosition.x, minPosition.y, minPosition.z);
    let v1: vec3<f32> = vec3<f32>(maxPosition.x, minPosition.y, minPosition.z);
    let v2: vec3<f32> = vec3<f32>(minPosition.x, minPosition.y, maxPosition.z);
    let v3: vec3<f32> = vec3<f32>(maxPosition.x, minPosition.y, maxPosition.z);

    let v4: vec3<f32> = vec3<f32>(minPosition.x, maxPosition.y, minPosition.z);
    let v5: vec3<f32> = vec3<f32>(maxPosition.x, maxPosition.y, minPosition.z);
    let v6: vec3<f32> = vec3<f32>(minPosition.x, maxPosition.y, maxPosition.z);
    let v7: vec3<f32> = vec3<f32>(maxPosition.x, maxPosition.y, maxPosition.z);

    let xform0: vec4<f32> = vec4<f32>(v0.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
    let xform1: vec4<f32> = vec4<f32>(v1.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
    let xform2: vec4<f32> = vec4<f32>(v2.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
    let xform3: vec4<f32> = vec4<f32>(v3.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;

    let xform4: vec4<f32> = vec4<f32>(v4.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
    let xform5: vec4<f32> = vec4<f32>(v5.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
    let xform6: vec4<f32> = vec4<f32>(v6.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
    let xform7: vec4<f32> = vec4<f32>(v7.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;

    let xyz0: vec3<f32> = xform0.xyz / xform0.w;
    let xyz1: vec3<f32> = xform1.xyz / xform1.w;
    let xyz2: vec3<f32> = xform2.xyz / xform2.w;
    let xyz3: vec3<f32> = xform3.xyz / xform3.w;

    let xyz4: vec3<f32> = xform4.xyz / xform4.w;
    let xyz5: vec3<f32> = xform5.xyz / xform5.w;
    let xyz6: vec3<f32> = xform6.xyz / xform6.w;
    let xyz7: vec3<f32> = xform7.xyz / xform7.w;

    //let xyz0: vec3<f32> = vec3<f32>(xform0.xy / xform0.w, xform0.z);
    //let xyz1: vec3<f32> = vec3<f32>(xform1.xy / xform1.w, xform1.z);
    //let xyz2: vec3<f32> = vec3<f32>(xform2.xy / xform2.w, xform2.z);
    //let xyz3: vec3<f32> = vec3<f32>(xform3.xy / xform3.w, xform3.z);
//
    //let xyz4: vec3<f32> = vec3<f32>(xform4.xy / xform4.w, xform4.z);
    //let xyz5: vec3<f32> = vec3<f32>(xform5.xy / xform5.w, xform5.z);
    //let xyz6: vec3<f32> = vec3<f32>(xform6.xy / xform6.w, xform6.z);
    //let xyz7: vec3<f32> = vec3<f32>(xform7.xy / xform7.w, xform7.z);
    
    var minXYZ: vec3<f32> = min(min(min(min(min(min(min(xyz0, xyz1), xyz2), xyz3), xyz4), xyz5), xyz6), xyz7);
    var maxXYZ: vec3<f32> = max(max(max(max(max(max(max(xyz0, xyz1), xyz2), xyz3), xyz4), xyz5), xyz6), xyz7);
    minXYZ = vec3<f32>(minXYZ.x * 0.5f + 0.5f, minXYZ.y * 0.5f + 0.5f, minXYZ.z);
    maxXYZ = vec3<f32>(maxXYZ.x * 0.5f + 0.5f, maxXYZ.y * 0.5f + 0.5f, maxXYZ.z);
    if(maxXYZ.z > 1.0f)
    {
        minXYZ.z = 0.0f;
    }

    let fMinDepth: f32 = minXYZ.z;

    let fAspectRatio: f32 = f32(defaultUniformData.miScreenHeight) / f32(defaultUniformData.miScreenWidth);
    let diffXYZ: vec2<f32> = abs(maxXYZ.xy - minXYZ.xy);
    let fMaxComp: f32 = max(diffXYZ.x, diffXYZ.y * fAspectRatio) * 512.0f;    // compute LOD from 1 to 8
    let iLOD: u32 =  min(u32(log2(fMaxComp)), 8u);
    var uv: vec2<f32> = vec2<f32>(maxXYZ.xy + minXYZ.xy) * 0.5f;
    uv.y = 1.0f - uv.y;
    var fSampleDepth: f32 = 0.0f;
    if(iLOD == 0u)
    {
        fSampleDepth = textureSampleLevel(
            depthTexture0,
            textureSampler,
            uv,
            0.0f
        ).x;
    }
    else if(iLOD == 1u)
    {
        fSampleDepth = textureSampleLevel(
            depthTexture1,
            textureSampler,
            uv,
            0.0f
        ).x;
    }
    else if(iLOD == 2u)
    {
        fSampleDepth = textureSampleLevel(
            depthTexture2,
            textureSampler,
            uv,
            0.0f
        ).x;
    }
    else if(iLOD == 3u)
    {
        fSampleDepth = textureSampleLevel(
            depthTexture3,
            textureSampler,
            uv,
            0.0f
        ).x;
    }
    else if(iLOD == 4u)
    {
        fSampleDepth = textureSampleLevel(
            depthTexture4,
            textureSampler,
            uv,
            0.0f
        ).x;
    }
    else if(iLOD == 5u)
    {
        fSampleDepth = textureSampleLevel(
            depthTexture5,
            textureSampler,
            uv,
            0.0f
        ).x;
    }
    else if(iLOD == 6u)
    {
        fSampleDepth = textureSampleLevel(
            depthTexture6,
            textureSampler,
            uv,
            0.0f
        ).x;
    }
    else if(iLOD == 7u)
    {
        fSampleDepth = textureSampleLevel(
            depthTexture7,
            textureSampler,
            uv,
            0.0f
        ).x;
    }
    else if(iLOD == 8u)
    {
        fSampleDepth = textureSampleLevel(
            depthTexture8,
            textureSampler,
            uv,
            0.0f
        ).x;
    }
    else
    {
        fSampleDepth = textureSampleLevel(
            depthTexture9,
            textureSampler,
            uv,
            0.0f
        ).x;
    }


    return (fMinDepth > fSampleDepth);
    
}