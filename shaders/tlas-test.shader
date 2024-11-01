const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const PROBE_IMAGE_SIZE: u32 = 8u;
const VALIDATION_STEP: u32 = 16u;

struct RandomResult {
    mfNum: f32,
    miSeed: u32,
};

struct Ray
{
    mOrigin: vec4<f32>,
    mDirection: vec4<f32>,
    mfT: vec4<f32>,
};

struct Tri
{
    miV0 : u32,
    miV1 : u32,
    miV2 : u32,
    mPadding : u32,

    mCentroid : vec4<f32>
};

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @location(2) normal : vec4<f32>
};
struct VertexOutput {
    @location(0) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) normal: vec4<f32>
};
struct FragmentOutput {
    @location(0) radianceOutput : vec4<f32>,
};

struct Vertex
{
    mPosition: vec4<f32>,
    mTexCoord: vec4<f32>,
    mNormal: vec4<f32>,
};

struct ReservoirResult
{
    mReservoir: vec4<f32>,
    mbExchanged: bool
};

struct IntersectBVHResult
{
    mHitPosition: vec3<f32>,
    mHitNormal: vec3<f32>,
    mBarycentricCoordinate: vec3<f32>,
    miHitTriangle: u32,
    miMeshID: u32,
};

struct RayTriangleIntersectionResult
{
    mIntersectPosition: vec3<f32>,
    mIntersectNormal: vec3<f32>,
    mBarycentricCoordinate: vec3<f32>,
};

struct DisocclusionResult
{
    mBackProjectScreenCoord: vec2<i32>,
    mbDisoccluded: bool,
};

struct DirectSunLightResult
{
    mReservoir: vec4<f32>,
    mRadiance: vec3<f32>,
    mRandomResult: RandomResult,
};

struct IrradianceCacheEntry
{
    mPosition:              vec4<f32>,
    mImageProbe:            array<vec4<f32>, 64>,            // 8x8 image probe
};

struct BVNodeIndex
{
    miStartNodeIndex: u32,
    miEndNodeIndex: u32,
    miBufferIndex: u32,
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
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var texCoordTexture: texture_2d<f32>;

@group(0) @binding(3)
var skyTexture: texture_2d<f32>;

@group(0) @binding(4)
var textureSampler: sampler;

struct UniformData
{  
    miPartitionNodeIndex: u32,
    miPartitionMeshIndex: u32,
};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

struct BVHNode2
{
    mMinBound: vec4<f32>,
    mMaxBound: vec4<f32>,
    mCentroid: vec4<f32>,
    
    miChildren0: u32,
    miChildren1: u32,
    miPrimitiveID: u32,
    miMeshID: u32,
};

@group(1) @binding(1)
var<storage, read> aTLASBVHNodes: array<BVHNode2>;

@group(1) @binding(2)
var<storage, read> aBLASBVHNodes0: array<BVHNode2>;

@group(1) @binding(3)
var<storage, read> aBLASBVHNodes1: array<BVHNode2>;

@group(1) @binding(4)
var<storage, read> aMeshBLASNodeIndices: array<BVNodeIndex>;

@group(1) @binding(5)
var<storage, read> aVertexBuffer: array<Vertex>;

@group(1) @binding(6)
var<storage, read> aiIndexBuffer: array<u32>;

@group(1) @binding(7)
var blueNoiseTexture: texture_2d<f32>;

@group(1) @binding(8)
var<uniform> defaultUniformData: DefaultUniformData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput 
{
    var out: VertexOutput;
    out.pos = in.pos;
    out.texCoord = in.texCoord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput 
{
    var fReservoirSize: f32 = 8.0f;

    var out: FragmentOutput;

    if(defaultUniformData.miFrame < 1)
    {
        return out;
    }

    var randomResult: RandomResult = initRand(
        u32(in.texCoord.x * 100.0f + in.texCoord.y * 200.0f) + u32(defaultUniformData.mfRand0 * 100.0f),
        u32(in.pos.x * 10.0f + in.pos.z * 20.0f) + u32(defaultUniformData.mfRand0 * 100.0f),
        10u);

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture, 
        textureSampler, 
        in.texCoord);

    let normal: vec3<f32> = textureSample(
        normalTexture, 
        textureSampler, 
        in.texCoord).xyz;

    if(worldPosition.w <= 0.0f)
    {
        out.radianceOutput = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        return out;
    }

    var blueNoise: vec3<f32> = textureSample(
        blueNoiseTexture,
        textureSampler,
        in.texCoord.xy
    ).xyz;

    var ray: Ray;
    ray.mOrigin = vec4<f32>(worldPosition.xyz + defaultUniformData.mLightDirection.xyz * 0.05f, 1.0f);
    ray.mDirection = vec4<f32>(defaultUniformData.mLightDirection.xyz, 1.0f);

    var intersectionInfo: IntersectBVHResult = intersectTotalBVH4(ray, 0u);
    var iMeshID: u32 = intersectionInfo.miMeshID;

    if(iMeshID == UINT32_MAX)
    {
        iMeshID = 0u;
    }
    let fMeshID: f32 = f32(iMeshID);

    out.radianceOutput = vec4<f32>(1.0f, 1.0f, 1.0f, fMeshID);
    if(intersectionInfo.miHitTriangle != UINT32_MAX)
    {
        out.radianceOutput = vec4<f32>(0.0f, 0.0f, 0.0f, fMeshID);
    }
    
    return out;
}

/////
fn initRand(
    val0: u32, 
    val1: u32, 
    backoff: u32) -> RandomResult
{
    var retResult: RandomResult;

    var v0: u32 = val0;
    var v1: u32 = val1;
    var s0: u32 = 0u;

    for(var n: u32 = 0; n < backoff; n++)
    {
        s0 += u32(0x9e3779b9);
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    retResult.miSeed = v0;
    retResult.mfNum = 0.0f;

    return retResult;
}

/////
fn nextRand(s: u32) -> RandomResult
{
    var retResult: RandomResult;

    var sCopy: u32 = s;
    sCopy = (1664525u * sCopy + 1013904223u);
    retResult.mfNum = f32(sCopy & 0x00FFFFFF) / f32(0x01000000);
    retResult.miSeed = sCopy;

    return retResult;
}

/////
fn uniformSampling(
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    fRand0: f32,
    fRand1: f32) -> Ray
{
    let fPhi: f32 = 2.0f * PI * fRand0;
    let fCosTheta: f32 = 1.0f - fRand1;
    let fSinTheta: f32 = sqrt(1.0f - fCosTheta * fCosTheta);
    let h: vec3<f32> = vec3<f32>(
        cos(fPhi) * fSinTheta,
        sin(fPhi) * fSinTheta,
        fCosTheta);

    var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    if(abs(normal.y) > 0.999f)
    {
        up = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    let tangent: vec3<f32> = normalize(cross(up, normal));
    let binormal: vec3<f32> = normalize(cross(normal, tangent));
    let rayDirection: vec3<f32> = normalize(tangent * h.x + binormal * h.y + normal * h.z);

    var ray: Ray;
    ray.mOrigin = vec4<f32>(worldPosition, 1.0f);
    ray.mDirection = vec4<f32>(rayDirection, 1.0f);
    ray.mfT = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

    return ray;
}

/////
fn octahedronMap2(direction: vec3<f32>) -> vec2<f32>
{
    let fDP: f32 = dot(vec3<f32>(1.0f, 1.0f, 1.0f), abs(direction));
    let newDirection: vec3<f32> = vec3<f32>(direction.x, direction.z, direction.y) / fDP;

    var ret: vec2<f32> =
        vec2<f32>(
            (1.0f - abs(newDirection.z)) * sign(newDirection.x),
            (1.0f - abs(newDirection.x)) * sign(newDirection.z));
       
    if(newDirection.y < 0.0f)
    {
        ret = vec2<f32>(
            newDirection.x, 
            newDirection.z);
    }

    ret = ret * 0.5f + vec2<f32>(0.5f, 0.5f);
    ret.y = 1.0f - ret.y;

    return ret;
}

/////
fn barycentric(
    p: vec3<f32>, 
    a: vec3<f32>, 
    b: vec3<f32>, 
    c: vec3<f32>) -> vec3<f32>
{
    let v0: vec3<f32> = b - a;
    let v1: vec3<f32> = c - a;
    let v2: vec3<f32> = p - a;
    let fD00: f32 = dot(v0, v0);
    let fD01: f32 = dot(v0, v1);
    let fD11: f32 = dot(v1, v1);
    let fD20: f32 = dot(v2, v0);
    let fD21: f32 = dot(v2, v1);
    let fOneOverDenom: f32 = 1.0f / (fD00 * fD11 - fD01 * fD01);
    let fV: f32 = (fD11 * fD20 - fD01 * fD21) * fOneOverDenom;
    let fW: f32 = (fD00 * fD21 - fD01 * fD20) * fOneOverDenom;
    let fU: f32 = 1.0f - fV - fW;

    return vec3<f32>(fU, fV, fW);
}


/////
fn rayPlaneIntersection(
    pt0: vec3<f32>,
    pt1: vec3<f32>,
    planeNormal: vec3<f32>,
    fPlaneDistance: f32) -> f32
{
    var fRet: f32 = FLT_MAX;
    let v: vec3<f32> = pt1 - pt0;

    let fDenom: f32 = dot(v, planeNormal);
    fRet = -(dot(pt0, planeNormal) + fPlaneDistance) / (fDenom + 1.0e-5f);

    return fRet;
}

/////
fn rayBoxIntersect(
    rayPosition: vec3<f32>,
    rayDir: vec3<f32>,
    bboxMin: vec3<f32>,
    bboxMax: vec3<f32>) -> bool
{
    //let oneOverRay: vec3<f32> = 1.0f / rayDir.xyz;
    let tMin: vec3<f32> = (bboxMin - rayPosition) / rayDir.xyz;
    let tMax: vec3<f32> = (bboxMax - rayPosition) / rayDir.xyz;

    var fTMin: f32 = min(tMin.x, tMax.x);
    var fTMax: f32 = max(tMin.x, tMax.x);

    fTMin = max(fTMin, min(tMin.y, tMax.y));
    fTMax = min(fTMax, max(tMin.y, tMax.y));

    fTMin = max(fTMin, min(tMin.z, tMax.z));
    fTMax = min(fTMax, max(tMin.z, tMax.z));

    return fTMax >= fTMin;
}



/////
fn rayTriangleIntersection(
    rayPt0: vec3<f32>, 
    rayPt1: vec3<f32>, 
    triPt0: vec3<f32>, 
    triPt1: vec3<f32>, 
    triPt2: vec3<f32>) -> RayTriangleIntersectionResult
{
    var ret: RayTriangleIntersectionResult;

    let v0: vec3<f32> = normalize(triPt1 - triPt0);
    let v1: vec3<f32> = normalize(triPt2 - triPt0);
    let cp: vec3<f32> = cross(v0, v1);

    let triNormal: vec3<f32> = normalize(cp);
    let fPlaneDistance: f32 = -dot(triPt0, triNormal);

    let fT: f32 = rayPlaneIntersection(
        rayPt0, 
        rayPt1, 
        triNormal, 
        fPlaneDistance);
    if(fT < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }

    let collisionPt: vec3<f32> = rayPt0 + (rayPt1 - rayPt0) * fT;
    
    let edge0: vec3<f32> = normalize(triPt1 - triPt0);
    let edge1: vec3<f32> = normalize(triPt2 - triPt0);
    let edge2: vec3<f32> = normalize(triPt0 - triPt2);

    // edge 0
    var C: vec3<f32> = cross(edge0, normalize(collisionPt - triPt0));
    if(dot(triNormal, C) < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }

    // edge 1
    C = cross(edge1, normalize(collisionPt - triPt1));
    if(dot(triNormal, C) < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }

    // edge 2
    C = cross(edge2, normalize(collisionPt - triPt2));
    if(dot(triNormal, C) < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }
    
    ret.mBarycentricCoordinate = barycentric(collisionPt, triPt0, triPt1, triPt2);
    ret.mIntersectPosition = (triPt0 * ret.mBarycentricCoordinate.x + triPt1 * ret.mBarycentricCoordinate.y + triPt2 * ret.mBarycentricCoordinate.z);
    ret.mIntersectNormal = triNormal.xyz;

    return ret;
}

struct Triangle
{
    miV0 : u32,
    miV1 : u32,
    miV2 : u32,
    mPadding : u32,

    mCentroid : vec4<f32>
};

struct LeafNode
{
    miTriangleIndex: u32,
};

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

struct BVHProcessInfo
{
    miStep: u32,
    miStartNodeIndex: u32,
    miEndNodeIndex: u32,
    miNumMeshes: u32, 
};

/////
fn intersectTri4(
    ray: Ray,
    iTriangleIndex: u32) -> RayTriangleIntersectionResult
{
    let pos0: vec4<f32> = aVertexBuffer[aiIndexBuffer[iTriangleIndex * 3]].mPosition;
    let pos1: vec4<f32> = aVertexBuffer[aiIndexBuffer[iTriangleIndex * 3 + 1]].mPosition;
    let pos2: vec4<f32> = aVertexBuffer[aiIndexBuffer[iTriangleIndex * 3 + 2]].mPosition;

    var iIntersected: u32 = 0;
    var fT: f32 = FLT_MAX;
    let intersectionInfo: RayTriangleIntersectionResult = rayTriangleIntersection(
        ray.mOrigin.xyz,
        ray.mOrigin.xyz + ray.mDirection.xyz * 1000.0f,
        pos0.xyz,
        pos1.xyz,
        pos2.xyz);
    
    return intersectionInfo;
}

/////
fn intersectBLASBVH4Part0(
    ray: Ray,
    iStartNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 16>;
    aiStack[iStackTop] = iStartNodeIndex;

    ret.mHitPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    ret.miHitTriangle = UINT32_MAX;
    var fClosestDistance: f32 = FLT_MAX;

    for(var iStep: u32 = 0u; iStep < 10000u; iStep++)
    {
        if(iStackTop < 0)
        {
            break;
        }

        let iNodeIndex: u32 = aiStack[iStackTop];
        iStackTop -= 1;

        if(aBLASBVHNodes0[iNodeIndex].miPrimitiveID != UINT32_MAX)
        {
            let intersectionInfo: RayTriangleIntersectionResult = intersectTri4(
                ray,
                aBLASBVHNodes0[iNodeIndex].miPrimitiveID);

            if(abs(intersectionInfo.mIntersectPosition.x) < 10000.0f)
            {
                let fDistanceToEye: f32 = length(intersectionInfo.mIntersectPosition.xyz - ray.mOrigin.xyz);
                {
                    ret.mHitPosition = intersectionInfo.mIntersectPosition.xyz;
                    ret.mHitNormal = intersectionInfo.mIntersectNormal.xyz;
                    ret.miHitTriangle = aBLASBVHNodes0[iNodeIndex].miPrimitiveID;
                    ret.mBarycentricCoordinate = intersectionInfo.mBarycentricCoordinate;
                    ret.miMeshID = aBLASBVHNodes0[iNodeIndex].miMeshID;

                    break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aBLASBVHNodes0[iNodeIndex].mMinBound.xyz,
                aBLASBVHNodes0[iNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aBLASBVHNodes0[iNodeIndex].miChildren0 + iStartNodeIndex;
                iStackTop += 1;
                aiStack[iStackTop] = aBLASBVHNodes0[iNodeIndex].miChildren1 + iStartNodeIndex;
            }
        }
    }

    return ret;
}


/////
fn intersectBLASBVH4Part1(
    ray: Ray,
    iStartNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 16>;
    aiStack[iStackTop] = iStartNodeIndex;

    ret.mHitPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    ret.miHitTriangle = UINT32_MAX;
    var fClosestDistance: f32 = FLT_MAX;

    for(var iStep: u32 = 0u; iStep < 10000u; iStep++)
    {
        if(iStackTop < 0)
        {
            break;
        }

        let iNodeIndex: u32 = aiStack[iStackTop];
        iStackTop -= 1;

        if(aBLASBVHNodes1[iNodeIndex].miPrimitiveID != UINT32_MAX)
        {
            let intersectionInfo: RayTriangleIntersectionResult = intersectTri4(
                ray,
                aBLASBVHNodes1[iNodeIndex].miPrimitiveID);

            if(abs(intersectionInfo.mIntersectPosition.x) < 10000.0f)
            {
                let fDistanceToEye: f32 = length(intersectionInfo.mIntersectPosition.xyz - ray.mOrigin.xyz);
                {
                    ret.mHitPosition = intersectionInfo.mIntersectPosition.xyz;
                    ret.mHitNormal = intersectionInfo.mIntersectNormal.xyz;
                    ret.miHitTriangle = aBLASBVHNodes1[iNodeIndex].miPrimitiveID;
                    ret.mBarycentricCoordinate = intersectionInfo.mBarycentricCoordinate;
                    ret.miMeshID = aBLASBVHNodes1[iNodeIndex].miMeshID;

                    break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aBLASBVHNodes1[iNodeIndex].mMinBound.xyz,
                aBLASBVHNodes1[iNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aBLASBVHNodes1[iNodeIndex].miChildren0 + iStartNodeIndex;
                iStackTop += 1;
                aiStack[iStackTop] = aBLASBVHNodes1[iNodeIndex].miChildren1 + iStartNodeIndex;
            }
        }
    }

    return ret;
}

/////
fn intersectTotalBVH4(
    ray: Ray,
    iRootNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 16>;
    aiStack[iStackTop] = iRootNodeIndex;

    ret.mHitPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    ret.miHitTriangle = UINT32_MAX;
    var fClosestDistance: f32 = FLT_MAX;

    for(var iStep: u32 = 0u; iStep < 10000u; iStep++)
    {
        if(iStackTop < 0)
        {
            break;
        }

        let iTLASNodeIndex: u32 = aiStack[iStackTop];
        iStackTop -= 1;

        if(aTLASBVHNodes[iTLASNodeIndex].miMeshID != UINT32_MAX)
        {
            // min and max bounds of the top mesh bvh node to check ray intersection against
            let iMeshID: u32 = aTLASBVHNodes[iTLASNodeIndex].miMeshID;
            let iBufferIndex: u32 = aMeshBLASNodeIndices[iMeshID].miBufferIndex;
            let iBLASNodeIndex: u32 = aMeshBLASNodeIndices[iMeshID].miStartNodeIndex;
            let diff: vec3<f32> = aTLASBVHNodes[iTLASNodeIndex].mCentroid.xyz - ray.mOrigin.xyz;
            let fDistance: f32 = dot(diff, diff);

            //if(fDistance <= 20.0f)
            {
                var intersectBLASResult: IntersectBVHResult;
                if(iBufferIndex == 0u)
                {
                    intersectBLASResult = intersectBLASBVH4Part0(ray, iBLASNodeIndex);
                }
                else 
                {
                    intersectBLASResult = intersectBLASBVH4Part1(ray, iBLASNodeIndex);
                }

                if(intersectBLASResult.miHitTriangle != UINT32_MAX)
                {
                    ret.miMeshID = iMeshID;
                    ret.miHitTriangle = intersectBLASResult.miHitTriangle;
                    break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aTLASBVHNodes[iTLASNodeIndex].mMinBound.xyz,
                aTLASBVHNodes[iTLASNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aTLASBVHNodes[iTLASNodeIndex].miChildren0;
                iStackTop += 1;
                aiStack[iStackTop] = aTLASBVHNodes[iTLASNodeIndex].miChildren1;
            }
        }
    }

    return ret;
}

