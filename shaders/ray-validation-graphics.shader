const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const fOneOverPI : f32 = 1.0f / PI;

struct RandomResult {
    mfNum: f32,
    miSeed: u32,
};

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @location(2) color : vec4<f32>
};
struct VertexOutput {
    @location(0) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) color: vec4<f32>
};
struct FragmentOutput {
    @location(0) temporalRadianceOutput : vec4<f32>,
    @location(1) spatialRadianceOutput : vec4<f32>,
    @location(2) temporalReservoirOutput: vec4<f32>,
    @location(3) spatialReservoirOutput: vec4<f32>,
    @location(4) hitPosition: vec4<f32>,
    @location(5) rayDirection: vec4<f32>,
    @location(6) ambientOcclusionOutput: vec4<f32>,
};

struct UniformData
{
    mfSampleRadius: f32,
    miNeighborBlockCheck: u32,
    miNumSamples: u32,
    miPadding: u32,
};

struct IntersectBVHResult
{
    mHitPosition: vec3<f32>,
    mHitNormal: vec3<f32>,
    miHitTriangle: u32,
    mBarycentricCoordinate: vec3<f32>,
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

struct Ray
{
    mOrigin: vec4<f32>,
    mDirection: vec4<f32>,
    mfT: vec4<f32>,
};

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
var temporalReservoirTexture: texture_2d<f32>;

@group(0) @binding(3)
var temporalRadianceTexture: texture_2d<f32>;

@group(0) @binding(4)
var spatialReservoirTexture: texture_2d<f32>;

@group(0) @binding(5)
var spatialRadianceTexture: texture_2d<f32>;

@group(0) @binding(6)
var spatialHitPositionTexture: texture_2d<f32>;

@group(0) @binding(7)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(8)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(9)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(10)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(11)
var prevTemporalReservoirTexture: texture_2d<f32>;

@group(0) @binding(12)
var prevSpatialReservoirTexture: texture_2d<f32>;

@group(0) @binding(13)
var prevSpatialRadianceTexture: texture_2d<f32>;

@group(0) @binding(14)
var prevSpatialHitPositionTexture: texture_2d<f32>;

@group(0) @binding(15)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(16)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> aDynamicBVHNodes: array<BVHNode2>;

@group(1) @binding(2)
var<storage, read> aDynamicVertexPositions: array<vec4<f32>>;

@group(1) @binding(3)
var<storage, read> aiDynamicTriangleIndices: array<u32>;

@group(1) @binding(4)
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
    var out: FragmentOutput;

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture, 
        textureSampler, 
        in.texCoord);

    if(worldPosition.w <= 0.0f)
    {
        out.temporalRadianceOutput = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        out.spatialRadianceOutput = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        out.spatialReservoirOutput = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        out.temporalReservoirOutput = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    var temporalReservoir: vec4<f32> = textureSample(
        temporalReservoirTexture,
        textureSampler,
        in.texCoord);
    var spatialReservoir: vec4<f32> = textureSample(
        spatialReservoirTexture,
        textureSampler,
        in.texCoord);
    var hitPosition: vec4<f32> = textureSample(
        spatialHitPositionTexture,
        textureSampler,
        in.texCoord);
    var temporalRadiance: vec4<f32> = textureSample(
        temporalRadianceTexture,
        textureSampler,
        in.texCoord);
    var spatialRadiance: vec4<f32> = textureSample(
        spatialRadianceTexture,
        textureSampler,
        in.texCoord);

    var prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);
    var prevTemporalReservoir: vec4<f32> = textureSample(
        prevTemporalReservoirTexture,
        textureSampler,
        in.texCoord);
    var prevSpatialReservoir: vec4<f32> = textureSample(
        prevSpatialReservoirTexture,
        textureSampler,
        prevScreenUV
    );
    var prevHitPosition: vec4<f32> = textureSample(
        prevSpatialHitPositionTexture,
        textureSampler,
        prevScreenUV
    );
    var prevRadiance: vec4<f32> = textureSample(
        prevSpatialRadianceTexture,
        textureSampler,
        prevScreenUV
    );

    var ambientOcclusionOutput: vec4<f32> = textureSample(
        ambientOcclusionTexture,
        textureSampler,
        in.texCoord
    );

    // intersection check to see if the neighbor's ray direction is blocked
    var ray: Ray;
    ray.mDirection = vec4<f32>(normalize(hitPosition.xyz - worldPosition.xyz), 1.0f);
    ray.mOrigin = vec4<f32>(worldPosition.xyz + ray.mDirection.xyz * 0.01f, 1.0f);
    var intersectionInfo: IntersectBVHResult;
    intersectionInfo = intersectBVH4(ray, 0u);
    //if(length(hitPosition.xyz) < 1000.0f)
    //{
    //    if(length(hitPosition.xyz - intersectionInfo.mHitPosition.xyz) >= 1.0f)
    //    {
    //        spatialRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    //        spatialReservoir = prevSpatialReservoir;
    //        spatialReservoir.y = 0.0f;
    //        hitPosition = vec4<f32>(intersectionInfo.mHitPosition.xyz, f32(intersectionInfo.miHitTriangle));
    //    }
    //}
    //else
    //{
    //    if(length(intersectionInfo.mHitPosition) < 1000.0f)
    //    {
    //        spatialRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    //        spatialReservoir = prevSpatialReservoir;
    //        spatialReservoir.y = 0.0f;
    //        hitPosition = vec4<f32>(intersectionInfo.mHitPosition.xyz, f32(intersectionInfo.miHitTriangle));
    //    }
    //}

    if(intersectionInfo.miHitTriangle != UINT32_MAX && length(hitPosition.xyz) >= 1000.0f)
    {
        ambientOcclusionOutput.x = clamp(ambientOcclusionOutput.x + 1.0f, 0.0f, ambientOcclusionOutput.y);
        temporalReservoir.x = 0.0f;
        spatialReservoir.x = 0.0f;
        temporalReservoir.y = 0.0f;
        spatialReservoir.y = 0.0f;

        temporalReservoir.w = clamp(temporalReservoir.x / max(temporalReservoir.z * temporalReservoir.y, 0.001f), 0.0f, 1.0f);
        spatialReservoir.w = clamp(spatialReservoir.x / max(spatialReservoir.z * spatialReservoir.y, 0.001f), 0.0f, 1.0f);

        temporalRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
        spatialRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    }

    ambientOcclusionOutput.z = 1.0f - (ambientOcclusionOutput.x / ambientOcclusionOutput.y);

    out.temporalRadianceOutput = vec4<f32>(temporalRadiance.xyz, 1.0f);
    out.spatialRadianceOutput = vec4<f32>(spatialRadiance.xyz, 1.0f);
    out.hitPosition = vec4<f32>(intersectionInfo.mHitPosition.xyz, f32(intersectionInfo.miHitTriangle));
    out.rayDirection = vec4<f32>(normalize(hitPosition.xyz - worldPosition.xyz), 1.0f);
    out.ambientOcclusionOutput = ambientOcclusionOutput;
    out.temporalReservoirOutput = temporalReservoir;
    out.spatialReservoirOutput = spatialReservoir;

    return out;
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

/////
fn intersectTri4(
    ray: Ray,
    iTriangleIndex: u32) -> RayTriangleIntersectionResult
{
    //let iIndex0: u32 = aiDynamicTriangleIndices[iTriangleIndex * 3];
    //let iIndex1: u32 = aiDynamicTriangleIndices[iTriangleIndex * 3 + 1];
    //let iIndex2: u32 = aiDynamicTriangleIndices[iTriangleIndex * 3 + 2];

    let pos0: vec4<f32> = aDynamicVertexPositions[aiDynamicTriangleIndices[iTriangleIndex * 3]];
    let pos1: vec4<f32> = aDynamicVertexPositions[aiDynamicTriangleIndices[iTriangleIndex * 3 + 1]];
    let pos2: vec4<f32> = aDynamicVertexPositions[aiDynamicTriangleIndices[iTriangleIndex * 3 + 2]];

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
fn intersectBVH4(
    ray: Ray,
    iRootNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 128>;
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

        let iNodeIndex: u32 = aiStack[iStackTop];
        iStackTop -= 1;

        //let node: BVHNode2 = aDynamicBVHNodes[iNodeIndex];
        if(aDynamicBVHNodes[iNodeIndex].miPrimitiveID != UINT32_MAX)
        {
            let intersectionInfo: RayTriangleIntersectionResult = intersectTri4(
                ray,
                aDynamicBVHNodes[iNodeIndex].miPrimitiveID);

            if(abs(intersectionInfo.mIntersectPosition.x) < 10000.0f)
            {
                let fDistanceToEye: f32 = length(intersectionInfo.mIntersectPosition.xyz - ray.mOrigin.xyz);
                //if(fDistanceToEye < fClosestDistance)
                {
                    
                    //fClosestDistance = fDistanceToEye;
                    ret.mHitPosition = intersectionInfo.mIntersectPosition.xyz;
                    ret.mHitNormal = intersectionInfo.mIntersectNormal.xyz;
                    ret.miHitTriangle = aDynamicBVHNodes[iNodeIndex].miPrimitiveID;
                    ret.mBarycentricCoordinate = intersectionInfo.mBarycentricCoordinate;

                    break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aDynamicBVHNodes[iNodeIndex].mMinBound.xyz,
                aDynamicBVHNodes[iNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aDynamicBVHNodes[iNodeIndex].miChildren0;
                iStackTop += 1;
                aiStack[iStackTop] = aDynamicBVHNodes[iNodeIndex].miChildren1;
            }
        }
    }

    return ret;
}

/////
fn getPreviousScreenUV(
    screenUV: vec2<f32>) -> vec2<f32>
{
    var screenUVCopy: vec2<f32> = screenUV;
    var motionVector: vec2<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        screenUVCopy).xy;
    motionVector = motionVector * 2.0f - 1.0f;
    var prevScreenUV: vec2<f32> = screenUVCopy - motionVector;

    var worldPosition: vec3<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        screenUVCopy
    ).xyz;
    var normal: vec3<f32> = textureSample(
        normalTexture,
        textureSampler,
        screenUVCopy
    ).xyz;

    var fOneOverScreenWidth: f32 = 1.0f / f32(defaultUniformData.miScreenWidth);
    var fOneOverScreenHeight: f32 = 1.0f / f32(defaultUniformData.miScreenHeight);

    var fShortestWorldDistance: f32 = FLT_MAX;
    var closestScreenUV: vec2<f32> = prevScreenUV;
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            var sampleUV: vec2<f32> = prevScreenUV + vec2<f32>(
                f32(iX) * fOneOverScreenWidth,
                f32(iY) * fOneOverScreenHeight 
            );

            sampleUV.x = clamp(sampleUV.x, 0.0f, 1.0f);
            sampleUV.y = clamp(sampleUV.y, 0.0f, 1.0f);

            var checkWorldPosition: vec3<f32> = textureSample(
                prevWorldPositionTexture,
                textureSampler,
                vec2<f32>(
                    clamp(sampleUV.x, 0.0f, 1.0f),
                    clamp(sampleUV.y, 0.0f, 1.0f)
                )
            ).xyz;
            var checkNormal: vec3<f32> = textureSample(
                prevNormalTexture,
                textureSampler,
                sampleUV
            ).xyz;
            var fNormalDP: f32 = abs(dot(checkNormal, normal));

            var worldPositionDiff: vec3<f32> = checkWorldPosition - worldPosition;
            var fLengthSquared: f32 = dot(worldPositionDiff, worldPositionDiff);
            if(fNormalDP >= 0.99f && fShortestWorldDistance > fLengthSquared)
            {
                fShortestWorldDistance = fLengthSquared;
                closestScreenUV = sampleUV;
            }
        }
    }

    return closestScreenUV;
}