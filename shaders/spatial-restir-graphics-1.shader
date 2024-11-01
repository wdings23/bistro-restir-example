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
    @location(0) radianceOutput : vec4<f32>,
    @location(1) spatialReservoir: vec4<f32>,
    @location(2) rayDirection: vec4<f32>,
    @location(3) hitPosition: vec4<f32>,
    @location(4) hitNormal: vec4<f32>,
};

struct UniformData
{
    mfSampleRadius: f32,
    miNeighborBlockCheck: u32,
    miNumSamples: u32,
    miPadding: u32,
};

struct SpatialRestirResult
{
    mRadiance: vec4<f32>,
    mReservoir: vec4<f32>,
    mRayDirection: vec4<f32>,
    mAmbientOcclusion: vec4<f32>,
    mHitPosition: vec4<f32>,
    mHitNormal: vec4<f32>,
    miHitMesh: u32,
    mRandomResult: RandomResult,
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
var texCoordTexture: texture_2d<f32>;

@group(0) @binding(3)
var skyTexture: texture_2d<f32>;

@group(0) @binding(4)
var temporalReservoirTexture: texture_2d<f32>;

@group(0) @binding(5)
var temporalRadianceTexture: texture_2d<f32>;

@group(0) @binding(6)
var temporalHitPositionTexture: texture_2d<f32>;

@group(0) @binding(7)
var temporalHitNormalTexture: texture_2d<f32>;

@group(0) @binding(8)
var spatialReservoirTexture: texture_2d<f32>;

@group(0) @binding(9)
var spatialRadianceTexture: texture_2d<f32>;

@group(0) @binding(10)
var spatialHitPositionTexture: texture_2d<f32>;

@group(0) @binding(11)
var spatialHitNormalTexture: texture_2d<f32>;

@group(0) @binding(12)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(13)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(14)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(15)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(16)
var prevFilteredAmbientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(17)
var prevSpatialReservoirTexture: texture_2d<f32>;

@group(0) @binding(18)
var prevSpatialRadianceTexture: texture_2d<f32>;

@group(0) @binding(19)
var prevSpatialHitPositionTexture: texture_2d<f32>;

@group(0) @binding(20)
var prevSpatialHitNormalTexture: texture_2d<f32>;

@group(0) @binding(21)
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

    var randomResult: RandomResult = initRand(
        u32(in.texCoord.x * 100.0f + in.texCoord.y * 200.0f) + u32(defaultUniformData.mfRand0 * 100.0f),
        u32(in.pos.x * 10.0f + in.pos.z * 20.0f) + u32(defaultUniformData.mfRand1 * 100.0f),
        10u);

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture, 
        textureSampler, 
        in.texCoord);

    if(worldPosition.w <= 0.0f)
    {
        out.radianceOutput = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        out.spatialReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    var iNumSamples: u32 = uniformData.miNumSamples; 
    var fSampleRadius: f32 = uniformData.mfSampleRadius;

    var result: SpatialRestirResult;
    //result.mReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    //result.mRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    //result.mHitPosition = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    //result.mHitNormal = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    result.mReservoir = textureSample(
        spatialReservoirTexture,
        textureSampler,
        in.texCoord);
    result.mRadiance = textureSample(
        spatialRadianceTexture,
        textureSampler,
        in.texCoord);
    result.mHitPosition = textureSample(
        spatialHitPositionTexture,
        textureSampler,
        in.texCoord);
    result.mHitNormal = textureSample(
        spatialHitNormalTexture,
        textureSampler,
        in.texCoord);

    // first result using the previous frame's info
    let prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);
    var firstResult: SpatialRestirResult;
    firstResult.mReservoir = textureSample(
        prevSpatialReservoirTexture,
        textureSampler,
        in.texCoord);
    firstResult.mRadiance = textureSample(
        prevSpatialRadianceTexture,
        textureSampler,
        prevScreenUV);
    firstResult.mHitPosition = textureSample(
        prevSpatialHitPositionTexture,
        textureSampler,
        prevScreenUV);
    firstResult.mHitNormal = textureSample(
        prevSpatialHitNormalTexture,
        textureSampler,
        prevScreenUV);

    result = spatialRestir(
        result,
        iNumSamples,
        fSampleRadius,
        in.texCoord,
        1200.0f,
        randomResult);
    
    // intersection check to see if the neighbor's ray direction is blocked
    if(uniformData.miNeighborBlockCheck > 0)
    {
        var ray: Ray;
        ray.mDirection = vec4<f32>(result.mRayDirection.xyz, 1.0f);
        ray.mOrigin = vec4<f32>(worldPosition.xyz + ray.mDirection.xyz * 0.01f, 1.0f);
        var intersectionInfo: IntersectBVHResult;
        intersectionInfo = intersectBVH4(ray, 0u);
        if((length(result.mHitPosition.xyz) >= 1000.0f && abs(intersectionInfo.mHitPosition.x) < 1000.0f) || 
        (length(result.mHitPosition.xyz) < 1000.0f && abs(intersectionInfo.mHitPosition.x) >= 1000.0f))
        {
            result = firstResult;
        }
    }
    
    out.radianceOutput = result.mRadiance * result.mReservoir.w;
    out.radianceOutput.w = 1.0f;
    out.spatialReservoir = result.mReservoir;
    out.rayDirection = vec4<f32>(normalize(result.mHitPosition.xyz - worldPosition.xyz), 1.0f);
    out.hitPosition = result.mHitPosition;
    out.hitNormal = result.mHitNormal;

    return out;
}

/////
fn spatialRestir(
    result: SpatialRestirResult,
    iNumSamples: u32,
    fSampleRadius: f32,
    texCoord: vec2<f32>,
    fMaxSpatialReservoirSamples: f32,
    randomResult: RandomResult) -> SpatialRestirResult
{
    var ret: SpatialRestirResult = result;
    ret.mRandomResult = randomResult;

    //var fSpatialToTemporalPct: f32 = 8.0f / 1200.0f;
    //ret.mReservoir *= fSpatialToTemporalPct;
    //ret.mReservoir.y = result.mReservoir.y;

    var fInitialSampleCount: f32 = result.mReservoir.z;

    let iImageWidth: u32 = u32(defaultUniformData.miScreenWidth);
    let iImageHeight: u32 = u32(defaultUniformData.miScreenHeight); 

    let iX: i32 = i32(floor(texCoord.x * f32(iImageWidth) + 0.5f));
    let iY: i32 = i32(floor(texCoord.y * f32(iImageHeight) + 0.5f));

    var centerWorldPosition: vec3<f32>;
    var centerNormal: vec3<f32>;

    var fCenterDepth: f32 = 1.0f;
    var iCenterMeshID: u32 = 0;

    var fNumValidSamples: f32 = ret.mReservoir.z * 0.7f;
    //var fNumValidSamples: f32 = 0.0f;

    var fPlaneD: f32 = 0.0f;

    var prevScreenUV: vec2<f32> = getPreviousScreenUV(texCoord);
    var centerAO: vec4<f32> = textureSample(
        prevFilteredAmbientOcclusionTexture,
        textureSampler,
        prevScreenUV);

    for(var iSample: u32 = 0u; iSample < iNumSamples; iSample++)
    {
        ret.mRandomResult = nextRand(ret.mRandomResult.miSeed);
        let fRand0: f32 = ret.mRandomResult.mfNum;
        ret.mRandomResult = nextRand(ret.mRandomResult.miSeed);
        let fRand1: f32 = ret.mRandomResult.mfNum;

        var iOffsetX: i32 = i32((fRand0 * fSampleRadius * 2.0f - fSampleRadius));
        var iOffsetY: i32 = i32((fRand1 * fSampleRadius * 2.0f - fSampleRadius));

        if(iSample == 0)
        {
            iOffsetX = 0;
            iOffsetY = 0;
        }

        let iCurrY: i32 = iY + iOffsetY;
        if(iCurrY < 0 || iCurrY >= i32(iImageHeight))
        {
            continue;
        }

        let iCurrX: i32 = iX + iOffsetX;
        if(iCurrX < 0 || iCurrX >= i32(iImageWidth))
        {
            continue;
        }

        let iNeighborX: i32 = iCurrX;
        let iNeighborY: i32 = iCurrY;
        
        // neighbor info
        var neighborUV: vec2<f32> = vec2<f32>(
            f32(iNeighborX) / f32(iImageWidth),
            f32(iNeighborY) / f32(iImageHeight)
        );

        if(iSample == 0)
        {
            neighborUV = texCoord;
        }

        var neighborSpatialReservoir: vec4<f32> = textureSample(
            spatialReservoirTexture,
            textureSampler,
            neighborUV);
        var neighborSpatialHitPosition: vec4<f32> = textureSample(
            spatialHitPositionTexture,
            textureSampler,
            neighborUV);
        var neighborSpatialHitNormal: vec4<f32> = textureSample(
            spatialHitNormalTexture,
            textureSampler,
            neighborUV);
        var neighborWorldPosition: vec4<f32> = textureSample(
            worldPositionTexture,
            textureSampler,
            neighborUV);
        var neighborNormal: vec4<f32> = textureSample(
            normalTexture,
            textureSampler,
            neighborUV);
        var neighborTemporalRadiance: vec4<f32> = textureSample(
            temporalRadianceTexture,
            textureSampler,
            neighborUV);

        var neighborAO: vec4<f32> = textureSample(
            prevFilteredAmbientOcclusionTexture,
            textureSampler,
            neighborUV);

        if(neighborSpatialReservoir.z <= 1.0f)
        {
            continue;
        }

        if(iSample == 0)
        {
            fPlaneD = -dot(neighborWorldPosition.xyz, neighborNormal.xyz);
        }

        let iHitPointMesh: u32 = u32(neighborSpatialHitPosition.w);
        
        // world position check
        if(neighborWorldPosition.w <= 0.0f)
        {
            continue;
        }

        // valid reservoir check
        if(neighborSpatialReservoir.z <= 0.0f)
        {
            continue;
        }

        var fJacobian: f32 = 1.0f;
        var fAttenuation: f32 = 1.0f;
        if(iSample == 0)
        {
            // save as center pixel
            centerWorldPosition = neighborWorldPosition.xyz;
            centerNormal = neighborNormal.xyz;
            fCenterDepth = fract(neighborWorldPosition.w);
            iCenterMeshID = u32(neighborWorldPosition.w - fCenterDepth);

            // BSDF terms
            let centerToNeighborHitPoint: vec3<f32> = neighborSpatialHitPosition.xyz - centerWorldPosition;
            let centerToNeighborHitPointNormalized: vec3<f32> = normalize(centerToNeighborHitPoint);
            let fDirectionDP: f32 = max(dot(centerToNeighborHitPointNormalized, centerNormal), 0.0f);
            var fCenterToHitPointLengthSquared: f32 = 1.0f;
            var fHitPointToNormalDP: f32 = 1.0f;
            var fOneOverHitPositionDistanceSquared: f32 = 1.0f;
            if(neighborSpatialHitPosition.x < 5000.0f)
            {
                fOneOverHitPositionDistanceSquared = 1.0f / max(fCenterToHitPointLengthSquared, 1.0f);
                fHitPointToNormalDP = max(dot(
                    vec3<f32>(-centerToNeighborHitPointNormalized.x, -centerToNeighborHitPointNormalized.y, -centerToNeighborHitPointNormalized.z), 
                    neighborSpatialHitNormal.xyz), 0.0f);
                fCenterToHitPointLengthSquared = dot(centerToNeighborHitPoint, centerToNeighborHitPoint);
            }
        }
        else
        {
            let kfNeighborDepthDifferenceThreshold: f32 = 0.05f;
            let kfNeighborNormalDifferenceThreshold: f32 = 0.80f;

            // neighbor normal difference check 
            let fDP: f32 = max(dot(neighborNormal.xyz, centerNormal), 0.0f);
            if(fDP <= kfNeighborNormalDifferenceThreshold)
            {
                continue;
            }

            // neightbor depth difference check
            let fNeighborDepth: f32 = fract(neighborWorldPosition.w);
            
            let fPlaneDistance: f32 = dot(neighborWorldPosition.xyz, centerNormal.xyz) + fPlaneD;
            if(abs(fPlaneDistance) >= 0.2f)
            {
                continue;
            }

            // mesh id difference check
            let iNeighborMeshID: u32 = u32(floor((neighborWorldPosition.w - fNeighborDepth) + 0.5f));
            if(iNeighborMeshID != iCenterMeshID)
            {
                continue;
            }
            
            // ambient occlusion difference check
            let fAODiff: f32 = abs(centerAO.x - neighborAO.x);
            if(fAODiff >= 0.1f)
            {
                continue;
            } 

            // jacobian
            // neighbor angle to hit position difference
            let centerToNeighborHitPointUnNormalized: vec3<f32> = neighborSpatialHitPosition.xyz - centerWorldPosition;
            let neighborToNeighborHitPointUnNormalized: vec3<f32> = neighborSpatialHitPosition.xyz - neighborWorldPosition.xyz;
            let centerToNeighborHitPointNormalized: vec3<f32> = normalize(centerToNeighborHitPointUnNormalized);
            let neighborToNeighborHitPointNormalized: vec3<f32> = normalize(neighborToNeighborHitPointUnNormalized);
            
            // compare normals for jacobian
            let fDP0: f32 = max(dot(neighborSpatialHitNormal.xyz, centerToNeighborHitPointNormalized * -1.0f), 0.0f);
            var fDP1: f32 = max(dot(neighborSpatialHitNormal.xyz, neighborToNeighborHitPointNormalized * -1.0f), 1.0e-4f);
            fJacobian = fDP0 / fDP1;

            // neighbor distance to hit position difference
            let fCenterToHitPointLength: f32 = length(centerToNeighborHitPointUnNormalized);
            let fNeighborToHitPointLength: f32 = length(neighborToNeighborHitPointUnNormalized);
            fJacobian *= ((fCenterToHitPointLength * fCenterToHitPointLength) / (fNeighborToHitPointLength * fNeighborToHitPointLength));
            fJacobian = clamp(fJacobian, 0.0f, 1.0f);

        }   // if sample > 0

        neighborTemporalRadiance.x *= fJacobian;
        neighborTemporalRadiance.y *= fJacobian;
        neighborTemporalRadiance.z *= fJacobian;

        let fPHat: f32 = length(ToneMapFilmic_Hejl2015(
            neighborTemporalRadiance.xyz, 
            max(max(defaultUniformData.mLightRadiance.x, defaultUniformData.mLightRadiance.y), defaultUniformData.mLightRadiance.z)
        ));

        ret = updateSpatialReservoir(
            ret,

            neighborSpatialReservoir,
            neighborSpatialHitPosition,
            neighborSpatialHitNormal,
            neighborWorldPosition,
            neighborTemporalRadiance,
            fPHat,

            iHitPointMesh,

            fMaxSpatialReservoirSamples,
            neighborSpatialReservoir.z,
            ret.mRandomResult,
            fJacobian,
            iSample);
        
        fNumValidSamples += neighborSpatialReservoir.z * f32(fPHat > 0.0f);

    }   // for sample = 0 to num samples

    // weight
    ret.mReservoir.w = clamp(ret.mReservoir.x / (fNumValidSamples * ret.mReservoir.y + 0.0001f), 0.2f, 1.0f);
    
    return ret;
}

//////
fn updateSpatialReservoir(
    result: SpatialRestirResult,
     
    temporalReservoir: vec4<f32>,
    temporalHitPosition: vec4<f32>,
    temporalHitNormal: vec4<f32>,
    worldPosition: vec4<f32>,
    radiance: vec4<f32>,
    fPHat: f32,

    iHitPointMesh: u32,
    
    fMaxReservoirSamples: f32,
    fReservoirSize: f32,
    randomResult: RandomResult,
    fJacobian: f32,
    iSampleIndex: u32) -> SpatialRestirResult
{
    var ret: SpatialRestirResult = result; 
    ret.mRandomResult = randomResult;
    ret.mRandomResult = nextRand(ret.mRandomResult.miSeed);

    let fRand0: f32 = ret.mRandomResult.mfNum;
    
    // update temporal reservoir with radiance
    let fWeight: f32 = fPHat * temporalReservoir.z * temporalReservoir.w;
    ret.mReservoir.x += fWeight;
    ret.mReservoir.z += fReservoirSize;
    
    let fWeightPct: f32 = fWeight / max(ret.mReservoir.x, 0.001f);
    let fRand: f32 = fRand0;
    if(fRand < fWeightPct || ret.mReservoir.z <= 0.0f || iSampleIndex == 0u)
    {
        ret.mReservoir.y = fPHat;
        ret.mRadiance = radiance;
        ret.mHitPosition = temporalHitPosition;
        ret.mRayDirection = vec4<f32>(normalize(temporalHitPosition.xyz - worldPosition.xyz), fJacobian);
        ret.mHitNormal = temporalHitNormal;
        ret.miHitMesh = iHitPointMesh;
    }

    // clamp reservoir
    if(ret.mReservoir.z > fMaxReservoirSamples)
    {
        let fPct: f32 = fMaxReservoirSamples / ret.mReservoir.z;
        ret.mReservoir.x *= fPct;
        ret.mReservoir.z = fMaxReservoirSamples;
    }

    return ret;
}

//////
fn checkClipSpaceBlock(
    centerWorldPosition: vec3<f32>,
    centerToNeighborHitPointNormalized: vec3<f32>
) -> bool
{
    var bBlocked: bool = false;
    let iNumBlockSteps: i32 = 6;
    var currCheckPosition: vec3<f32> = centerWorldPosition;
    var startScreenPosition: vec2<i32> = vec2<i32>(-1, -1);
    var currScreenPosition: vec2<i32> = vec2<i32>(-1, -1);
    for(var iStep: i32 = 0; iStep < iNumBlockSteps; iStep++)
    {
        // convert to clipspace for fetching world position from texture
        var clipSpacePosition: vec4<f32> = vec4<f32>(currCheckPosition, 1.0f) * defaultUniformData.mViewProjectionMatrix;
        clipSpacePosition.x /= clipSpacePosition.w;
        clipSpacePosition.y /= clipSpacePosition.w;
        clipSpacePosition.z /= clipSpacePosition.w;
        currCheckPosition += centerToNeighborHitPointNormalized * 0.05f;

        let currScreenUV = vec2<f32>(
            clipSpacePosition.x * 0.5f + 0.5f,
            1.0f - (clipSpacePosition.y * 0.5f + 0.5f)
        );

        currScreenPosition.x = i32(currScreenUV.x * f32(defaultUniformData.miScreenWidth));
        currScreenPosition.y = i32(currScreenUV.y * f32(defaultUniformData.miScreenHeight));

        // only check the surrounding pixel 
        if(abs(currScreenPosition.x - startScreenPosition.x) > 6 || abs(currScreenPosition.y - startScreenPosition.y) > 6)
        {
            continue;
        }

        // out of bounds
        if(currScreenPosition.x < 0 || currScreenPosition.x >= i32(defaultUniformData.miScreenWidth) || 
            currScreenPosition.y < 0 || currScreenPosition.y >= i32(defaultUniformData.miScreenHeight))
        {
            continue;
        }

        if(iStep == 0)
        {
            // save the starting screen position
            startScreenPosition = currScreenPosition;
            continue;
        }
        else if(startScreenPosition.x >= 0)
        {   
            // still at the same pixel position
            if(currScreenPosition.x == startScreenPosition.x && currScreenPosition.y == startScreenPosition.y)
            {
                iStep -= 1;
                continue;
            }
        }

        // compare depth value, smaller is in front, therefore blocked
        let currWorldPosition: vec4<f32> = textureSample(
            worldPositionTexture,
            textureSampler,
            currScreenUV);
        if(currWorldPosition.w == 0.0f)
        {
            continue;
        }
        let fCurrDepth: f32 = fract(currWorldPosition.w);
        if(fCurrDepth < clipSpacePosition.z)
        {
            bBlocked = true;

            break;
        }
    }

    return bBlocked;
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
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
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
fn isDisoccluded(
    inputTexCoord: vec2<f32>) -> bool
{
    // world position, normal, and motion vector
    let worldPosition = textureSample(
        worldPositionTexture,
        textureSampler,
        inputTexCoord);
    let normal = textureSample(
        normalTexture,
        textureSampler,
        inputTexCoord);
    var motionVector: vec4<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        inputTexCoord);
    motionVector.x = motionVector.x * 2.0f - 1.0f;
    motionVector.y = motionVector.y * 2.0f - 1.0f;
    
    let iMesh: u32 = u32(ceil(motionVector.z - 0.5f)) - 1;

    // world position, normal, motion vector from previous frame with back projected uv
    var backProjectedScreenUV: vec2<f32> = inputTexCoord - motionVector.xy;
    if(backProjectedScreenUV.x < 0.0f || backProjectedScreenUV.y < 0.0f || 
       backProjectedScreenUV.x > 1.0f || backProjectedScreenUV.y > 1.0f)
    {
        return true;
    }

    var prevWorldPosition: vec4<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        backProjectedScreenUV
    );
    var prevNormal: vec4<f32> = textureSample(
        prevNormalTexture,
        textureSampler,
        backProjectedScreenUV
    );
    var prevMotionVectorAndMeshIDAndDepth: vec4<f32> = textureSample(
        prevMotionVectorTexture,
        textureSampler,
        backProjectedScreenUV
    );

    var fOneOverScreenHeight: f32 = 1.0f / f32(defaultUniformData.miScreenHeight);
    var fOneOverScreenWidth: f32 = 1.0f / f32(defaultUniformData.miScreenWidth);

    var bestBackProjectedScreenUV: vec2<f32> = backProjectedScreenUV;
    var fShortestWorldDistance: f32 = FLT_MAX;
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        var fSampleY: f32 = backProjectedScreenUV.y + f32(iY) * fOneOverScreenHeight;
        fSampleY = clamp(fSampleY, 0.0f, 1.0f);
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            var fSampleX: f32 = backProjectedScreenUV.x + f32(iX) * fOneOverScreenWidth;
            fSampleX = clamp(fSampleX, 0.0f, 1.0f);

            var sampleUV: vec2<f32> = vec2<f32>(fSampleX, fSampleY);
            var checkPrevWorldPosition = textureSample(
                prevWorldPositionTexture,
                textureSampler,
                sampleUV
            );

            var worldPositionDiff: vec3<f32> = checkPrevWorldPosition.xyz - worldPosition.xyz;
            var fCheckWorldPositionDistance: f32 = dot(worldPositionDiff, worldPositionDiff);
            if(fCheckWorldPositionDistance < fShortestWorldDistance)
            {
                fShortestWorldDistance = fCheckWorldPositionDistance;
                bestBackProjectedScreenUV = sampleUV;
            }
        }
    }

    backProjectedScreenUV = bestBackProjectedScreenUV;
    prevWorldPosition = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        backProjectedScreenUV
    );
    prevNormal = textureSample(
        prevNormalTexture,
        textureSampler,
        backProjectedScreenUV
    );
    prevMotionVectorAndMeshIDAndDepth = textureSample(
        prevMotionVectorTexture,
        textureSampler,
        backProjectedScreenUV
    );
    
    let toCurrentDir: vec3<f32> = worldPosition.xyz - prevWorldPosition.xyz;
    //let fPlaneDistance: f32 = abs(dot(toCurrentDir, normal.xyz)); 
    let fPrevPlaneDistance: f32 = abs(dot(prevWorldPosition.xyz, normal.xyz)) - abs(dot(worldPosition.xyz, normal.xyz));

    // compute difference in world position, depth, and angle from previous frame
    
    var fDepth: f32 = motionVector.w;
    var fPrevDepth: f32 = prevMotionVectorAndMeshIDAndDepth.w;
    var fCheckDepth: f32 = abs(fDepth - fPrevDepth);
    var worldPositionDiff: vec3<f32> = prevWorldPosition.xyz - worldPosition.xyz;
    var fCheckDP: f32 = abs(dot(normalize(normal.xyz), normalize(prevNormal.xyz)));
    let iPrevMesh: u32 = u32(ceil(prevMotionVectorAndMeshIDAndDepth.z - 0.5f)) - 1;
    var fCheckWorldPositionDistance: f32 = dot(worldPositionDiff, worldPositionDiff);

    //return !(iMesh == iPrevMesh && fCheckDepth <= 0.004f && fCheckWorldPositionDistance <= 0.001f && fCheckDP >= 0.99f);
    return !(iMesh == iPrevMesh && fCheckWorldPositionDistance <= 0.00025f && fCheckDP >= 0.99f);
    //return !(iMesh == iPrevMesh && fPrevPlaneDistance <= 0.00005f && fCheckDP >= 0.99f);
}

/////
fn getPreviousScreenUV(
    screenUV: vec2<f32>) -> vec2<f32>
{
    var motionVector: vec2<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        screenUV).xy;
    motionVector = motionVector * 2.0f - 1.0f;
    var prevScreenUV: vec2<f32> = screenUV - motionVector;

    var worldPosition: vec3<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        screenUV
    ).xyz;
    var normal: vec3<f32> = textureSample(
        normalTexture,
        textureSampler,
        screenUV
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
                sampleUV
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

/////
fn isDisoccluded2(
    screenUV: vec2<f32>,
    prevScreenUV: vec2<f32>
) -> bool
{
    var worldPosition: vec3<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        screenUV).xyz;

    var prevWorldPosition: vec3<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        prevScreenUV).xyz;

    var normal: vec3<f32> = textureSample(
        normalTexture,
        textureSampler,
        screenUV).xyz;

    var prevNormal: vec3<f32> = textureSample(
        prevNormalTexture,
        textureSampler,
        prevScreenUV).xyz;

    var motionVector: vec4<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        screenUV);

    var prevMotionVectorAndMeshIDAndDepth: vec4<f32> = textureSample(
        prevMotionVectorTexture,
        textureSampler,
        prevScreenUV);

    let iMesh = u32(ceil(motionVector.z - 0.5f)) - 1;
    var fDepth: f32 = motionVector.w;
    var fPrevDepth: f32 = prevMotionVectorAndMeshIDAndDepth.w;
    var fCheckDepth: f32 = abs(fDepth - fPrevDepth);
    var worldPositionDiff: vec3<f32> = prevWorldPosition.xyz - worldPosition.xyz;
    var fCheckDP: f32 = abs(dot(normalize(normal.xyz), normalize(prevNormal.xyz)));
    let iPrevMesh: u32 = u32(ceil(prevMotionVectorAndMeshIDAndDepth.z - 0.5f)) - 1;
    var fCheckWorldPositionDistance: f32 = dot(worldPositionDiff, worldPositionDiff);

    return !(iMesh == iPrevMesh && fCheckDepth <= 0.01f && fCheckWorldPositionDistance <= 0.01f && fCheckDP >= 0.99f);
}

///// 
fn isPrevUVOutOfBounds(inputTexCoord: vec2<f32>) -> bool
{
    var motionVector: vec4<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        inputTexCoord);
    motionVector.x = motionVector.x * 2.0f - 1.0f;
    motionVector.y = motionVector.y * 2.0f - 1.0f;
    let backProjectedScreenUV: vec2<f32> = inputTexCoord - motionVector.xy;

    return (backProjectedScreenUV.x < 0.0f || backProjectedScreenUV.x > 1.0 || backProjectedScreenUV.y < 0.0f || backProjectedScreenUV.y > 1.0f);
}

/////
fn ToneMapFilmic_Hejl2015(
    hdr: vec3<f32>, 
    whitePt: f32) -> vec3<f32>
{
    let vh: vec4<f32> = vec4<f32>(hdr, whitePt);
    let va: vec4<f32> = (vh * 1.425f) + 0.05f;
    let vf: vec4<f32> = ((vh * va + 0.004f) / ((vh * (va + 0.55f) + 0.0491f)));// - vec4<f32>(0.0821f);
    var ret: vec3<f32> = vf.rgb / vf.w;
    ret = clamp(ret.xyz, vec3<f32>(0.0f, 0.0f, 0.0f), vec3<f32>(1.0f, 1.0f, 1.0f));

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