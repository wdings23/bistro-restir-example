const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const PROBE_IMAGE_SIZE: u32 = 8u;
const VALIDATION_STEP: u32 = 6u;

struct RandomResult {
    mfNum: f32,
    miSeed: u32,
};

struct ReservoirResult
{
    mReservoir: vec4<f32>,
    mbExchanged: bool
};

struct UniformData
{
    miFrame: u32,
    miRand0: u32,
    miRand1: u32,
    miRand2: u32,

    mViewProjectionMatrix: mat4x4<f32>,
    
    miScreenWidth: u32,
    miScreenHeight: u32,
    miFrame: u32,
    miNumMeshes: u32,

    mLightDirection: vec4<f32>,
    mLightRadiance: vec4<f32>,
};

struct RadianceResult
{
    mReservoir: vec4<f32>,
    mRadiance: vec3<f32>,
    mRandomResult: RandomResult,
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

struct EmissiveLightResult
{
    mReservoir: vec4<f32>,
    mRadiance: vec4<f32>,
    mHitPosition: vec4<f32>,
    mHitNormal: vec4<f32>,
    mRandomResult: RandomResult,
    mbExchanged: bool,
};

struct SpecularResult
{
    mReservoir: vec4<f32>,
    mRadiance: vec4<f32>,
    mRandomResult: RandomResult,
};

struct IrradianceCacheEntry
{
    mPosition:              vec4<f32>,
    mImageProbe:            array<vec4<f32>, 64>,            // 8x8 image probe
};

struct Ray
{
    mOrigin: vec4<f32>,
    mDirection: vec4<f32>,
    mfT: vec4<f32>,
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
var prevEmissiveRadianceTexture: texture_2d<f32>;

@group(0) @binding(3)
var prevEmissiveReservoirTexture: texture_2d<f32>;

@group(0) @binding(4)
var rayDirectionTexture: texture_2d<f32>;

@group(0) @binding(5)
var rayHitPositionTexture: texture_2d<f32>;

@group(0) @binding(6)
var prevEmissiveTemporalHitPositionTexture: texture_2d<f32>;

@group(0) @binding(7)
var prevEmissiveTemporalHitNormalTexture: texture_2d<f32>;

@group(0) @binding(8)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(9)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(10)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(11)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(12)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> materialData: array<vec4<f32>, 32>;

@group(1) @binding(2)
var<storage, read> meshTriangleRangeData: array<vec2<u32>, 32>;

@group(1) @binding(3)
var<storage, read> aDynamicBVHNodes: array<BVHNode2>;

@group(1) @binding(4)
var<storage, read> aDynamicVertexPositions: array<vec4<f32>>;

@group(1) @binding(5)
var<storage, read> aiDynamicTriangleIndices: array<u32>;

@group(1) @binding(6)
var<uniform> defaultUniformData: DefaultUniformData;

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
    @location(0) emissiveRadiance : vec4<f32>,
    @location(1) emissiveReservoir: vec4<f32>,
    @location(2) emissiveHitPosition: vec4<f32>,
    @location(3) emissiveHitNormal: vec4<f32>,

};

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

    if(defaultUniformData.miFrame < 1)
    {
        return out;
    }

    var randomResult: RandomResult = initRand(
        u32(in.texCoord.x * 100.0f + in.texCoord.y * 200.0f) + u32(defaultUniformData.mfRand0 * 100.0f),
        u32(in.pos.x * 10.0f + in.pos.z * 20.0f) + u32(defaultUniformData.mfRand1 * 100.0f),
        10u);

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture, 
        textureSampler, 
        in.texCoord);

    let normal: vec3<f32> = textureSample(
        normalTexture, 
        textureSampler, 
        in.texCoord).xyz;

    let rayDirection: vec4<f32> = textureSample(
        rayDirectionTexture,
        textureSampler,
        in.texCoord);

    let rayHitPosition: vec4<f32> = textureSample(
        rayHitPositionTexture,
        textureSampler,
        in.texCoord);

    if(worldPosition.w <= 0.0f)
    {
        return out;
    }

    var emissiveResult: EmissiveLightResult; 

    emissiveResult.mReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    emissiveResult.mRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    emissiveResult.mHitPosition = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    emissiveResult.mHitNormal = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    emissiveResult.mRandomResult = randomResult;
    
    var finalRayDirection: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);

    // use previous frame's reservoir and radiance if not disoccluded
    var prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);
    var bDisoccluded: bool = isDisoccluded2(in.texCoord, prevScreenUV);
    if(!bDisoccluded && !isPrevUVOutOfBounds(in.texCoord))
    {
        emissiveResult.mRadiance = textureSample(
            prevEmissiveRadianceTexture,
            textureSampler,
            prevScreenUV
        );
        
        emissiveResult.mReservoir = textureSample(
            prevEmissiveReservoirTexture,
            textureSampler,
            prevScreenUV
        );

        emissiveResult.mHitPosition = textureSample(
            prevEmissiveTemporalHitPositionTexture,
            textureSampler,
            prevScreenUV
        );

        emissiveResult.mHitNormal = textureSample(
            prevEmissiveTemporalHitNormalTexture,
            textureSampler,
            prevScreenUV
        );
    
        finalRayDirection = normalize(emissiveResult.mHitPosition.xyz - worldPosition.xyz);
    }

    
    emissiveResult = temporalRestirEmissiveLight(
        emissiveResult,
        worldPosition.xyz,
        normal.xyz,
        rayDirection.xyz,
        rayHitPosition.xyz,
        rayHitPosition.w,
        1.0f,
        randomResult);

    //if(emissiveResult.mbExchanged)
    //{
        finalRayDirection = normalize(emissiveResult.mHitPosition.xyz - worldPosition.xyz);
    //}

    let fCenterDepth: f32 = fract(worldPosition.w);
    let iCenterMeshID: i32 = i32(worldPosition.w - fCenterDepth);
    let iNumPermutations: i32 = 2;
    let origScreenCoord: vec2<i32> = vec2<i32>(
        i32(in.texCoord.x * f32(defaultUniformData.miScreenWidth)),
        i32(in.texCoord.y * f32(defaultUniformData.miScreenHeight))
    );

    // permutation samples
    var permutationEmissiveResult: EmissiveLightResult = emissiveResult;
    var permutationSampleRayDirection: vec3<f32> = normalize(emissiveResult.mHitPosition.xyz - worldPosition.xyz);
    for(var iSample: i32 = 0; iSample < iNumPermutations; iSample++)
    {
        var aXOR: array<vec2<i32>, 4>;
        aXOR[0] = vec2<i32>(3, 3);
        aXOR[1] = vec2<i32>(2, 1);
        aXOR[2] = vec2<i32>(1, 2);
        aXOR[3] = vec2<i32>(3, 3);
        
        var aOffsets: array<vec2<i32>, 4>;
        aOffsets[0] = vec2<i32>(-1, -1);
        aOffsets[1] = vec2<i32>(1, 1);
        aOffsets[2] = vec2<i32>(-1, 1);
        aOffsets[3] = vec2<i32>(1, -1);

        // apply permutation offset to screen coordinate, converting to uv after
        let iFrame: i32 = i32(defaultUniformData.miFrame);
        let iIndex0: i32 = iFrame & 3;
        let iIndex1: i32 = (iSample + (iFrame ^ 1)) & 3;
        let offset: vec2<i32> = aOffsets[iIndex0] + aOffsets[iIndex1];
        let screenCoord: vec2<i32> = (origScreenCoord + offset) ^ aXOR[iFrame & 3]; 
        
        var sampleRayDirection: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        
        // get sample world position, normal, and ray direction
        var fJacobian: f32 = 1.0f;
        
        // permutation uv
        var sampleUV: vec2<f32> = vec2<f32>(
            ceil(f32(screenCoord.x) + 0.5f) / f32(defaultUniformData.miScreenWidth),
            ceil(f32(screenCoord.y) + 0.5f) / f32(defaultUniformData.miScreenHeight));

        // back project to previous frame's screen coordinate
        var motionVector: vec2<f32> = textureSample(
            motionVectorTexture,
            textureSampler,
            sampleUV).xy;
        motionVector = motionVector * 2.0f - 1.0f;
        sampleUV -= motionVector;

        // sample world position
        let sampleWorldPosition: vec4<f32> = textureSample(
            prevWorldPositionTexture,
            textureSampler,
            sampleUV);

        let sampleNormal: vec3<f32> = textureSample(
            prevNormalTexture,
            textureSampler,
            sampleUV).xyz;

        // neighbor normal difference check 
        let fDP: f32 = dot(sampleNormal, normal);
        if(fDP <=  0.6f)
        {
            continue;
        }

        // neightbor depth difference check
        let fSampleDepth: f32 = fract(sampleWorldPosition.w);
        let fDepthDiff: f32 = abs(fCenterDepth - fSampleDepth);
        if(fDepthDiff >= 0.05f)
        {
            continue;
        } 

        // mesh id difference check
        let iSampleMeshID: i32 = i32(floor((sampleWorldPosition.w - fSampleDepth) + 0.5f));
        if(iSampleMeshID != iCenterMeshID)
        {
            continue;
        }

        // hit point and hit normal for jacobian
        let sampleHitPoint4: vec4<f32> = textureSample(
            prevEmissiveTemporalHitPositionTexture,
            textureSampler,
            sampleUV);

        let sampleHitPoint: vec3<f32> = sampleHitPoint4.xyz;
        let fHitTriangleIndex = sampleHitPoint4.w;

        var neighborHitNormal: vec3<f32> = textureSample(
            prevEmissiveTemporalHitNormalTexture,
            textureSampler,
            sampleUV).xyz;
        let centerToNeighborHitPointUnNormalized: vec3<f32> = sampleHitPoint - worldPosition.xyz;
        let neighborToNeighborHitPointUnNormalized: vec3<f32> = sampleHitPoint - sampleWorldPosition.xyz;
        let centerToNeighborHitPointNormalized: vec3<f32> = normalize(centerToNeighborHitPointUnNormalized);
        let neighborToNeighborHitPointNormalized: vec3<f32> = normalize(neighborToNeighborHitPointUnNormalized);
        
        // compare normals for jacobian
        let fDP0: f32 = max(dot(neighborHitNormal, centerToNeighborHitPointNormalized * -1.0f), 0.0f);
        var fDP1: f32 = max(dot(neighborHitNormal, neighborToNeighborHitPointNormalized * -1.0f), 1.0e-4f);
        fJacobian = fDP0 / fDP1;

        // compare length for jacobian 
        let fCenterToHitPointLength: f32 = length(centerToNeighborHitPointUnNormalized);
        let fNeighborToHitPointLength: f32 = length(neighborToNeighborHitPointUnNormalized);
        fJacobian *= ((fCenterToHitPointLength * fCenterToHitPointLength) / (fNeighborToHitPointLength * fNeighborToHitPointLength));
        fJacobian = clamp(fJacobian, 0.0f, 1.0f);

        sampleRayDirection = centerToNeighborHitPointNormalized;
        emissiveResult = temporalRestirEmissiveLight(
            emissiveResult,
            worldPosition.xyz,
            normal.xyz,
            sampleRayDirection.xyz,
            sampleHitPoint.xyz,
            fHitTriangleIndex,
            fJacobian,
            randomResult);
        
        finalRayDirection = normalize(emissiveResult.mHitPosition.xyz - worldPosition.xyz);
    }

    finalRayDirection = normalize(emissiveResult.mHitPosition.xyz - worldPosition.xyz);

    var ray: Ray;
    ray.mOrigin = vec4<f32>(worldPosition.xyz + normal.xyz * 0.02f, 1.0f);
    ray.mDirection = vec4<f32>(finalRayDirection, 1.0f);
    let intersectionInfo: IntersectBVHResult = intersectBVH4(ray, 0u);
    var hitMeshRadiance: vec3<f32> = getMeshEmissiveRadiance(i32(intersectionInfo.miHitTriangle));

    // validate that the current ray direction is not blocked
    if(length(hitMeshRadiance) <= 0.0f)
    {
        emissiveResult.mReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        emissiveResult.mRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    }

    out.emissiveRadiance = vec4<f32>(emissiveResult.mRadiance.xyz * emissiveResult.mReservoir.w, 1.0f);
    out.emissiveReservoir = emissiveResult.mReservoir;
    out.emissiveHitPosition = emissiveResult.mHitPosition;
    out.emissiveHitNormal = emissiveResult.mHitNormal;

    return out;
}

/////
fn getMeshEmissiveRadiance(
    iHitTriangleIndex: i32
) -> vec3<f32>
{
    var iMeshIndex: u32 = UINT32_MAX;
    for(var i: u32 = 0; i < defaultUniformData.miNumMeshes; i++)
    {
        let iStartTriangleIndex: i32 = i32(meshTriangleRangeData[i].x);
        let iEndTriangleIndex: i32 = i32(meshTriangleRangeData[i].y);
        if(iHitTriangleIndex >= iStartTriangleIndex && 
            iHitTriangleIndex <= iEndTriangleIndex)
        {
            iMeshIndex = i;
            break;
        }
    }

    var retRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    if(iMeshIndex < defaultUniformData.miNumMeshes)
    {
        retRadiance = materialData[iMeshIndex * 2 + 1].xyz; 
    }

    return retRadiance;
}

/////
fn temporalRestirEmissiveLight(
    result: EmissiveLightResult,
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    sampleRayDirection: vec3<f32>,
    hitPosition: vec3<f32>,
    fHitTriangleIndex: f32,
    fJacobian: f32,
    randomResult: RandomResult) -> EmissiveLightResult
{
    var ret: EmissiveLightResult = result;
    ret.mRandomResult = randomResult;
    let fMaxTemporalReservoirSamples: f32 = 8.0f;

    // check if hit any triangle, update hit triangle emissive value 
    // check if any triangle is hit
    let iHitTriangleIndex: i32 = i32(floor(fHitTriangleIndex + 0.5f)) - 1;
    
    // get the emissive radiance from mesh material
    var sampleRadiance: vec3<f32> = getMeshEmissiveRadiance(iHitTriangleIndex);
    let fDP: f32 = max(dot(normal, sampleRayDirection), 0.0f);
    let worldToHitPosition: vec3<f32> = hitPosition - worldPosition.xyz;
    let fDistanceSquared: f32 = max(dot(worldToHitPosition, worldToHitPosition), 1.0f);
    sampleRadiance = ((sampleRadiance * fDP * fJacobian) / fDistanceSquared);
    
    ret.mRandomResult = nextRand(ret.mRandomResult.miSeed);
    let fRand: f32 = ret.mRandomResult.mfNum;

    let fLuminance: f32 = computeLuminance(sampleRadiance);
    
    var fM: f32 = 1.0f;
    if(length(sampleRadiance.xyz) <= 0.0f)
    {
        fM = 0.01f;
    }

    // update reservoir to determine if radiance is updated
    var updateResult: ReservoirResult = updateReservoir(
        ret.mReservoir,
        fLuminance,
        fM,
        fRand);
    if(updateResult.mbExchanged)
    {
        ret.mRadiance = vec4<f32>(sampleRadiance, fHitTriangleIndex);
        ret.mHitPosition = vec4<f32>(hitPosition.x, hitPosition.y, hitPosition.z, fHitTriangleIndex);
        
        let invRayDirection: vec3<f32> = normalize(worldToHitPosition) * -1.0f;
        ret.mHitNormal = vec4<f32>(invRayDirection.x, invRayDirection.y, invRayDirection.z, 1.0f);
    }

    // clamp reservoir
    if(updateResult.mReservoir.z > 100.0f)
    {
        let fPct: f32 = 100.0f / updateResult.mReservoir.z;
        updateResult.mReservoir.x *= fPct;
        updateResult.mReservoir.z = 100.0f;
    }
    
    // weight
    updateResult.mReservoir.w = clamp(updateResult.mReservoir.x / max(updateResult.mReservoir.z * updateResult.mReservoir.y, 0.001f), 0.05f, 1.0f);
    
    // update to new reservoir
    ret.mReservoir = updateResult.mReservoir;
    
    ret.mbExchanged = updateResult.mbExchanged;

    return ret;
}

/////
fn updateReservoir(
    reservoir: vec4<f32>,
    fPHat: f32,
    fUpdateSpeedMultiplier: f32,
    fRand: f32) -> ReservoirResult
{
    var ret: ReservoirResult;
    ret.mReservoir = reservoir;
    ret.mbExchanged = false;

    ret.mReservoir.z += fUpdateSpeedMultiplier;
    var fWeight: f32 = fPHat; // * ret.mReservoir.z * ret.mReservoir.w;
    ret.mReservoir.x += fWeight;
    var fWeightPct: f32 = fWeight / ret.mReservoir.x;

    if(fRand < fWeightPct || reservoir.z <= 0.0f)
    {
        ret.mReservoir.y = fPHat;
        ret.mbExchanged = true;
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
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
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
    //return !(iMesh == iPrevMesh && fCheckWorldPositionDistance <= 0.00025f && fCheckDP >= 0.99f);
    return !(iMesh == iPrevMesh && fPrevPlaneDistance <= 0.005f && fCheckDP >= 0.99f);
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
fn intersectTri4(
    ray: Ray,
    iTriangleIndex: u32) -> RayTriangleIntersectionResult
{
    let iIndex0: u32 = aiDynamicTriangleIndices[iTriangleIndex * 3];
    let iIndex1: u32 = aiDynamicTriangleIndices[iTriangleIndex * 3 + 1];
    let iIndex2: u32 = aiDynamicTriangleIndices[iTriangleIndex * 3 + 2];

    let pos0: vec4<f32> = aDynamicVertexPositions[iIndex0];
    let pos1: vec4<f32> = aDynamicVertexPositions[iIndex1];
    let pos2: vec4<f32> = aDynamicVertexPositions[iIndex2];

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
    var aiStack: array<u32, 256>;
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

        let node: BVHNode2 = aDynamicBVHNodes[iNodeIndex];
        if(node.miPrimitiveID != UINT32_MAX)
        {
            let intersectionInfo: RayTriangleIntersectionResult = intersectTri4(
                ray,
                node.miPrimitiveID);

            if(abs(intersectionInfo.mIntersectPosition.x) < 10000.0f)
            {
                let fDistanceToEye: f32 = length(intersectionInfo.mIntersectPosition.xyz - ray.mOrigin.xyz);
                //if(fDistanceToEye < fClosestDistance)
                {
                    
                    //fClosestDistance = fDistanceToEye;
                    ret.mHitPosition = intersectionInfo.mIntersectPosition.xyz;
                    ret.mHitNormal = intersectionInfo.mIntersectNormal.xyz;
                    ret.miHitTriangle = node.miPrimitiveID;
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
                node.mMinBound.xyz,
                node.mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = node.miChildren0;
                iStackTop += 1;
                aiStack[iStackTop] = node.miChildren1;
            }
        }
    }

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
    let oneOverRay: vec3<f32> = 1.0f / rayDir.xyz;
    let tMin: vec3<f32> = (bboxMin - rayPosition) * oneOverRay;
    let tMax: vec3<f32> = (bboxMax - rayPosition) * oneOverRay;

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