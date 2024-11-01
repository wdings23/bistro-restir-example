const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const PROBE_IMAGE_SIZE: u32 = 8u;

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
    mRandomResult: RandomResult,
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
var prevDirectSunLightRadianceTexture: texture_2d<f32>;

@group(0) @binding(3)
var prevDirectSunLightReservoirTexture: texture_2d<f32>;

@group(0) @binding(4)
var prevSpecularRadianceTexture: texture_2d<f32>;

@group(0) @binding(5)
var prevSpecularReservoirTexture: texture_2d<f32>;

@group(0) @binding(6)
var rayDirectionTexture: texture_2d<f32>;

@group(0) @binding(7)
var rayHitPositionTexture: texture_2d<f32>;

@group(0) @binding(8)
var hitPositionTexture: texture_2d<f32>;

@group(0) @binding(9)
var hitNormalTexture: texture_2d<f32>;

@group(0) @binding(10)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(11)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(12)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(13)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(14)
var prevTotalRadianceTexture: texture_2d<f32>;

@group(0) @binding(15)
var<storage, read> irradianceCache: array<IrradianceCacheEntry>;

@group(0) @binding(16)
var skyTexture: texture_2d<f32>;

@group(0) @binding(17)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(18)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> materialData: array<vec4<f32>, 32>;

@group(1) @binding(2)
var<storage, read> meshTriangleRangeData: array<vec2<u32>, 32>;

@group(1) @binding(3)
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
    @location(0) directSunOutput : vec4<f32>,
    @location(1) directSunReservoir: vec4<f32>,

    @location(2) specularOutput : vec4<f32>,
    @location(3) specularReservoir: vec4<f32>,

    @location(4) ambientOcclusionOutput: vec4<f32>,


    @location(5) debug: vec4<f32>,

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

    let hitPosition: vec3<f32> = textureSample(
        hitPositionTexture,
        textureSampler,
        in.texCoord).xyz;

    let rayHitPosition: vec4<f32> = textureSample(
        rayHitPositionTexture,
        textureSampler,
        in.texCoord);

    if(worldPosition.w <= 0.0f)
    {
        out.directSunOutput = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        out.directSunReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    let result: DirectSunLightResult = temporalRestirDirectSunLight(
        worldPosition.xyz,
        normal.xyz,
        in.texCoord,
        rayDirection,
        hitPosition.xyz,
        f32(rayHitPosition.w > 0.0f),
        randomResult);

    out.directSunOutput = vec4<f32>(result.mRadiance * result.mReservoir.w, 1.0f);
    out.directSunReservoir = result.mReservoir;

    let iMesh: u32 = u32(floor((worldPosition.w - fract(worldPosition.w)) + 0.5f)) - 1u;
    let specularResult: SpecularResult = temporalRestirSpecular(
        worldPosition.xyz,
        normal.xyz,
        in.texCoord,
        rayDirection.xyz,
        rayHitPosition.xyz,
        rayHitPosition.w,
        iMesh,
        randomResult);

    out.specularOutput = specularResult.mRadiance * specularResult.mReservoir.w;
    out.specularReservoir = specularResult.mReservoir;

    // ambient occlusion
    var ambientOcclusion: vec3<f32> = textureSample(
        ambientOcclusionTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    var fAO: f32 = 1.0f - (ambientOcclusion.x / ambientOcclusion.y); 
    fAO = smoothstep(0.0f, 1.0f, fAO);
    out.ambientOcclusionOutput = vec4<f32>(fAO, fAO, fAO, 1.0f);

var prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);
var bDisoccluded: bool = isDisoccluded2(in.texCoord, prevScreenUV);
out.debug = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
if(bDisoccluded || isPrevUVOutOfBounds(in.texCoord))
{
    out.debug = vec4<f32>(1.0f, 1.0f, 1.0f, 1.0f);
}

out.debug.w = materialData[iMesh * 2].w;

    return out;
}


/////
fn temporalRestirDirectSunLight(
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    texCoord: vec2<f32>,
    sampleRayDirection: vec4<f32>,
    hitPosition: vec3<f32>,
    fRayCollision: f32,
    randomResult: RandomResult) -> DirectSunLightResult
{
    var ret: DirectSunLightResult;
    ret.mRandomResult = randomResult;

    ret.mRadiance = vec3<f32>(0.0f, 0.0f, 0.0f);
    ret.mReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

    let fDirectSunLightConeAngle: f32 = defaultUniformData.mLightRadiance.w;
    let fMaxTemporalReservoirSamples: f32 = 8.0f;

    // use previous frame's reservoir and radiance if not disoccluded
    var prevScreenUV: vec2<f32> = getPreviousScreenUV(texCoord);
    var bDisoccluded: bool = isDisoccluded2(texCoord, prevScreenUV);
    ret.mRadiance = vec3<f32>(0.0f, 0.0f, 0.0f);
    ret.mReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    if(!bDisoccluded && !isPrevUVOutOfBounds(texCoord))
    {
        ret.mRadiance = textureSample(
            prevDirectSunLightRadianceTexture,
            textureSampler,
            prevScreenUV).xyz;
        
        ret.mReservoir = textureSample(
            prevDirectSunLightReservoirTexture,
            textureSampler,
            prevScreenUV);
    }

    //if(dot(hitPosition, hitPosition) >= 100.0)
    var fDirectSunConeDP: f32 = max(dot(sampleRayDirection.xyz, defaultUniformData.mLightDirection.xyz), 0.0f);
    if(fDirectSunConeDP >= 0.95f/* && fRayCollision < 1.0f*/)
    {
        fDirectSunConeDP = max(fDirectSunConeDP * (1.0f - fRayCollision), 0.0f);

        ret.mRandomResult = nextRand(ret.mRandomResult.miSeed);
        let fRand: f32 = ret.mRandomResult.mfNum;

        let SUN_DIRECTION_DISTANCE: f32 =                  6371000.0f;

        // luminance from the sun light
        let fDP: f32 = max(dot(defaultUniformData.mLightDirection.xyz, normal), 0.0f);
        var transmittance: vec3<f32>;
        let scatteringResult: ScatteringResult = IntegrateScattering(
            worldPosition.xyz,
            defaultUniformData.mLightDirection.xyz,
            SUN_DIRECTION_DISTANCE,
            defaultUniformData.mLightDirection.xyz,
            defaultUniformData.mLightRadiance.xyz,
            transmittance);
        var sampleRadiance: vec3<f32> = scatteringResult.radiance * scatteringResult.transmittance * fDirectSunConeDP * fDP * 0.1f;
        
        let fLuminance: f32 = computeLuminance(sampleRadiance);
        
        var fM: f32 = 1.0f;
        if(fRayCollision >= 1.0f)
        {
            fM = 0.005f;
        }

        // update reservoir to determine if radiance is updated
        var updateResult: ReservoirResult = updateReservoir(
            ret.mReservoir,
            fLuminance,
            fM,
            fRand);
        if(updateResult.mbExchanged)
        {
            ret.mRadiance = sampleRadiance;
        }

        // clamp reservoir
        if(updateResult.mReservoir.z > 12.0f)
        {
            let fPct: f32 = 12.0f / updateResult.mReservoir.z;
            updateResult.mReservoir.x *= fPct;
            updateResult.mReservoir.z = 12.0f;
        }
        
        // weight
        updateResult.mReservoir.w = clamp(updateResult.mReservoir.x / max(updateResult.mReservoir.z * updateResult.mReservoir.y, 0.001f), 0.1f, 1.0f);
        
        // update to new reservoir
        ret.mReservoir = updateResult.mReservoir;
    }

    return ret;
}

/////
fn temporalRestirSpecular(
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    texCoord: vec2<f32>,
    sampleRayDirection: vec3<f32>,
    hitPosition: vec3<f32>,
    fHitTriangleIndex: f32,
    iOriginMeshIndex: u32,
    randomResult: RandomResult) -> SpecularResult
{
    let fMaxTemporalReservoirSamples: f32 = 8.0f;

    var ret: SpecularResult;
    ret.mRandomResult = randomResult;

    ret.mRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    ret.mReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

    // use previous frame's reservoir and radiance if not disoccluded
    var prevScreenUV: vec2<f32> = getPreviousScreenUV(texCoord);
    var bDisoccluded: bool = isDisoccluded2(texCoord, prevScreenUV);
    if(!bDisoccluded && !isPrevUVOutOfBounds(texCoord))
    {
        ret.mRadiance = textureSample(
            prevSpecularRadianceTexture,
            textureSampler,
            prevScreenUV);
        
        ret.mReservoir = textureSample(
            prevSpecularReservoirTexture,
            textureSampler,
            prevScreenUV);
    }

    // check if any triangle is hit
    let iHitTriangleIndex: i32 = i32(floor(fHitTriangleIndex + 0.5f)) - 1;
    var iHitMeshIndex: u32 = UINT32_MAX;
    if(fHitTriangleIndex < 1.0e+4f && iHitTriangleIndex >= 0)
    {
        for(var i: u32 = 0; i < defaultUniformData.miNumMeshes; i++)
        {
            let iStartTriangleIndex: i32 = i32(meshTriangleRangeData[i].x);
            let iEndTriangleIndex: i32 = i32(meshTriangleRangeData[i].y);
            if(iHitTriangleIndex >= iStartTriangleIndex && 
                iHitTriangleIndex <= iEndTriangleIndex)
            {
                iHitMeshIndex = i;
                break;
            }
        }
    }

    // radiance from skylight or hit mesh 
    var candidateRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    if(fHitTriangleIndex >= 1.0e+4f || iHitTriangleIndex < 0)
    {
        // didn't hit anything, use skylight
        let skyUV: vec2<f32> = octahedronMap2(sampleRayDirection.xyz);
        candidateRadiance = textureSample(
            skyTexture,
            textureSampler,
            skyUV).xyz;
    } 
    else
    {
        // check if hit position is on screen
        // get on-screen radiance if there's any
        var clipSpacePosition: vec4<f32> = vec4<f32>(hitPosition.xyz, 1.0) * defaultUniformData.mViewProjectionMatrix;
        clipSpacePosition.x /= clipSpacePosition.w;
        clipSpacePosition.y /= clipSpacePosition.w;
        clipSpacePosition.z /= clipSpacePosition.w;
        clipSpacePosition.x = clipSpacePosition.x * 0.5f + 0.5f;
        clipSpacePosition.y = 1.0f - (clipSpacePosition.y * 0.5f + 0.5f);
        clipSpacePosition.z = clipSpacePosition.z * 0.5f + 0.5f;
        
        // compare depth at this screen position to see if it's indeed on screen or behind the showing pixel
        let worldSpaceHitPosition: vec4<f32> = textureSample(
            worldPositionTexture,
            textureSampler,
            clipSpacePosition.xy);
        var hitPositionClipSpace: vec4<f32> = vec4<f32>(worldSpaceHitPosition.xyz, 1.0f) * defaultUniformData.mViewProjectionMatrix;
        hitPositionClipSpace.x /= hitPositionClipSpace.w;
        hitPositionClipSpace.y /= hitPositionClipSpace.w;
        hitPositionClipSpace.z /= hitPositionClipSpace.w;
        hitPositionClipSpace.x = hitPositionClipSpace.x * 0.5f + 0.5f;
        hitPositionClipSpace.y = 1.0f - (hitPositionClipSpace.y * 0.5f + 0.5f);
        hitPositionClipSpace.z = hitPositionClipSpace.z * 0.5f + 0.5f;
        let fDepthDiff: f32 = hitPositionClipSpace.z - clipSpacePosition.z;

        var hitPositionRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        if(clipSpacePosition.x >= 0.0f && clipSpacePosition.x <= 1.0f &&
           clipSpacePosition.y >= 0.0f && clipSpacePosition.y <= 1.0f &&
           fDepthDiff >= 0.1f)
        {
            // on screen, use the previous frame's radiance

            // back project to previous frame's clip space coordinate
            var motionVector: vec2<f32> = textureSample(
                motionVectorTexture,
                textureSampler,
                texCoord).xy;
            motionVector = motionVector * 2.0f - 1.0f;
            clipSpacePosition.x -= motionVector.x;
            clipSpacePosition.y -= motionVector.y;

            candidateRadiance = textureSample(
                prevTotalRadianceTexture,
                textureSampler,
                clipSpacePosition.xy).xyz;
        }
        else 
        {
            // not on-screen, check irradiance cache

            // emissive material of hit mesh
            var sampleRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
            if(iHitMeshIndex < defaultUniformData.miNumMeshes)
            {
                let emissiveRadiance: vec3<f32> = materialData[iHitMeshIndex * 2 + 1].xyz;
                candidateRadiance += emissiveRadiance;
            }

            let iHitIrradianceCacheIndex: u32 = fetchIrradianceCacheIndex(hitPosition);
            let irradianceCachePosition: vec4<f32> = getIrradianceCachePosition(iHitIrradianceCacheIndex);
            if(irradianceCachePosition.w > 0.0f)
            {
                let positionToCacheDirection: vec3<f32> = normalize(irradianceCachePosition.xyz - worldPosition) * -1.0f; 
                candidateRadiance += getRadianceFromIrradianceCacheProbe(
                    positionToCacheDirection, 
                    iHitIrradianceCacheIndex);
            }
        }
        
        // apply distance attenuation
        let worldToHitPosition: vec3<f32> = hitPosition - worldPosition.xyz;
        let fDistanceSquared: f32 = max(dot(worldToHitPosition, worldToHitPosition), 1.0f);
        candidateRadiance = candidateRadiance / fDistanceSquared;
    }
        
    // apply specular cut-off angle
    let fSpecularCutOff: f32 = 0.7f;
    let fSpecular: f32 = materialData[iOriginMeshIndex * 2].w;
    var fSampleConeThreshold: f32 = max(fSpecular, fSpecularCutOff);
    let fRadianceDP: f32 = clamp(dot(normal, sampleRayDirection), 0.0f, 1.0f);
    let fSpecularConeDP: f32 = ceil(max(fRadianceDP - fSampleConeThreshold, 0.0f)) * fRadianceDP; 

    ret.mRandomResult = nextRand(ret.mRandomResult.miSeed);
    let fRand: f32 = ret.mRandomResult.mfNum;

    candidateRadiance *= fSpecularConeDP;

    let fLuminance: f32 = computeLuminance(candidateRadiance);
    
    // update reservoir to determine if radiance is updated
    var updateResult: ReservoirResult;
    updateResult.mReservoir = ret.mReservoir;
    updateResult.mbExchanged = false;
    if(fLuminance > 0.0f)
    {
        updateResult = updateReservoir(
            ret.mReservoir,
            fLuminance,
            1.0f,
            fRand);
    }
    if(updateResult.mbExchanged)
    {   
        var specularLight: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        {
            let fRoughness: f32 = 0.3f;     // temp for now

            let viewDir: vec3<f32> = normalize(defaultUniformData.mCameraPosition.xyz - worldPosition.xyz);

            var F0: vec3<f32> = vec3<f32>(0.04f, 0.04f, 0.04f);
            F0 = mix(F0, vec3<f32>(1.0f, 1.0f, 1.0f), fSpecular);

            // calculate per-light radiance
            let lightDir: vec3<f32> = normalize(hitPosition.xyz - worldPosition.xyz);
            let halfV: vec3<f32> = normalize(viewDir + lightDir);

            // cook-torrance brdf
            let NDF: f32 = distributionGGX(normal.xyz, halfV, fRoughness);
            let geometry: f32 = geometrySmith(normal.xyz, viewDir, lightDir, fRoughness);
            let fresnel: vec3<f32> = fresnelShlickRoughness(max(dot(halfV, viewDir), 0.0f), F0, fRoughness);

            let numerator: vec3<f32> = fresnel * (NDF * geometry);
            let denominator: f32 = 4.0f * max(dot(normal.xyz, viewDir), 0.0f) * max(dot(normal.xyz, lightDir), 0.0f) + 0.0001f;
            let specular: vec3<f32> = clamp(
                numerator / denominator, 
                vec3<f32>(0.0f, 0.0f, 0.0f), 
                vec3<f32>(1.0f, 1.0f, 1.0f));

            specularLight = specular * candidateRadiance;
        }

        ret.mRadiance = vec4<f32>(specularLight, fHitTriangleIndex);
    }

    // clamp reservoir
    if(updateResult.mReservoir.z > fMaxTemporalReservoirSamples)
    {
        let fPct: f32 = fMaxTemporalReservoirSamples / updateResult.mReservoir.z;
        updateResult.mReservoir.x *= fPct;
        updateResult.mReservoir.z = fMaxTemporalReservoirSamples;
    }
    
    // weight, which tends to be really small due to not have separate sampling function such as VNDF instead of just uniform sampling
    updateResult.mReservoir.w = clamp(updateResult.mReservoir.x / max(updateResult.mReservoir.z * updateResult.mReservoir.y, 0.001f), 0.1f, 1.0f);
    
    //ret.mRadiance.x *= updateResult.mReservoir.w;
    //ret.mRadiance.y *= updateResult.mReservoir.w;
    //ret.mRadiance.z *= updateResult.mReservoir.w;

    // update to new reservoir
    ret.mReservoir = updateResult.mReservoir;
    
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

    ret.mReservoir.x += fPHat;
    ret.mReservoir.z += fUpdateSpeedMultiplier;
    var fWeightPct: f32 = fPHat / ret.mReservoir.x;

    if(fRand < fWeightPct/* || reservoir.z <= 0.0f*/)
    {
        ret.mReservoir.y = fPHat;
        ret.mbExchanged = true;
    }

    return ret;
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
fn murmurHash13(
    src: vec3<u32>) -> u32
{
    var srcCopy: vec3<u32> = src;
    var M: u32 = 0x5bd1e995u;
    var h: u32 = 1190494759u;
    srcCopy *= M; srcCopy.x ^= srcCopy.x >> 24u; srcCopy.y ^= srcCopy.y >> 24u; srcCopy.z ^= srcCopy.z >> 24u; srcCopy *= M;
    h *= M; h ^= srcCopy.x; h *= M; h ^= srcCopy.y; h *= M; h ^= srcCopy.z;
    h ^= h >> 13u; h *= M; h ^= h >> 15u;
    return h;
}

/////
fn hash13(
    src: vec3<f32>,
    iNumSlots: u32) -> u32
{
    let srcU32: vec3<u32> = vec3<u32>(
        bitcast<u32>(src.x),
        bitcast<u32>(src.y),
        bitcast<u32>(src.z)
    );

    let h: u32 = u32(murmurHash13(srcU32));
    var fValue: f32 = bitcast<f32>((h & 0x007ffffffu) | 0x3f800000u) - 1.0f;
    let iRet: u32 = clamp(u32(fValue * f32(iNumSlots - 1)), 0u, iNumSlots - 1);
    return iRet;
}

/////
fn fetchIrradianceCacheIndex(
    position: vec3<f32>
) -> u32
{
    var scaledPosition: vec3<f32> = position * 10.0f;
    let iHashKey: u32 = hash13(
        scaledPosition,
        5000u
    );

    return iHashKey;
}

/////
fn getRadianceFromIrradianceCacheProbe(
    rayDirection: vec3<f32>,
    iIrradianceCacheIndex: u32
) -> vec3<f32>
{
    if(irradianceCache[iIrradianceCacheIndex].mPosition.w == 0.0f)
    {
        return vec3<f32>(0.0f, 0.0f, 0.0f);
    }

    let probeImageUV: vec2<f32> = octahedronMap2(rayDirection);
    var iImageY: u32 = clamp(u32(probeImageUV.y * f32(PROBE_IMAGE_SIZE)), 0u, PROBE_IMAGE_SIZE - 1u);
    var iImageX: u32 = clamp(u32(probeImageUV.x * f32(PROBE_IMAGE_SIZE)), 0u, PROBE_IMAGE_SIZE - 1u);
    var iImageIndex: u32 = iImageY * PROBE_IMAGE_SIZE + iImageX;

    return irradianceCache[iIrradianceCacheIndex].mImageProbe[iImageIndex].xyz;
}

/////
fn getIrradianceCachePosition(
    iIrradianceCacheIndex: u32
) -> vec4<f32>
{
    return irradianceCache[iIrradianceCacheIndex].mPosition;
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
fn distributionGGX(
    N: vec3<f32>,
    H: vec3<f32>,
    roughness: f32) -> f32
{
    let a: f32 = roughness * roughness;
    let a2: f32 = a * a;
    let NdotH: f32 = max(dot(N, H), 0.0f);
    let NdotH2: f32 = NdotH * NdotH;

    let num: f32 = a2;
    var denom: f32 = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;

    return num / denom;
}

/////
fn geometrySchlickGGX(
    NdotV: f32,
    roughness: f32) -> f32
{
    let r: f32 = (roughness + 1.0f);
    let k: f32 = (r * r) / 8.0f;

    let num: f32 = NdotV;
    let denom: f32 = NdotV * (1.0f - k) + k;

    return num / denom;
}


/////
fn geometrySmith(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    roughness: f32) -> f32
{
    let NdotV: f32 = max(dot(N, V), 0.0f);
    let NdotL: f32 = max(dot(N, L), 0.0f);
    let ggx2: f32 = geometrySchlickGGX(NdotV, roughness);
    let ggx1: f32 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

/////
fn fresnelShlickRoughness(
    fCosTheta: f32,
    f0: vec3<f32>,
    fRoughness: f32) -> vec3<f32>
{
    let fOneMinusRoughness: f32 = 1.0f - fRoughness;
    let fClamp: f32 = clamp(1.0f - fCosTheta, 0.0f, 1.0f);
    return f0 + (max(vec3<f32>(fOneMinusRoughness, fOneMinusRoughness, fOneMinusRoughness), f0) - f0) * pow(fClamp, 5.0f);
}






// -------------------------------------
// Defines
const EPS: f32 =               1e-6f;
const PLANET_RADIUS: f32 =       6371000.0f;
const PLANET_CENTER: vec3<f32> =       vec3<f32>(0.0f, -PLANET_RADIUS, 0.0f);
const ATMOSPHERE_HEIGHT: f32 =   100000.0f;
const RAYLEIGH_HEIGHT: f32 =     (ATMOSPHERE_HEIGHT * 0.08f);
const MIE_HEIGHT: f32 =          (ATMOSPHERE_HEIGHT * 0.012f);

// -------------------------------------
// Coefficients
const C_RAYLEIGH: vec3<f32> =          (vec3<f32>(5.802f, 13.558f, 33.100f) * 1e-6f);
const C_MIE: vec3<f32> =               (vec3<f32>(3.996f,  3.996f,  3.996f) * 1e-6f);
const C_OZONE: vec3<f32> =             (vec3<f32>(0.650f,  1.881f,  0.085f) * 1e-6f);

const ATMOSPHERE_DENSITY: f32 =  1.0f;
const EXPOSURE: f32 =            20.0f;

// -------------------------------------
// Math
fn SphereIntersection(
	rayStart: vec3<f32>, 
	rayDir: vec3<f32>, 
	sphereCenter: vec3<f32>, 
	sphereRadius: f32) -> vec2<f32>
{
	let rayStartDiff: vec3<f32> = rayStart - sphereCenter;
	let a: f32 = dot(rayDir, rayDir);
	let b: f32 = 2.0f * dot(rayStartDiff, rayDir);
	let c: f32 = dot(rayStartDiff, rayStartDiff) - (sphereRadius * sphereRadius);
	var d: f32 = b * b - 4.0f * a * c;
	if(d < 0)
	{
		return vec2<f32>(-1.0f, -1.0f);
	}
	else
	{
		d = sqrt(d);
		return vec2<f32>(-b - d, -b + d) / (2.0f * a);
	}
}
fn PlanetIntersection(
	rayStart: vec3<f32>, 
	rayDir: vec3<f32>) -> vec2<f32>
{
	return SphereIntersection(rayStart, rayDir, PLANET_CENTER, PLANET_RADIUS);
}
fn AtmosphereIntersection(
	rayStart: vec3<f32>, 
	rayDir: vec3<f32>) -> vec2<f32>
{
	return SphereIntersection(rayStart, rayDir, PLANET_CENTER, PLANET_RADIUS + ATMOSPHERE_HEIGHT);
}

// -------------------------------------
// Phase functions
fn PhaseRayleigh(costh: f32) -> f32
{
	return 3.0f * (1.0f + costh * costh) / (16.0f * PI);
}
fn PhaseMie(
    costh: f32, 
    g: f32) -> f32
{
	let gCopy: f32 = min(g, 0.9381f);
	let k: f32 = 1.55f * gCopy - 0.55f * gCopy * gCopy * gCopy;
	let kcosth: f32 = k * costh;
	return (1.0f - k * k) / ((4.0f * PI) * (1.0f - kcosth) * (1.0f - kcosth));
}

// -------------------------------------
// Atmosphere
fn AtmosphereHeight(positionWS: vec3<f32>) -> f32
{
	//return distance(positionWS, PLANET_CENTER) - PLANET_RADIUS;

	return length(positionWS - PLANET_CENTER) - PLANET_RADIUS;
}
fn DensityRayleigh(h: f32) -> f32
{
	return exp(-max(0.0f, h / RAYLEIGH_HEIGHT));
}
fn DensityMie(h: f32) -> f32
{
	return exp(-max(0.0f, h / MIE_HEIGHT));
}
fn DensityOzone(h: f32) -> f32
{
	// The ozone layer is represented as a tent function with a width of 30km, centered around an altitude of 25km.
	return max(0.0f, 1.0f - abs(h - 25000.0f) / 15000.0f);
}
fn AtmosphereDensity(h: f32) -> vec3<f32>
{
	return vec3<f32>(DensityRayleigh(h), DensityMie(h), DensityOzone(h));
}

// Optical depth is a unitless measurement of the amount of absorption of a participating medium (such as the atmosphere).
// This function calculates just that for our three atmospheric elements:
// R: Rayleigh
// G: Mie
// B: Ozone
// If you find the term "optical depth" confusing, you can think of it as "how much density was found along the ray in total".
fn IntegrateOpticalDepth(
    rayStart: vec3<f32>, 
    rayDir: vec3<f32>) -> vec3<f32>
{
	let intersection: vec2<f32> = AtmosphereIntersection(rayStart, rayDir);
	let  rayLength: f32 = intersection.y;

	let    sampleCount: i32 = 8;
	let  stepSize: f32 = rayLength / f32(sampleCount);

	var opticalDepth: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);

	for(var i: i32 = 0; i < sampleCount; i++)
	{
		let localPosition: vec3<f32> = rayStart + rayDir * (f32(i) + 0.5f) * stepSize;
		let  localHeight: f32 = AtmosphereHeight(localPosition);
		let localDensity: vec3<f32> = AtmosphereDensity(localHeight);

		opticalDepth += localDensity * stepSize;
	}

	return opticalDepth;
}

// Calculate a luminance transmittance value from optical depth.
fn Absorb(opticalDepth: vec3<f32>) -> vec3<f32>
{
	// Note that Mie results in slightly more light absorption than scattering, about 10%
	let ret: vec3<f32> = ((C_RAYLEIGH * opticalDepth.x) + (C_MIE * (opticalDepth.y * 1.1f)) + (C_OZONE * opticalDepth.z)) * ATMOSPHERE_DENSITY * -1.0f;
	return vec3<f32>(
		exp(ret.x),
		exp(ret.y),
		exp(ret.z));

}

struct ScatteringResult
{
    radiance: vec3<f32>,
    transmittance: vec3<f32>,
};

// Integrate scattering over a ray for a single directional light source.
// Also return the transmittance for the same ray as we are already calculating the optical depth anyway.
fn IntegrateScattering(
	rayStart: vec3<f32>, 
	rayDir: vec3<f32>, 
	rayLength: f32, 
	lightDir: vec3<f32>, 
	lightColor: vec3<f32>, 
	transmittance: vec3<f32>) -> ScatteringResult
{
	// We can reduce the number of atmospheric samples required to converge by spacing them exponentially closer to the camera.
	// This breaks space view however, so let's compensate for that with an exponent that "fades" to 1 as we leave the atmosphere.
	var rayStartCopy: vec3<f32> = rayStart;
    let  rayHeight: f32 = AtmosphereHeight(rayStartCopy);
	let  sampleDistributionExponent: f32 = 1.0f + clamp(1.0f - rayHeight / ATMOSPHERE_HEIGHT, 0.0f, 1.0f) * 8; // Slightly arbitrary max exponent of 9

	let intersection: vec2<f32> = AtmosphereIntersection(rayStartCopy, rayDir);
	var rayLengthCopy: f32 = min(rayLength, intersection.y);
	if(intersection.x > 0)
	{
		// Advance ray to the atmosphere entry point
		rayStartCopy += rayDir * intersection.x;
		rayLengthCopy -= intersection.x;
	}

	let  costh: f32 = dot(rayDir, lightDir);
	let  phaseR: f32 = PhaseRayleigh(costh);
	let  phaseM: f32 = PhaseMie(costh, 0.85f);

	let    sampleCount: i32 = 64;

	var opticalDepth: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
	var rayleigh: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
	var mie: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);

	var  prevRayTime: f32 = 0.0f;

	for(var i: i32 = 0; i < sampleCount; i++)
	{
		let  rayTime: f32 = pow(f32(i) / f32(sampleCount), sampleDistributionExponent) * rayLengthCopy;
		// Because we are distributing the samples exponentially, we have to calculate the step size per sample.
		let  stepSize: f32 = (rayTime - prevRayTime);

		let localPosition: vec3<f32> = rayStartCopy + rayDir * rayTime;
		let  localHeight: f32 = AtmosphereHeight(localPosition);
		let localDensity: vec3<f32> = AtmosphereDensity(localHeight);

		opticalDepth += localDensity * stepSize;

		// The atmospheric transmittance from rayStart to localPosition
		let viewTransmittance: vec3<f32> = Absorb(opticalDepth);

		let opticalDepthlight: vec3<f32> = IntegrateOpticalDepth(localPosition, lightDir);
		// The atmospheric transmittance of light reaching localPosition
		let lightTransmittance: vec3<f32> = Absorb(opticalDepthlight);

		rayleigh += viewTransmittance * lightTransmittance * phaseR * localDensity.x * stepSize;
		mie += viewTransmittance * lightTransmittance * phaseM * localDensity.y * stepSize;

		prevRayTime = rayTime;
	}

    var ret: ScatteringResult;
	ret.transmittance = Absorb(opticalDepth);
    ret.radiance = (rayleigh * C_RAYLEIGH + mie * C_MIE) * lightColor * EXPOSURE;

	return ret;
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