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
    mCandidateRadiance: vec4<f32>,
};

struct IrradianceCacheEntry
{
    mPosition:              vec4<f32>,
    mImageProbe:            array<vec4<f32>, 64>,            // 8x8 image probe
};

struct Material
{
    mDiffuse: vec4<f32>,
    mSpecular: vec4<f32>,
    mEmissive: vec4<f32>,

    miID: u32,
    miAlbedoTextureID: u32,
    miNormalTextureID: u32,
    miEmissiveTextureID: u32,
};

struct SphericalHarmonicCoefficients
{
    mCoCg: vec2<f32>,
    mY: vec4<f32>,
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
var prevSpecularRadianceTexture: texture_2d<f32>;

@group(0) @binding(3)
var prevSpecularReservoirTexture: texture_2d<f32>;

@group(0) @binding(4)
var rayDirectionTexture: texture_2d<f32>;

@group(0) @binding(5)
var rayHitPositionTexture: texture_2d<f32>;

@group(0) @binding(6)
var hitPositionTexture: texture_2d<f32>;

@group(0) @binding(7)
var hitNormalTexture: texture_2d<f32>;

@group(0) @binding(8)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(9)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(10)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(11)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(12)
var prevTotalRadianceTexture: texture_2d<f32>;

@group(0) @binding(13)
var skyTexture: texture_2d<f32>;

@group(0) @binding(14)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(15)
var hitUVAndMeshTexture: texture_2d<f32>;

@group(0) @binding(16)
var roughMetalTexture: texture_2d<f32>;

@group(0) @binding(17)
var<storage, read_write> sphericalHarmonicCoefficient0: array<vec4<f32>>;

@group(0) @binding(18)
var<storage, read_write> sphericalHarmonicCoefficient1: array<vec4<f32>>;

@group(0) @binding(19)
var<storage, read_write> sphericalHarmonicCoefficient2: array<vec4<f32>>;

@group(0) @binding(20)
var<storage, read> prevSphericalHarmonicCoefficient0: array<vec4<f32>>;

@group(0) @binding(21)
var<storage, read> prevSphericalHarmonicCoefficient1: array<vec4<f32>>;

@group(0) @binding(22)
var<storage, read> prevSphericalHarmonicCoefficient2: array<vec4<f32>>;

@group(0) @binding(23)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage> aMaterials: array<Material>;

@group(1) @binding(2)
var<storage> aMeshMaterialID: array<u32>;

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
    @location(0) specularOutput : vec4<f32>,
    @location(1) specularReservoir: vec4<f32>,
    @location(2) debugOutput: vec4<f32>,

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
        out.specularOutput = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        out.specularReservoir = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    let hitUVAndMesh: vec4<f32> = textureSample(
        hitUVAndMeshTexture,
        textureSampler,
        in.texCoord);

    let roughMetal: vec4<f32> = textureSample(
        roughMetalTexture,
        textureSampler,
        in.texCoord);

    let iMesh: u32 = u32(floor((worldPosition.w - fract(worldPosition.w)) + 0.5f)) - 1u;
    let specularResult: SpecularResult = temporalRestirSpecular(
        worldPosition.xyz,
        normal.xyz,
        in.texCoord,
        rayDirection.xyz,
        rayHitPosition.xyz,
        rayHitPosition.w,
        iMesh,
        hitUVAndMesh.xy,
        u32(hitUVAndMesh.z),
        max(roughMetal.y - 0.1f, 0.3f),
        randomResult);

    // ambient occlusion
    var ambientOcclusion: vec3<f32> = textureSample(
        ambientOcclusionTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    out.specularOutput = specularResult.mRadiance * specularResult.mReservoir.w * ambientOcclusion.z;
    out.specularReservoir = specularResult.mReservoir;
    out.debugOutput = specularResult.mCandidateRadiance;

    return out;
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
    hitUV: vec2<f32>,
    iHitMeshID: u32,
    fSpecular: f32,
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

    let diff: vec3<f32> = worldPosition - hitPosition;
    let fHitPositionDistance: f32 = dot(diff, diff);

    // radiance from skylight or hit mesh 
    var candidateRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    if(fHitPositionDistance > 1000.0f)
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
            
        }
        
        // apply distance attenuation
        let worldToHitPosition: vec3<f32> = hitPosition - worldPosition.xyz;
        let fDistanceSquared: f32 = max(dot(worldToHitPosition, worldToHitPosition), 1.0f);
        candidateRadiance = candidateRadiance / fDistanceSquared;
    }
    
    // apply specular cut-off angle
    let fSpecularCutOff: f32 = 0.5f;
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

    if(fRadianceDP >= 0.9f || updateResult.mbExchanged == true)
    {
        encodeToSphericalHarmonicCoefficients(
            candidateRadiance.xyz / fSpecularConeDP,
            sampleRayDirection,
            texCoord,
            prevScreenUV,
            i32(bDisoccluded)
        );
    }


    // PBR
    if(updateResult.mbExchanged)
    {   
        var specularLight: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        var kS: f32 = 0.0f;
        {
            let fRoughness: f32 = 0.6f;     // temp for now

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

            specularLight = specular * candidateRadiance * max(dot(normal.xyz, lightDir.xyz), 0.0f) * 1.0f;
            kS = 1.0f - fresnel.x;
        }

        ret.mRadiance = vec4<f32>(specularLight, kS);

        ret.mCandidateRadiance = vec4<f32>(candidateRadiance.xyz, 1.0f);
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
    
    ret.mRadiance.x *= updateResult.mReservoir.w;
    ret.mRadiance.y *= updateResult.mReservoir.w;
    ret.mRadiance.z *= updateResult.mReservoir.w;

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

/////
fn getPreviousScreenUV(
    screenUV: vec2<f32>) -> vec2<f32>
{
    var motionVector: vec2<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        screenUV).xy;
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
    let backProjectedScreenUV: vec2<f32> = inputTexCoord - motionVector.xy;

    return (backProjectedScreenUV.x < 0.0f || backProjectedScreenUV.x > 1.0 || backProjectedScreenUV.y < 0.0f || backProjectedScreenUV.y > 1.0f);
}

// https://media.contentapi.ea.com/content/dam/eacom/frostbite/files/gdc2018-precomputedgiobalilluminationinfrostbite.pdf
// http://orlandoaguilar.github.io/sh/spherical/harmonics/irradiance/map/2017/02/12/SphericalHarmonics.html
/////
fn encodeToSphericalHarmonicCoefficients(
    radiance: vec3<f32>,
    direction: vec3<f32>,
    texCoord: vec2<f32>,
    prevTexCoord: vec2<f32>,
    iDisoccluded: i32
) 
{
    let iPrevOutputX: i32 = i32(prevTexCoord.x * f32(defaultUniformData.miScreenWidth));
    let iPrevOutputY: i32 = i32(prevTexCoord.y * f32(defaultUniformData.miScreenHeight));
    let iPrevImageIndex: i32 = iPrevOutputY * defaultUniformData.miScreenWidth + iPrevOutputX;

    var SHCoefficent0: vec4<f32> = prevSphericalHarmonicCoefficient0[iPrevImageIndex];
    var SHCoefficent1: vec4<f32> = prevSphericalHarmonicCoefficient1[iPrevImageIndex];
    var SHCoefficent2: vec4<f32> = prevSphericalHarmonicCoefficient2[iPrevImageIndex];

    if(iDisoccluded >= 1)
    {
        SHCoefficent0 = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        SHCoefficent1 = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        SHCoefficent2 = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    }

    let afC: vec4<f32> = vec4<f32>(
        0.282095f,
        0.488603f,
        0.488603f,
        0.488603f
    );
    
    let A: vec4<f32> = vec4<f32>(
        0.886227f,
        1.023326f,
        1.023326f,
        1.023326f
    );

    // encode coefficients with direction
    let coefficient: vec4<f32> = vec4<f32>(
        afC.x * A.x,
        afC.y * direction.y * A.y,
        afC.z * direction.z * A.z,
        afC.w * direction.x * A.w
    );

    // encode with radiance
    var aResults: array<vec3<f32>, 4>;
    aResults[0] = radiance.xyz * coefficient.x;
    aResults[1] = radiance.xyz * coefficient.y;
    aResults[2] = radiance.xyz * coefficient.z;
    aResults[3] = radiance.xyz * coefficient.w;

    SHCoefficent0.x += aResults[0].x;
    SHCoefficent0.y += aResults[0].y;
    SHCoefficent0.z += aResults[0].z;
    SHCoefficent0.w += aResults[1].x;

    SHCoefficent1.x += aResults[1].y;
    SHCoefficent1.y += aResults[1].z;
    SHCoefficent1.z += aResults[2].x;
    SHCoefficent1.w += aResults[2].y;

    SHCoefficent2.x += aResults[2].z;
    SHCoefficent2.y += aResults[3].x;
    SHCoefficent2.z += aResults[3].y;
    SHCoefficent2.w += aResults[3].z;

    let iOutputX: i32 = i32(texCoord.x * f32(defaultUniformData.miScreenWidth));
    let iOutputY: i32 = i32(texCoord.y * f32(defaultUniformData.miScreenHeight));
    let iImageIndex: i32 = iOutputY * defaultUniformData.miScreenWidth + iOutputX;

    sphericalHarmonicCoefficient0[iImageIndex] = SHCoefficent0;
    sphericalHarmonicCoefficient1[iImageIndex] = SHCoefficent1;
    sphericalHarmonicCoefficient2[iImageIndex] = SHCoefficent2;
}