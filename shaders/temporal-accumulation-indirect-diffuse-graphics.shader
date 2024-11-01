const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const kfMaxBlendFrames = 60.0f;
const kfOneOverMaxBlendFrames: f32 = 1.0f / kfMaxBlendFrames;

const INDIRECT_DIFFUSE: i32 = 0;

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
    @location(0) indirectDiffuseOutput: vec4<f32>,
    @location(1) indirectDiffuseMomentOutput: vec4<f32>,
    @location(2) emissiveOutput: vec4<f32>,
    @location(3) directSunOutput: vec4<f32>,
    @location(4) specularOutput: vec4<f32>,
    @location(5) specularMomentOutput: vec4<f32>,
    @location(6) debugOutput: vec4<f32>,
};

struct RandomResult 
{
    mfNum: f32,
    miSeed: u32,
};

struct UniformData
{
    miFrame: u32,
    miScreenWidth: u32,
    miScreenHeight: u32,
    mfRand: f32,

    maSamples: array<vec4<f32>, 64>,

    mfRand0: f32,
    mfRand1: f32,
    mfRand2: f32,
    mfRand3: f32,

};

struct DisocclusionResult
{
    mBackProjectScreenCoord: vec2<i32>,
    mBackProjectScreenUV: vec2<f32>,
    mbDisoccluded: bool,
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
var indirectDiffuseRadianceTexture: texture_2d<f32>;

@group(0) @binding(3)
var emissiveRadianceTexture: texture_2d<f32>;

@group(0) @binding(4)
var directSunRadianceTexture: texture_2d<f32>;

@group(0) @binding(5)
var specularRadianceTexture: texture_2d<f32>;

@group(0) @binding(6)
var indirectDiffuseRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(7)
var indirectDiffuseMomentHistoryTexture: texture_2d<f32>;

@group(0) @binding(8)
var emissiveRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(9)
var directSunRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(10)
var specularRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(11)
var specularMomentHistoryTexture: texture_2d<f32>;

@group(0) @binding(12)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(13)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(14)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(15)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(16)
var indirectDiffuseTemporalReservoirTexture: texture_2d<f32>;

@group(0) @binding(17)
var emissiveTemporalReservoirTexture: texture_2d<f32>;

@group(0) @binding(18)
var directTemporalReservoirTexture: texture_2d<f32>;

@group(0) @binding(19)
var specularTemporalReservoirTexture: texture_2d<f32>;

@group(0) @binding(20)
var rayCountTexture: texture_2d<f32>;

@group(0) @binding(21)
var skinAmbientOcclusionScreenSpaceTexture: texture_2d<f32>;

@group(0) @binding(22)
var indirectDiffuseScreenSpaceTexture: texture_2d<f32>;

@group(0) @binding(23)
var skinClipSpaceTexture: texture_2d<f32>;

@group(0) @binding(24)
var indirectDiffuseSpatialReservoirTexture: texture_2d<f32>;

@group(0) @binding(25)
var indirectDiffuseMIPTexture0: texture_2d<f32>;

@group(0) @binding(26)
var indirectDiffuseMIPTexture1: texture_2d<f32>;

@group(0) @binding(27)
var indirectDiffuseMIPTexture2: texture_2d<f32>;

@group(0) @binding(28)
var indirectDiffuseMIPTexture3: texture_2d<f32>;

@group(0) @binding(29)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> materialData: array<vec4<f32>, 32>;

@group(1) @binding(2)
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
        out.indirectDiffuseOutput = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        out.indirectDiffuseMomentOutput = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    var indirectDiffuseRadiance: vec3<f32> = textureSample(
        indirectDiffuseRadianceTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    var emissiveRadiance: vec3<f32> = textureSample(
        emissiveRadianceTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    var directSunRadiance: vec3<f32> = textureSample(
        directSunRadianceTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    var specularRadiance: vec3<f32> = textureSample(
        specularRadianceTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    

    // previous radiance
    let prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);
    //let indirectDiffuseRadianceHistory: vec4<f32> = textureSample(
    //    indirectDiffuseRadianceHistoryTexture,
    //    textureSampler,
    //    prevScreenUV                    
    //);

    let afWeights: array<f32, 4> = array<f32, 4>(1.0f, 1.0f, 1.0f, 1.0f);
    let indirectDiffuseRadianceHistory: vec4<f32> = bilinearSample(
        prevScreenUV,
        0,
        afWeights
    );

    let emissiveRadianceHistory: vec4<f32> = textureSample(
        emissiveRadianceHistoryTexture,
        textureSampler,
        prevScreenUV                    
    );
    let directSunRadianceHistory: vec4<f32> = textureSample(
        directSunRadianceHistoryTexture,
        textureSampler,
        prevScreenUV                    
    );
    let specularRadianceHistory: vec4<f32> = textureSample(
        specularRadianceHistoryTexture,
        textureSampler,
        prevScreenUV                    
    );

    let indirectDiffuseTemporalReservoir: vec4<f32> = textureSample(
        indirectDiffuseTemporalReservoirTexture,
        textureSampler,
        in.texCoord
    );
    let emissiveTemporalReservoir: vec4<f32> = textureSample(
        emissiveTemporalReservoirTexture,
        textureSampler,
        in.texCoord
    );
    let directTemporalReservoir: vec4<f32> = textureSample(
        directTemporalReservoirTexture,
        textureSampler,
        in.texCoord
    );
    let specularTemporalReservoir: vec4<f32> = textureSample(
        specularTemporalReservoirTexture,
        textureSampler,
        in.texCoord
    );

    var fIndirectDiffuseReservoirWeight: f32 = indirectDiffuseTemporalReservoir.w;
    var fEmissiveReservoirWeight: f32 = emissiveTemporalReservoir.w;
    var fDirectSunReservoirWeight: f32 = directTemporalReservoir.w;
    var fSpecularReservoirWeight: f32 = specularTemporalReservoir.w;

    // small weight right after disocclusion, just apply multiplier to avoid being too dark
    if(indirectDiffuseRadianceHistory.w < 20.0f)
    {
        fIndirectDiffuseReservoirWeight = clamp(indirectDiffuseTemporalReservoir.w * 5.0f, 0.5f, 1.0f);
        fEmissiveReservoirWeight = clamp(emissiveTemporalReservoir.w * 5.0f, 0.5f, 1.0f);
        fDirectSunReservoirWeight = clamp(directTemporalReservoir.w * 5.0f, 0.5f, 1.0f);
        fSpecularReservoirWeight = clamp(specularTemporalReservoir.w * 5.0f, 0.5f, 1.0f);
    }

    let fLuminance: f32 = computeLuminance(indirectDiffuseRadiance.xyz);
    let indirectDiffuseMoment: vec4<f32> = vec4<f32>(
        fLuminance,
        fLuminance * fLuminance,
        0.0f, 
        0.0f
    );

    let fSpecularLuminance: f32 = computeLuminance(specularRadiance.xyz);
    let specularMoment: vec4<f32> = vec4<f32>(
        fSpecularLuminance,
        fSpecularLuminance * fSpecularLuminance,
        0.0f, 
        0.0f
    );

    // ray counts for ambient occlusion
    let rayCount: vec4<f32> = textureSample(
        rayCountTexture,
        textureSampler,
        prevScreenUV
    );

    // ambient occlusion based on static or skinned mesh
    var fAO: f32 = smoothstep(0.0f, 1.0f, smoothstep(0.0f, 1.0f, rayCount.z));
    let fSkinMesh: f32 = textureSample(
        skinClipSpaceTexture,
        textureSampler,
        in.texCoord
    ).w;

    // disoccluded
    let bDisoccluded: bool = isDisoccluded2(in.texCoord, prevScreenUV);
    if(bDisoccluded)
    {
        indirectDiffuseRadiance = textureSample(
            indirectDiffuseMIPTexture3,
            textureSampler,
            in.texCoord
        ).xyz;

        out.indirectDiffuseOutput = vec4<f32>(indirectDiffuseRadiance, 0.0f) * fAO;
        out.indirectDiffuseMomentOutput = indirectDiffuseMoment;

        out.emissiveOutput = vec4<f32>(emissiveRadiance, 0.0f) * fAO;
        out.directSunOutput = vec4<f32>(directSunRadiance, 0.0f) * fAO;
        out.specularOutput = vec4<f32>(specularRadiance, 0.0f) * fAO;

        out.debugOutput = vec4<f32>(1.0f, 0.0f, 0.0f, 1.0f);

        return out;
    }

    // small history size, use correct higher MIP for more samples 
    if(indirectDiffuseRadianceHistory.w <= 10.0f)
    {
        let iMIPlevel: i32 = i32(
            (1.0f - clamp(indirectDiffuseRadianceHistory.w / 10.0f, 0.0f, 1.0f)) * 3.0f
        );

        if(iMIPlevel == 0)
        {
            indirectDiffuseRadiance = textureSample(
                indirectDiffuseMIPTexture0,
                textureSampler,
                in.texCoord
            ).xyz;
        }
        else if(iMIPlevel == 1)
        {
            indirectDiffuseRadiance = textureSample(
                indirectDiffuseMIPTexture1,
                textureSampler,
                in.texCoord
            ).xyz;
        }
        else if(iMIPlevel == 2)
        {
            indirectDiffuseRadiance = textureSample(
                indirectDiffuseMIPTexture2,
                textureSampler,
                in.texCoord
            ).xyz;
        }
        else if(iMIPlevel == 3)
        {
            indirectDiffuseRadiance = textureSample(
                indirectDiffuseMIPTexture3,
                textureSampler,
                in.texCoord
            ).xyz;
        }
    }

    indirectDiffuseRadiance += textureSample(
        indirectDiffuseScreenSpaceTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    let indirectDiffuseMomentHistory: vec4<f32> = textureSample(
        indirectDiffuseMomentHistoryTexture,
        textureSampler,
        prevScreenUV
    );

    // accumulation weight
    var fAccumulationBlendWeight: f32 = 1.0f / 10.0f;

    // mix previous to current radiance
    indirectDiffuseRadiance *= fAO * fIndirectDiffuseReservoirWeight;
    let mixedIndirectDiffuseRadiance: vec3<f32> = mix(
        indirectDiffuseRadianceHistory.xyz,
        indirectDiffuseRadiance,
        fAccumulationBlendWeight
    );

    emissiveRadiance *= fAO * fEmissiveReservoirWeight;
    let mixedEmissiveRadiance: vec3<f32> = mix(
        emissiveRadianceHistory.xyz,
        emissiveRadiance.xyz,
        fAccumulationBlendWeight
    );

    directSunRadiance *= fAO * fDirectSunReservoirWeight;
    let mixedDirectSunRadiance: vec3<f32> = mix(
        directSunRadianceHistory.xyz,
        directSunRadiance.xyz,
        fAccumulationBlendWeight
    );

    let specularMomentHistory: vec4<f32> = textureSample(
        specularMomentHistoryTexture,
        textureSampler,
        prevScreenUV
    );

    var specularStd: vec3<f32> = getStandardDeviation(in.texCoord);
    var clampedSpecularRadiance: vec3<f32> = clamp(
        specularRadiance.xyz, 
        specularRadiance.xyz - specularStd,
        specularRadiance.xyz + specularStd);

    specularRadiance *= fAO * fSpecularReservoirWeight;
    let mixedSpecularRadiance: vec3<f32> = mix(
        specularRadianceHistory.xyz,
        clampedSpecularRadiance.xyz,
        fAccumulationBlendWeight
    );

    let mixedIndirectDiffuseMoment: vec3<f32> = mix(
        indirectDiffuseMomentHistory.xyz,
        indirectDiffuseMoment.xyz,
        fAccumulationBlendWeight
    );

    let mixedSpecularMoment: vec3<f32> = mix(
        specularMomentHistory.xyz,
        specularMoment.xyz,
        fAccumulationBlendWeight
    );

    out.indirectDiffuseOutput.x = mixedIndirectDiffuseRadiance.x;
    out.indirectDiffuseOutput.y = mixedIndirectDiffuseRadiance.y;
    out.indirectDiffuseOutput.z = mixedIndirectDiffuseRadiance.z;
    out.indirectDiffuseOutput.w = indirectDiffuseRadianceHistory.w + 1.0f;

    out.emissiveOutput.x = mixedEmissiveRadiance.x;
    out.emissiveOutput.y = mixedEmissiveRadiance.y;
    out.emissiveOutput.z = mixedEmissiveRadiance.z;
    out.emissiveOutput.w = emissiveRadianceHistory.w + 1.0f;

    out.directSunOutput.x = mixedDirectSunRadiance.x;
    out.directSunOutput.y = mixedDirectSunRadiance.y;
    out.directSunOutput.z = mixedDirectSunRadiance.z;
    out.directSunOutput.w = directSunRadianceHistory.w + 1.0f;

    out.specularOutput.x = mixedSpecularRadiance.x;
    out.specularOutput.y = mixedSpecularRadiance.y;
    out.specularOutput.z = mixedSpecularRadiance.z;
    out.specularOutput.w = specularRadianceHistory.w + 1.0f;

    out.specularMomentOutput.x = mixedSpecularMoment.x;
    out.specularMomentOutput.y = mixedSpecularMoment.y;
    out.specularMomentOutput.z = abs(mixedSpecularMoment.y - mixedSpecularMoment.x * mixedSpecularMoment.x);
    out.specularMomentOutput.w = specularRadianceHistory.w + 1.0f;

    // moment with variance in z component
    out.indirectDiffuseMomentOutput = vec4<f32>(
        mixedIndirectDiffuseMoment.x,
        mixedIndirectDiffuseMoment.y,
        abs(mixedIndirectDiffuseMoment.y - mixedIndirectDiffuseMoment.x * mixedIndirectDiffuseMoment.x),
        indirectDiffuseMomentHistory.w + 1.0f
    );

    out.debugOutput = vec4<f32>(
        emissiveRadiance.xyz * fEmissiveReservoirWeight,
        1.0f
    );

    return out;
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
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
fn getStandardDeviation(texCoord: vec2<f32>) -> vec3<f32>
{
    var mean: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var fCount: f32 = 0.0f;
    var aRadiance: array<vec3<f32>, 9>;
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            let sampleUV: vec2<f32> = vec2<f32>(
                texCoord.x + (f32(iY) * (1.0f / f32(defaultUniformData.miScreenWidth))),
                texCoord.y + (f32(iX) * (1.0f / f32(defaultUniformData.miScreenHeight)))
            );

            aRadiance[u32(fCount)] = textureSample(
                specularRadianceTexture,
                textureSampler,
                sampleUV
            ).xyz;

            mean += aRadiance[u32(fCount)];

            fCount += 1.0f;
        }
    }

    mean /= fCount;
    
    fCount = 0.0f;
    var standardDeviation: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            let sampleUV: vec2<f32> = vec2<f32>(
                texCoord.x + (f32(iY) * (1.0f / f32(defaultUniformData.miScreenWidth))),
                texCoord.y + (f32(iX) * (1.0f / f32(defaultUniformData.miScreenHeight)))
            );

            let diff: vec3<f32> = aRadiance[u32(fCount)] - mean;
            standardDeviation = diff * diff;

            fCount += 1.0f;
        }
    }

    standardDeviation /= fCount;

    return standardDeviation;
}

/////
fn bilinearSample(
    texCoord: vec2<f32>,
    iTextureType: i32,
    afWeights: array<f32, 4>) -> vec4<f32>
{
    let fOffsetU: f32 = 1.0f / f32(defaultUniformData.miScreenWidth);
    let fOffsetV: f32 = 1.0f / f32(defaultUniformData.miScreenHeight);

    var fTotalWeights: f32 = 0.0f;
    var ret: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    if(iTextureType == INDIRECT_DIFFUSE)
    {
        var output: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        let sample: vec4<f32> = textureSample(
            indirectDiffuseRadianceHistoryTexture,
            textureSampler,
            texCoord                    
        );
        output += sample.xyz;
        fTotalWeights += 1.0f;

        var currTexCoord: vec2<f32> = texCoord + vec2<f32>(-fOffsetU, 0.0f);
        currTexCoord.x = max(currTexCoord.x, 0.0f);
        var spatialReservoir: vec4<f32> = textureSample(
            indirectDiffuseSpatialReservoirTexture,
            textureSampler,
            currTexCoord
        );
        output += textureSample(
            indirectDiffuseRadianceHistoryTexture,
            textureSampler,
            currTexCoord                
        ).xyz * afWeights[0] * clamp(spatialReservoir.z, 0.0f, 1.0f);
        fTotalWeights += afWeights[0] * clamp(spatialReservoir.z, 0.0f, 1.0f);

        currTexCoord = texCoord + vec2<f32>(fOffsetU, 0.0f);
        currTexCoord.x = min(currTexCoord.x, 1.0f);
        spatialReservoir = textureSample(
            indirectDiffuseSpatialReservoirTexture,
            textureSampler,
            currTexCoord
        );
        output += textureSample(
            indirectDiffuseRadianceHistoryTexture,
            textureSampler,
            currTexCoord                 
        ).xyz * afWeights[1] * clamp(spatialReservoir.z, 0.0f, 1.0f);
        fTotalWeights += afWeights[1] * clamp(spatialReservoir.z, 0.0f, 1.0f);

        currTexCoord = texCoord + vec2<f32>(0.0f, -fOffsetV);
        currTexCoord.y = max(currTexCoord.y, 0.0f);
        spatialReservoir = textureSample(
            indirectDiffuseSpatialReservoirTexture,
            textureSampler,
            currTexCoord
        );
        output += textureSample(
            indirectDiffuseRadianceHistoryTexture,
            textureSampler,
            currTexCoord                 
        ).xyz * afWeights[2] * clamp(spatialReservoir.z, 0.0f, 1.0f);
        fTotalWeights += afWeights[2] * clamp(spatialReservoir.z, 0.0f, 1.0f);

        currTexCoord = texCoord + vec2<f32>(0.0f, fOffsetV);
        currTexCoord.y = min(currTexCoord.y, 1.0f);
        spatialReservoir = textureSample(
            indirectDiffuseSpatialReservoirTexture,
            textureSampler,
            currTexCoord
        );
        output += textureSample(
            indirectDiffuseRadianceHistoryTexture,
            textureSampler,
            currTexCoord                   
        ).xyz * afWeights[3] * clamp(spatialReservoir.z, 0.0f, 1.0f);
        fTotalWeights += afWeights[3] * clamp(spatialReservoir.z, 0.0f, 1.0f);

        output /= fTotalWeights;
        ret = vec4<f32>(output.xyz, sample.w);
    }

    return ret;
}