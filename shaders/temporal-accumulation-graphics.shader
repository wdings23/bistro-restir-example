const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const kfOneOverMaxBlendFrames: f32 = 1.0f / 10.0f;

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
    @location(1) indirectDiffuseMomentOutput: vec4<f32>
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
    miPadding: u32,

};

@group(0) @binding(0)
var worldPositionTexture: texture_2d<f32>;

@group(1) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var indirectDiffuseRadianceTexture: texture_2d<f32>;

@group(0) @binding(3)
var indirectDiffuseRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(4)
var indirectDiffuseMomentHistoryTexture: texture_2d<f32>;

@group(0) @binding(5)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

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

    let fOneOverNine: f32 = 1.0f / 9.0f;

    // sample radiance from surrounding neighbor box
    var totalIndirectDiffuseRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    for(var iOffsetY: i32 = -1; iOffsetY <= 1; iOffsetY++)
    {
        let fOffsetY: f32 = clamp(f32(iOffsetY) / f32(uniformData.miScreenHeight), 0.0f, f32(uniformData.miScreenHeight - 1));
        for(var iOffsetX: i32 = -1; iOffsetX <= 1; iOffsetX++)
        {
            let fOffsetX: f32 = clamp(f32(iOffsetX) / f32(uniformData.miScreenWidth), 0.0f, f32(uniformData.miScreenWidth - 1));
            var sampleCoord: vec2<f32> = vec2<f32>(fOffsetX, fOffsetY);

            let indirectDiffuseRadiance: vec4<f32> = textureSample(
                indirectDiffuseRadianceTexture,
                textureSampler,
                clamp(in.texCoord + sampleCoord, vec2<f32>(0.0f, 0.0f), vec2<f32>(1.0f, 1.0f))
            );

            let fNormalStoppingWeight: f32 = computeNormalStoppingWeight(
                sampleNormal,
                normal,
                1.0f;
            );

            let fDepthStoppingWeight: f32 = computeDepthStoppingWeight(
                fSampleDepth,
                fDepth,
                1.0f
            )

            if(indirectDiffuseRadiance.w <= 0.0f)
            {
                continue;
            }

            totalIndirectDiffuseRadiance.x += indirectDiffuseRadiance.x;
            totalIndirectDiffuseRadiance.y += indirectDiffuseRadiance.y;
            totalIndirectDiffuseRadiance.z += indirectDiffuseRadiance.z;
        }
    }

    // mean
    let totalIndirectDiffuseMean: vec3<f32> = totalIndirectDiffuseRadiance * fOneOverNine;

    // standard deviation for radiance
    totalIndirectDiffuseRadiance = vec3<f32>(0.0f, 0.0f, 0.0f);
    for(var iOffsetY: i32 = -1; iOffsetY <= 1; iOffsetY++)
    {
        let fOffsetY: f32 = clamp(f32(iOffsetY) / f32(uniformData.miScreenHeight), 0.0f, f32(uniformData.miScreenHeight - 1));
        for(var iOffsetX: i32 = -1; iOffsetX <= 1; iOffsetX++)
        {
            let fOffsetX: f32 = clamp(f32(iOffsetX) / f32(uniformData.miScreenWidth), 0.0f, f32(uniformData.miScreenWidth - 1));
            var sampleCoord: vec2<f32> = vec2<f32>(fOffsetX, fOffsetY);

            let indirectDiffuseRadiance: vec4<f32> = textureSample(
                indirectDiffuseRadianceTexture,
                textureSampler,
                clamp(in.texCoord + sampleCoord, vec2<f32>(0.0f, 0.0f), vec2<f32>(1.0f, 1.0f))
            );

            if(indirectDiffuseRadiance.w <= 0.0f)
            {
                continue;
            }

            let indirectDiffuseDiff: vec3<f32> = indirectDiffuseRadiance.xyz - totalIndirectDiffuseMean;
            totalIndirectDiffuseRadiance += indirectDiffuseDiff * indirectDiffuseDiff;
        }
    }

    // standard deviation 
    let stdDeviationIndirectDiffuse: vec3<f32> = vec3<f32>(
        sqrt(totalIndirectDiffuseRadiance.x) * fOneOverNine,
        sqrt(totalIndirectDiffuseRadiance.y) * fOneOverNine,
        sqrt(totalIndirectDiffuseRadiance.z) * fOneOverNine
    );

    // min and max for clamping
    let minIndirectDiffuseRadiance: vec3<f32> = max(
        totalIndirectDiffuseMean - stdDeviationIndirectDiffuse,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );
    let maxIndirectDiffuseRadiance: vec3<f32> = max(
        totalIndirectDiffuseMean + stdDeviationIndirectDiffuse,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );

    let indirectDiffuseRadiance: vec4<f32> = textureSample(
        indirectDiffuseRadianceTexture,
        textureSampler,
        in.texCoord
    );

    let indirectDiffuseRadianceHistory: vec4<f32> = textureSample(
        indirectDiffuseRadianceHistoryTexture,
        textureSampler,
        in.texCoord
    );

    // box clamp for radiance
    var clampedIndirectDiffuse: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    clampedIndirectDiffuse.x = clamp(
        indirectDiffuseRadiance.x,
        minIndirectDiffuseRadiance.x,
        maxIndirectDiffuseRadiance.x
    );
    clampedIndirectDiffuse.y = clamp(
        indirectDiffuseRadiance.y,
        minIndirectDiffuseRadiance.y,
        maxIndirectDiffuseRadiance.y
    );
    clampedIndirectDiffuse.z = clamp(
        indirectDiffuseRadiance.z,
        minIndirectDiffuseRadiance.z,
        maxIndirectDiffuseRadiance.z
    );

    // blend
    let mixedIndirectDiffuseRadiance: vec3<f32> = mix(
        indirectDiffuseRadianceHistory.xyz,
        clampedIndirectDiffuse.xyz,
        kfOneOverMaxBlendFrames
    );

    out.indirectDiffuseOutput.x = mixedIndirectDiffuseRadiance.x;
    out.indirectDiffuseOutput.y = mixedIndirectDiffuseRadiance.y;
    out.indirectDiffuseOutput.z = mixedIndirectDiffuseRadiance.z;

    // radiance history count
    out.indirectDiffuseOutput.w = indirectDiffuseRadianceHistory.w + 1.0f;

    // moment and variance
    let indirectDiffuseMomentHistory: vec4<f32> = textureSample(
        indirectDiffuseMomentHistoryTexture,
        textureSampler,
        in.texCoord
    );
    let fLuminance: f32 = computeLuminance(out.indirectDiffuseOutput.xyz);
    let indirectDiffuseMoment: vec3<f32> = vec3<f32>(
        fLuminance,
        fLuminance * fLuminance,
        0.0f
    );
    out.indirectDiffuseMomentOutput.x = mix(
        indirectDiffuseMoment.x,
        indirectDiffuseMomentHistory.x,
        kfOneOverMaxBlendFrames);
    out.indirectDiffuseMomentOutput.y = mix(
        indirectDiffuseMoment.y,
        indirectDiffuseMomentHistory.y,
        kfOneOverMaxBlendFrames);
    
    // variance
    out.indirectDiffuseMomentOutput.z = max(
        out.indirectDiffuseMomentOutput.y - out.indirectDiffuseMomentOutput.x * out.indirectDiffuseMomentOutput.x,
        0.0f);

    // moment and variance history count
    out.indirectDiffuseMomentOutput.w = indirectDiffuseMomentHistory.w + 1.0f;

    return out;
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}

/////
fn computeNormalStoppingWeight(
    sampleNormal: vec3<f32>,
    normal: vec3<f32>,
    fPower: f32) -> f32
{
    let fDP: f32 = clamp(dot(normal, sampleNormal), 0.0f, 1.0f);
    return pow(fDP, fPower);
};

/////
fn computeDepthStoppingWeight(
    fSampleDepth: f32,
    fDepth: f32,
    fPhi: f32) -> f32
{
    let kfEpsilon: f32 = 0.001f;
    return exp(-abs(fDepth - fSampleDepth) / (fPhi + kfEpsilon));
}