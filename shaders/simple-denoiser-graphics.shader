const UINT32_MAX: u32 = 1000000;
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
    @location(0) radianceOutput: vec4<f32>
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
    mfRand3: f32
};

@group(0) @binding(0)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var radianceTexture: texture_2d<f32>;

@group(0) @binding(3)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(4)
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
        out.radianceOutput = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    let normal: vec4<f32> = textureSample(
        normalTexture, 
        textureSampler, 
        in.texCoord);

    let fPlaneD: f32 = -(dot(normal.xyz, worldPosition.xyz));

    var fTotalWeight: f32 = 1.0f;
    var totalRadiance: vec3<f32> = textureSample(
        radianceTexture,
        textureSampler,
        in.texCoord
    ).xyz;
    var ambientOcclusion: vec4<f32> = textureSample(
        ambientOcclusionTexture, 
        textureSampler,
        in.texCoord);
    var fAO: f32 = 1.0f - (ambientOcclusion.x / ambientOcclusion.y); 
    totalRadiance *= fAO;

    let iSampleOffset: u32 = 0u; //(uniformData.miFrame % 4);
    let fTheta: f32 = (f32(uniformData.miFrame % 4) * 0.25f) * 2.0f * PI;
    let fCosTheta: f32 = cos(fTheta);
    let fSinTheta: f32 = sin(fTheta);

    let fCenterDepth: f32 = fract(worldPosition.w);

    let fMaxPlaneDistance: f32 = 2.0f;
    let iNumSamples: u32 = 8u;
    var fRadius: f32 = 8.0f; //uniformData.mfRand0 * 6.0f;
    let fKernelWeight: f32 = (1.0f / f32(iNumSamples)) * 2.0f;
    for(var iSample: u32 = 0; iSample < iNumSamples; iSample++)
    {
        var tapPosition: vec2<f32> = uniformData.maSamples[iSample + iSampleOffset].xy * fRadius;
        //tapPosition.x = clamp(tapPosition.x * fCosTheta - tapPosition.y * fSinTheta, 0.0f, 1.0f);
        //tapPosition.y = clamp(tapPosition.x * fSinTheta + tapPosition.y * fCosTheta, 0.0f, 1.0f);

        var radiance: vec4<f32> = textureSample(
            radianceTexture,
            textureSampler,
            in.texCoord + tapPosition);

        var sampleWorldPosition: vec4<f32> = textureSample(
            worldPositionTexture, 
            textureSampler, 
            in.texCoord + tapPosition);

        var sampleNormal: vec4<f32> = textureSample(
            normalTexture,
            textureSampler,
            in.texCoord + tapPosition);

        if(radiance.w <= 0.0f || sampleNormal.w <= 0.0f)
        {
            continue;
        }

        ambientOcclusion = textureSample(
            ambientOcclusionTexture, 
            textureSampler,
            in.texCoord + tapPosition);
        fAO = 1.0f;
        if(ambientOcclusion.w > 0.0f)
        {
            fAO = 1.0f - (ambientOcclusion.x / ambientOcclusion.y); 
        }
        radiance.x *= fAO;
        radiance.y *= fAO;
        radiance.z *= fAO;

        let fSampleDepth: f32 = fract(sampleWorldPosition.w);

        let fPlaneDistanceWeight: f32 = clamp(fMaxPlaneDistance - (dot(sampleWorldPosition.xyz, normal.xyz) + fPlaneD), 0.0f, 1.0f); 

        let fNormalWeight: f32 = pow(max(dot(sampleNormal.xyz, normal.xyz), 0.0f), 10.0f);
        let fDepthWeight: f32 = exp(-abs(fCenterDepth - fSampleDepth) / (0.5f + 0.0001f));
        
        let fSampleWeight = fNormalWeight * fDepthWeight * fKernelWeight * fPlaneDistanceWeight;

        totalRadiance.x += radiance.x * fSampleWeight;
        totalRadiance.y += radiance.y * fSampleWeight;
        totalRadiance.z += radiance.z * fSampleWeight;

        fTotalWeight += fNormalWeight * fDepthWeight * fKernelWeight;
    }
    
    out.radianceOutput = vec4<f32>((totalRadiance / fTotalWeight) * fAO, 1.0f);

    return out;
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}

