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
    @location(0) radiance: vec4<f32>,
    @location(1) moment: vec4<f32>
};

struct UniformData
{
    miFrame: u32,
    miScreenWidth: u32,
    miScreenHeight: u32,
    mfRand: f32,

    mViewProjectionMatrix: mat4x4<f32>,
    
    miStep: u32,

};

struct RandomResult 
{
    mfNum: f32,
    miSeed: u32,
};

struct SVGFFilterResult
{
    mRadiance: vec3<f32>,
    mMoments: vec3<f32>,
};

@group(0) @binding(0)
var indirectDiffuseRadianceTexture: texture_2d<f32>;

@group(0) @binding(1)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(2)
var normalTexture: texture_2d<f32>;

@group(0) @binding(3)
var momentTexture: texture_2d<f32>;

@group(0) @binding(4)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(5)
var textureSampler: sampler;

@group(0) @binding(6)
var linearTextureSampler: sampler;

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
        out.radiance = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        out.moment = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    }

    let screenCoord: vec2<i32> = vec2<i32>(
        i32(floor(in.texCoord.x * f32(uniformData.miScreenWidth) + 0.5f)),
        i32(floor(in.texCoord.y * f32(uniformData.miScreenHeight) + 0.5f)) 
    );

    let svgfFilterResult: SVGFFilterResult = svgFilter(
        screenCoord.x,
        screenCoord.y,
        i32(uniformData.miStep),
        30.0f);

    out.radiance = vec4<f32>(
        svgfFilterResult.mRadiance, 
        1.0f);
    out.moment = vec4<f32>(
        svgfFilterResult.mMoments.x,
        svgfFilterResult.mMoments.y,
        abs(svgfFilterResult.mMoments.y - svgfFilterResult.mMoments.x * svgfFilterResult.mMoments.x),
        1.0f);

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
fn computeSVGFNormalStoppingWeight(
    sampleNormal: vec3<f32>,
    normal: vec3<f32>,
    fPower: f32) -> f32
{
    let fDP: f32 = clamp(dot(normal, sampleNormal), 0.0f, 1.0f);
    return pow(fDP, fPower);
};

/////
fn computeSVGFDepthStoppingWeight(
    fSampleDepth: f32,
    fDepth: f32,
    fPhi: f32) -> f32
{
    let kfEpsilon: f32 = 0.001f;
    return exp(-abs(fDepth - fSampleDepth) / (fPhi + kfEpsilon));
}

/////
fn computeSVGFLuminanceStoppingWeight(
    fSampleLuminance: f32,
    fLuminance: f32,
    iX: i32,
    iY: i32,
    fPower: f32) -> f32
{
    let kfEpsilon: f32 = 0.001f;

    let fOneOverSixteen: f32 = 1.0f / 16.0f;
    let fOneOverEight: f32 = 1.0f / 8.0f;
    let fOneOverFour: f32 = 1.0f / 4.0f;
    var afKernel: array<f32, 9> = array<f32, 9>(
        fOneOverSixteen, fOneOverEight, fOneOverSixteen,
        fOneOverEight, fOneOverFour, fOneOverEight,
        fOneOverSixteen, fOneOverEight, fOneOverSixteen,
    );

    // gaussian blur for variance
    var fLuminanceDiff: f32 = abs(fLuminance - fSampleLuminance);
    var fTotalVariance: f32 = 0.0f;
    var fTotalKernel: f32 = 0.0f;
    for(var iOffsetY: i32 = -1; iOffsetY <= 1; iOffsetY++)
    {
        var iSampleY: i32 = iY + iOffsetY;
        if(iSampleY < 0 && iSampleY >= i32(uniformData.miScreenHeight))
        {
            continue;
        }
        for(var iOffsetX: i32 = -1; iOffsetX <= 1; iOffsetX++)
        {
            var iSampleX: i32 = iX + iOffsetX;
            if(iSampleX < 0 && iSampleY >= i32(uniformData.miScreenWidth))
            {
                continue;
            }

            let sampleUV: vec2<f32> = vec2<f32>(
                f32(iSampleX) / f32(uniformData.miScreenWidth), 
                f32(iSampleY) / f32(uniformData.miScreenHeight));
            let moment: vec4<f32> = textureSample(
                momentTexture,
                textureSampler,
                sampleUV);

            let iIndex: i32 = (iOffsetY + 1) * 3 + (iOffsetX + 1);
            fTotalVariance += afKernel[iIndex] * abs(moment.y - moment.x * moment.x);
            fTotalKernel += afKernel[iIndex];
        }
    }

    let fRet: f32 = exp(-fLuminanceDiff / (sqrt(fTotalVariance / fTotalKernel) * fPower + kfEpsilon));
    return fRet;
}



/////
fn svgFilter(
    iX: i32,
    iY: i32,
    iStep: i32,
    fLuminancePhi: f32) -> SVGFFilterResult
{
    var ret: SVGFFilterResult;

    var retRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var fRetVariance: f32 = 0.0f;

    // 5x5 A-Trous kernel
    var afKernel: array<f32, 25> = array<f32, 25>
    (
        1.0f / 256.0f, 1.0f / 64.0f, 3.0f / 128.0f, 1.0f / 64.0f, 1.0f / 256.0f,
        1.0f / 64.0f, 1.0f / 16.0f, 3.0f / 32.0f, 1.0f / 16.0f, 1.0f / 64.0f,
        3.0f / 128.0f, 3.0f / 32.0f, 9.0f / 64.0f, 3.0f / 32.0f, 3.0f / 128.0f,
        1.0f / 64.0f, 1.0f / 16.0f, 3.0f / 32.0f, 1.0f / 16.0f, 1.0f / 64.0f,
        1.0f / 256.0f, 1.0f / 64.0f, 3.0f / 128.0f, 1.0f / 64.0f, 1.0f / 256.0f
    );

    let kfNormalPower: f32 = 128.0f;
    let kfDepthPhi: f32 = 1.0e-2f;

    let imageUV: vec2<f32> = vec2<f32>(
        f32(iX) / f32(uniformData.miScreenWidth), 
        f32(iY) / f32(uniformData.miScreenHeight)
    );
    
    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        imageUV
    );
    if(worldPosition.w == 0.0f)
    {
        return ret;
    }

    let normal: vec4<f32> = textureSample(
        normalTexture,
        textureSampler,
        imageUV
    );

    var fDepth: f32 = fract(worldPosition.w);

    var radiance: vec4<f32> = textureSample(
        indirectDiffuseRadianceTexture,
        textureSampler,
        imageUV
    );
    let fLuminance: f32 = computeLuminance(radiance.xyz);

    var totalRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var fTotalWeight: f32 = 0.0f;
    var fTotalSquaredWeight: f32 = 0.0f;
    var fTotalWeightedVariance: f32 = 0.0f;
    let iStepSize: u32 = 1u << u32(iStep);

    var totalMoments: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);

    for(var iStepY: i32 = -2; iStepY <= 2; iStepY++)
    {
        var iOffsetY: i32 = iStepY * i32(iStepSize);
        var iSampleY: i32 = iY + iOffsetY;
        if(iSampleY < 0 || iSampleY >= i32(uniformData.miScreenHeight))
        {
            continue;
        }

        var iCount: i32 = 0;
        for(var iStepX: i32 = -2; iStepX <= 2; iStepX++)
        {
            var iOffsetX: i32 = iStepX * i32(iStepSize);
            var iSampleX: i32 = iX + iOffsetX;
            if(iSampleX < 0 || iSampleX >= i32(uniformData.miScreenWidth))
            {
                continue;
            }

            let sampleUV: vec2<f32> = vec2<f32>(
                f32(iSampleX) / f32(uniformData.miScreenWidth), 
                f32(iSampleY) / f32(uniformData.miScreenHeight)
            );

            var sampleRadiance: vec4<f32> = textureSample(
                indirectDiffuseRadianceTexture,
                textureSampler,
                sampleUV
            );
            let sampleWorldPosition: vec4<f32> = textureSample(
                worldPositionTexture,
                textureSampler,
                sampleUV
            );
            let sampleNormal: vec4<f32> = textureSample(
                normalTexture,
                textureSampler,
                sampleUV
            );

            let moment: vec4<f32> = textureSample(
                momentTexture,
                textureSampler,
                sampleUV
            );

            var sampleAmbientOcclusion: vec4<f32> = textureSample(
                ambientOcclusionTexture, 
                textureSampler,
                sampleUV);
            var fSampleAO: f32 = 1.0f - (sampleAmbientOcclusion.x / sampleAmbientOcclusion.y); 
            if(uniformData.miStep > 0)
            {
                fSampleAO = 1.0f;
            }

fSampleAO = 1.0f;

            sampleRadiance.x *= fSampleAO;
            sampleRadiance.y *= fSampleAO;
            sampleRadiance.z *= fSampleAO;

            let fSampleVariance: f32 = abs(moment.y - moment.x * moment.x);
            let fSampleLuminance: f32 = computeLuminance(sampleRadiance.xyz);

            if(sampleWorldPosition.w == 0.0f)
            {
                continue;
            }

            var fSampleDepth: f32 = fract(sampleWorldPosition.w); 

            let fSampleNormalWeight: f32 = computeSVGFNormalStoppingWeight(
                sampleNormal.xyz,
                normal.xyz,
                kfNormalPower);
            let fSampleDepthWeight: f32 = computeSVGFDepthStoppingWeight(
                fDepth,
                fSampleDepth,
                kfDepthPhi);
            
            let fRetTotalVariance: f32 = 0.0f;
            let fSampleLuminanceWeight: f32 = computeSVGFLuminanceStoppingWeight(
                fSampleLuminance,
                fLuminance,
                iX,
                iY,
                fLuminancePhi);

            let fSampleWeight: f32 = fSampleNormalWeight * fSampleDepthWeight * fSampleLuminanceWeight;

            let fKernel: f32 = afKernel[iCount];
            iCount += 1;
            let fKernelWeight: f32 = fKernel * fSampleWeight;

            totalRadiance += sampleRadiance.xyz * fKernelWeight;
            fTotalWeight += fKernelWeight;
            fTotalSquaredWeight += fKernelWeight * fKernelWeight;

            fTotalWeightedVariance += fKernelWeight * fSampleVariance;

            totalMoments += moment.xyz * fKernelWeight;
        }
    }

    retRadiance = totalRadiance / (fTotalWeight + 0.0001f);
    fRetVariance = fTotalWeightedVariance / (fTotalSquaredWeight + 0.0001f);

    ret.mRadiance = retRadiance;
    ret.mMoments = totalMoments / (fTotalWeight + 1.0e-4f);

    return ret;
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}