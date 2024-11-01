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
    @location(0) indirectDiffuseOutput: vec4<f32>
};

struct UniformData
{
    miFrame: u32,
    miScreenWidth: u32,
    miScreenHeight: u32,
    mfRand: f32,

    maSamples: array<vec4<f32>, 64>,

    mViewProjectionMatrix: mat4x4<f32>,

    mfRand0: f32,
    mfRand1: f32,
    mfRand2: f32,
    mfRand3: f32,
};

struct RandomResult 
{
    mfNum: f32,
    miSeed: u32,
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
var textureSampler: sampler;

@group(0) @binding(5)
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
        out.indirectDiffuseOutput = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    let normal: vec4<f32> = textureSample(
        normalTexture,
        textureSampler,
        in.texCoord);

    let fCenterDepth: f32 = fract(worldPosition.w);
    let fPlaneD: f32 = -(dot(normal.xyz, worldPosition.xyz));

    // filter parameters
    let iNumFramesPerSwitch: u32 = 1u;
    let iNumTotalDistribution = 16u;
    let iNumSamples: u32 = 8u;
    let iNumRotations: u32 = 8u;
    let fMaxPlaneDistance: f32 = 2.0f;

    var randomResult: RandomResult = initRand(
        u32(in.texCoord.x * 100.0f + in.texCoord.y * 200.0f) + u32(uniformData.mfRand0 * 100.0f),
        u32(in.pos.x * 10.0f + in.pos.z * 20.0f) + u32(uniformData.mfRand1 * 100.0f),
        10u);

    randomResult = nextRand(randomResult.miSeed);
    let fRand0: f32 = randomResult.mfNum;
    let fTheta: f32 = fRand0 * 2.0f * PI;

    randomResult = nextRand(randomResult.miSeed);
    let fRand1: f32 = randomResult.mfNum;
    let fRadius: f32 = 32.0f; //f32(u32(fRand1 * 32.0f));

    let iSampleOffset: u32 = 0u; //((uniformData.miFrame / iNumFramesPerSwitch) % (iNumTotalDistribution / iNumSamples)) * iNumSamples;
    //let fTheta: f32 = (f32((uniformData.miFrame / iNumFramesPerSwitch) % iNumRotations) / f32(iNumRotations)) * 2.0f * PI;
    let fCosTheta: f32 = cos(fTheta);
    let fSinTheta: f32 = sin(fTheta);

    // filter current frame neighbor pixel samples
    var totalIndirectDiffuseRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var fTotalWeight: f32 = 0.0f;
    for(var iSample: u32 = 0; iSample < iNumSamples; iSample++)
    {
        var tapPosition: vec2<f32> = uniformData.maSamples[iSample + iSampleOffset].xy * fRadius;
        var tapOffset: vec2<f32> = vec2<f32>(
            tapPosition.x * fCosTheta - tapPosition.y * fSinTheta,
            tapPosition.x * fSinTheta + tapPosition.y * fCosTheta
        );
        var sampleCoord: vec2<f32> = clamp(
            in.texCoord + tapOffset,
            vec2<f32>(0.0f, 0.0f),
            vec2<f32>(1.0f, 1.0f)
        );

        var radiance: vec4<f32> = textureSample(
            indirectDiffuseRadianceTexture,
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

        if(radiance.w <= 0.0f || sampleWorldPosition.w <= 0.0f)
        {
            continue;
        }

        let fSampleDepth: f32 = fract(sampleWorldPosition.w);

        let fPlaneDistanceWeight: f32 = clamp(fMaxPlaneDistance - (dot(sampleWorldPosition.xyz, normal.xyz) + fPlaneD), 0.0f, 1.0f); 

        let fNormalWeight: f32 = pow(max(dot(sampleNormal.xyz, normal.xyz), 0.0f), 10.0f);
        let fDepthWeight: f32 = exp(-abs(fCenterDepth - fSampleDepth) / (0.5f + 0.0001f));
        
        let fSampleWeight: f32 = fDepthWeight * fNormalWeight * fPlaneDistanceWeight;

        totalIndirectDiffuseRadiance.x += radiance.x * fSampleWeight;
        totalIndirectDiffuseRadiance.y += radiance.y * fSampleWeight;
        totalIndirectDiffuseRadiance.z += radiance.z * fSampleWeight;

        fTotalWeight += fSampleWeight;
    }

    totalIndirectDiffuseRadiance = totalIndirectDiffuseRadiance / (fTotalWeight + 1.0e-4f);

    out.indirectDiffuseOutput = vec4<f32>(totalIndirectDiffuseRadiance, 1.0f);
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

struct SVGFFilterResult
{
    mRadiance: vec3<f32>,
    mfVariance: f32,
};

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
            fTotalVariance += afKernel[iIndex] * moment.z;
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

    let kfNormalPower: f32 = 8.0f;
    let kfDepthPhi: f32 = 0.5f;

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

    var clipSpace: vec4<f32> = worldPosition * uniformData.mViewProjectionMatrix;
    clipSpace.z /= clipSpace.w;

    let radiance: vec4<f32> = textureSample(
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

            let sampleRadiance: vec4<f32> = textureSample(
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

            let sRGB: vec3<f32> = sampleRadiance.xyz; 
            let fSampleVariance: f32 = moment.z;
            let fSampleLuminance: f32 = computeLuminance(sampleRadiance.xyz);

            if(sampleWorldPosition.w == 0.0f)
            {
                continue;
            }

            var sampleClipSpace: vec4<f32> = sampleWorldPosition * uniformData.mViewProjectionMatrix;
            sampleClipSpace.z /= sampleClipSpace.w;

            let fSampleNormalWeight: f32 = computeSVGFNormalStoppingWeight(
                sampleNormal.xyz,
                normal.xyz,
                kfNormalPower);
            let fSampleDepthWeight: f32 = computeSVGFDepthStoppingWeight(
                clipSpace.z,
                sampleClipSpace.z,
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

            fTotalWeightedVariance += fKernelWeight * fKernelWeight * fSampleVariance;
        }
    }

    retRadiance = totalRadiance / fTotalWeight;
    fRetVariance = fTotalWeightedVariance / fTotalSquaredWeight;

    ret.mRadiance = retRadiance;
    ret.mfVariance = fRetVariance;

    return ret;
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}