const FLT_MAX: f32 = 1000000.0f;
const UINT32_MAX: u32 = 0xffffffff;
const PI: f32 = 3.14159f;

struct AxisInfo
{
    mTangent: vec3<f32>,
    mBinormal: vec3<f32>, 
};

struct FilterResult
{
    mOutput0: vec4<f32>,
    mOutput1: vec4<f32>,
    mOutput2: vec4<f32>,
    mOutput3: vec4<f32>,
    mOutput4: vec4<f32>,
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

struct VertexInput 
{
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @location(2) normal : vec4<f32>
};
struct VertexOutput 
{
    @location(0) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) normal: vec4<f32>
};
struct FragmentOutput 
{
    @location(0) output0 : vec4<f32>,
    @location(1) output1 : vec4<f32>,
    @location(2) output2 : vec4<f32>,
    @location(3) output3 : vec4<f32>,
    @location(4) output4 : vec4<f32>,

    @location(5) debug: vec4<f32>,
};

@group(0) @binding(0)
var indirectDiffuseTexture: texture_2d<f32>;

@group(0) @binding(1)
var emissiveTexture: texture_2d<f32>;

@group(0) @binding(2)
var directSunTexture: texture_2d<f32>;

@group(0) @binding(3)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(4)
var shadowTexture: texture_2d<f32>;

@group(0) @binding(5)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(6)
var normalTexture: texture_2d<f32>;

@group(0) @binding(7)
var textureSampler: sampler;

struct UniformData
{
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    mfRand0: f32,
    mfRand1: f32,

    miScreenWidth: u32,
    miScreenHeight: u32,

    mfPositionScale: f32,

    miFrameIndex: u32,
};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var blueNoiseTexture: texture_2d<f32>;

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
    var result: FilterResult = blur2(in.texCoord);
    out.output0 = result.mOutput0;
    out.output1 = result.mOutput1;
    out.output2 = result.mOutput2;
    out.output3 = result.mOutput3;
    out.output4 = result.mOutput4;

    return out;
}

//////
fn pow2(val: f32) -> f32
{
    return val * val;
}

/////
fn gaussian(i: vec2<f32>) -> f32 
{
    let fSigma: f32 = 35.0f; 
    return 1.0f / (2.0f * PI * pow2(fSigma)) * exp(-((pow2(i.x) + pow2(i.y)) / (2.0f * pow2(fSigma))));
}

 

/////
fn blur2(
    texCoord: vec2<f32>) -> FilterResult
{
    let fTwoPI: f32 = 6.283185f;
    let fQuality: f32 = 8.0f;
    let kfPlaneThreshold: f32 = 0.5f;
    let kfPositionThreshold: f32 = 0.4f;
    let kfAngleThreshold: f32 = 0.9f;
    var fRadius: f32 = 16.0f;

    let textureSize: vec2<u32> = textureDimensions(blueNoiseTexture, 0);
    let iTotalTextureSize: u32 = textureSize.x * textureSize.y;
    let iTileSize: u32 = 32u;
    let iNumTilesPerRow: u32 = textureSize.x / iTileSize;
    let iNumTotalTiles: u32 = iNumTilesPerRow * iNumTilesPerRow;

    var fOneOverScreenWidth: f32 = 1.0f / f32(defaultUniformData.miScreenWidth);
    var fOneOverScreenHeight: f32 = 1.0f / f32(defaultUniformData.miScreenHeight);

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        texCoord
    );

    let normal: vec3<f32> = textureSample(
        normalTexture,
        textureSampler,
        texCoord
    ).xyz;

    let iMesh: u32 = u32(worldPosition.w - fract(worldPosition.w));

    let axisInfo: AxisInfo = getAxis(normal);
    let fPlaneD: f32 = dot(normal, worldPosition.xyz) * -1.0f;

    var indirectDiffuseWithHistory: vec4<f32> = textureSample(
        indirectDiffuseTexture,
        textureSampler,
        texCoord);

    var ret0: vec3<f32> = indirectDiffuseWithHistory.xyz;
    var fHistoryCount: f32 = indirectDiffuseWithHistory.w;

    var ret1: vec3<f32> = textureSample(
        emissiveTexture,
        textureSampler,
        texCoord).xyz;

    var ret2: vec3<f32> = textureSample(
        directSunTexture,
        textureSampler,
        texCoord).xyz;
    
    var ret3: vec3<f32> = textureSample(
        ambientOcclusionTexture,
        textureSampler,
        texCoord).xyz;

    var ret4: vec3<f32> = textureSample(
        shadowTexture,
        textureSampler,
        texCoord).xyz;

    var screenCoord: vec2<u32> = vec2<u32>(
        u32(texCoord.x * f32(defaultUniformData.miScreenWidth)),
        u32(texCoord.y * f32(defaultUniformData.miScreenHeight))
    );

    let oneOverScreenDimension: vec2<f32> = vec2<f32>(
        1.0f / f32(defaultUniformData.miScreenWidth),
        1.0f / f32(defaultUniformData.miScreenHeight)
    );

    let iNumSamples: i32 = 16;
    var screenUV: vec2<f32> = texCoord.xy;
    var fTotalWeight: f32 = 1.0f;
    var iCurrIndex: u32 = u32(defaultUniformData.miFrame) * u32(iNumSamples);
    for(var iSample: i32 = 0; iSample < iNumSamples; iSample++)
    {
        let iTileX: u32 = (iCurrIndex + u32(iSample)) % iNumTilesPerRow;
        let iTileY: u32 = ((iCurrIndex + u32(iSample)) / iNumTilesPerRow) % (iNumTilesPerRow * iNumTilesPerRow);

        let iTileOffsetX: u32 = (iCurrIndex + u32(iSample)) % iTileSize;
        let iTileOffsetY: u32 = ((iCurrIndex + u32(iSample)) / iTileSize) % (iTileSize * iTileSize);

        let iOffsetX: u32 = iTileOffsetX + iTileX * iTileSize;
        let iOffsetY: u32 = iTileOffsetY + iTileY * iTileSize; 

        screenCoord.x = (screenCoord.x + iOffsetX) % textureSize.x;
        screenCoord.y = (screenCoord.y + iOffsetY) % textureSize.y;
        var sampleBlueNoiseUV: vec2<f32> = vec2<f32>(
            f32(screenCoord.x) / f32(textureSize.x),
            f32(screenCoord.y) / f32(textureSize.y) 
        );

        var blueNoise: vec3<f32> = textureSample(
            blueNoiseTexture,
            textureSampler,
            sampleBlueNoiseUV
        ).xyz;


        let fTheta: f32 = blueNoise.x * 2.0f * PI;
        let fCosTheta: f32 = cos(fTheta);
        let fSinTheta: f32 = sin(fTheta);

        let kfRadiusMultSpeed: f32 = 100.0f;
        let fCurrRadius: f32 = max(
            fRadius * ((kfRadiusMultSpeed - min(indirectDiffuseWithHistory.w, (kfRadiusMultSpeed - 1.0f))) / kfRadiusMultSpeed),
            4.0f);

        let fSampleRadius: f32 = (fCurrRadius * max(blueNoise.y, 0.7f));
        let offset: vec2<f32> = vec2<f32>(
            fCosTheta * fSampleRadius,
            fSinTheta * fSampleRadius
        );
        let uvOffset: vec2<f32> = offset * oneOverScreenDimension;

        let sampleUV: vec2<f32> = screenUV + uvOffset;
        let sampleWorldPosition: vec4<f32> = textureSample(
            worldPositionTexture,
            textureSampler,
            sampleUV
        );
        let sampleNormal: vec3<f32> = textureSample(
            normalTexture,
            textureSampler,
            sampleUV
        ).xyz;
        let diff: vec3<f32> = sampleWorldPosition.xyz - worldPosition.xyz;

        let iSampleMesh: u32 = u32(sampleWorldPosition.w - fract(sampleWorldPosition.w));

        let fPlaneDistance: f32 = abs(dot(diff, normal));
        let fDP: f32 = dot(sampleNormal, normal);
        if(length(diff) > kfPositionThreshold ||
            abs(fDP) < kfAngleThreshold || 
            iSampleMesh != iMesh)
        {
            continue;
        }

        var radiance0: vec3<f32> =  textureSample(
            indirectDiffuseTexture,
            textureSampler,
            sampleUV  
        ).xyz;

        var radiance1: vec3<f32> =  textureSample(
            emissiveTexture,
            textureSampler,
            sampleUV  
        ).xyz;

        var radiance2: vec3<f32> =  textureSample(
            directSunTexture,
            textureSampler,
            sampleUV  
        ).xyz;
        
        var radiance3: vec3<f32> =  textureSample(
            ambientOcclusionTexture,
            textureSampler,
            sampleUV  
        ).xyz;

        var radiance4: vec3<f32> = textureSample(
            shadowTexture,
            textureSampler,
            sampleUV
        ).xyz;

        let fWeight: f32 = 1.0f; 
        ret0 += radiance0 * fWeight;
        ret1 += radiance1 * fWeight;
        ret2 += radiance2 * fWeight;
        ret3 += radiance3 * fWeight;
        ret4 += radiance4 * fWeight;

        fTotalWeight += fWeight;
    }

    ret0 /= fTotalWeight;
    ret1 /= fTotalWeight;
    ret2 /= fTotalWeight;
    ret3 /= fTotalWeight;
    ret4 /= fTotalWeight;

    var result: FilterResult;
    result.mOutput0 = vec4<f32>(ret0.xyz, fHistoryCount);
    result.mOutput1 = vec4<f32>(ret1.xyz, fHistoryCount);
    result.mOutput2 = vec4<f32>(ret2.xyz, fHistoryCount);
    result.mOutput3 = vec4<f32>(ret3.xyz, fHistoryCount);
    result.mOutput4 = vec4<f32>(ret4.xyz, fHistoryCount);

    return result;
}

/////
fn getAxis(
    normal: vec3<f32>) -> AxisInfo
{
    var ret: AxisInfo;

    var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    if(abs(normal.y) > abs(normal.x) && abs(normal.y) > abs(normal.z)) 
    { 
        up = vec3<f32>(1.0f, 0.0f, 0.0f);
    }

    ret.mTangent = normalize(cross(up, normal));
    ret.mBinormal = normalize(cross(normal, ret.mTangent));

    return ret;
}
