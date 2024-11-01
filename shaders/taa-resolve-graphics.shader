const FLT_MAX: f32 = 1000000.0f;
const UINT32_MAX: u32 = 0xffffffff;
const PI: f32 = 3.14159f;
const NUM_HISTORY: f32 = 30.0f;

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
    @location(0) output : vec4<f32>,
};

struct UniformData
{
    mCurrViewProjectionMatrix: mat4x4<f32>,
    mPrevViewProjectionMatrix: mat4x4<f32>,

    miScreenWidth: u32,
    miScreenHeight: u32,
    miFrameIndex: u32,
    miPadding: u32,
};

@group(0) @binding(0)
var currOutputTexture: texture_2d<f32>;

@group(0) @binding(1)
var prevOutputTexture: texture_2d<f32>;

@group(0) @binding(2)
var currWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(3)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(4)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
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
    
    let currOutput: vec4<f32> = textureSample(
        currOutputTexture,
        textureSampler,
        in.texCoord
    );

    let prevWorldPosition: vec4<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        in.texCoord
    );

    var worldPosition: vec4<f32> = textureSample(
        currWorldPositionTexture,
        textureSampler,
        in.texCoord
    );

    if(worldPosition.w <= 0.0f)
    {
        out.output = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }
    
    // re-project screen uv
    worldPosition.w = 1.0f;
    var prevClipSpacePos: vec4<f32> = worldPosition * defaultUniformData.mPrevJitteredViewProjectionMatrix;
    prevClipSpacePos.x /= prevClipSpacePos.w;
    prevClipSpacePos.y /= prevClipSpacePos.w;
    prevClipSpacePos.x = prevClipSpacePos.x * 0.5f + 0.5f;
    prevClipSpacePos.y = prevClipSpacePos.y * 0.5f + 0.5f;
    let prevScreenUV: vec2<f32> = vec2<f32>(
        prevClipSpacePos.x,
        1.0f - prevClipSpacePos.y
    );
    let prevOutput: vec4<f32> = textureSample(
        prevOutputTexture,
        textureSampler,
        in.texCoord
    );

    let prevCenterWorldPosition: vec4<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        prevScreenUV
    );

    var totalPrevColor: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var fPrevTotalWeight: f32 = 0.0f;

    // color clamp
    var minColor: vec3<f32> = vec3<f32>(9999.0f, 9999.0f, 9999.0f);
    var maxColor: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            let sampleUV: vec2<f32> = vec2<f32>(
                in.texCoord.x + (f32(iX) / f32(defaultUniformData.miScreenWidth)),
                in.texCoord.y + (f32(iY) / f32(defaultUniformData.miScreenHeight))
            );

            let sampleColor: vec3<f32> = textureSample(
                currOutputTexture,
                textureSampler,
                sampleUV
            ).xyz;

            minColor = min(sampleColor, minColor);
            maxColor = max(sampleColor, maxColor);

            // sample previous output with weight
            let samplePrevUV: vec2<f32> = vec2<f32>(
                prevScreenUV.x + (f32(iX) / f32(defaultUniformData.miScreenWidth)),
                prevScreenUV.y + (f32(iY) / f32(defaultUniformData.miScreenHeight)),
            );
            let samplePrevColor: vec3<f32> = textureSample(
                prevOutputTexture,
                textureSampler,
                samplePrevUV
            ).xyz;

            let prevSampleWorldPosition: vec3<f32> = textureSample(
                prevWorldPositionTexture,
                textureSampler,
                samplePrevUV
            ).xyz;

            //let worldPositionDiff: vec3<f32> = worldPosition.xyz - prevSampleWorldPosition.xyz; 
            var fWeight: f32 = exp(-3.0f * f32(iX * iX + iY * iY) / f32(3 * 3)); // * (max(3.0f - length(worldPositionDiff), 0.0f) / 3.0f);

            fPrevTotalWeight += fWeight;
            totalPrevColor += samplePrevColor * fWeight;
        }
    }

    // normalize previous sample color with total weight
    totalPrevColor /= (fPrevTotalWeight + 0.0001f);
    var fHistoryPct: f32 = 1.0f / NUM_HISTORY;
    if(fPrevTotalWeight <= 0.0f)
    {
        fHistoryPct = 1.0f;
    }

    // clamp previous color with current min and max color in box
    let prevClampedOutput: vec3<f32> = clamp(totalPrevColor.xyz, minColor, maxColor);
    var mixedOutput: vec4<f32> = mix(
        vec4<f32>(prevClampedOutput.xyz, 1.0f),
        currOutput,
        fHistoryPct);

    out.output = mixedOutput;

    return out;
}

