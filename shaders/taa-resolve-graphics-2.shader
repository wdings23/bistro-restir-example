const FLT_MAX: f32 = 1000000.0f;
const UINT32_MAX: u32 = 0xffffffff;
const PI: f32 = 3.14159f;
const NUM_HISTORY: f32 = 10.0f;

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
var motionTexture: texture_2d<f32>;

@group(0) @binding(3)
var currWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(4)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(5)
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
    
    var currOutput: vec4<f32> = textureSample(
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
    
    var clipSpaceDiff: vec4<f32> = textureSample(
        motionTexture,
        textureSampler,
        in.texCoord
    );
    clipSpaceDiff.x = clipSpaceDiff.x;
    clipSpaceDiff.y = clipSpaceDiff.y; 

    //let prevScreenUV: vec2<f32> = getPrevScreenUV(
    //    worldPosition.xyz,
    //    in.texCoord
    //);
    let prevScreenUV: vec2<f32> = vec2<f32>(
        in.texCoord.x - clipSpaceDiff.x,
        in.texCoord.y - clipSpaceDiff.y
    );
    var prevOutput: vec3<f32> = textureSample(
        prevOutputTexture,
        textureSampler,
        prevScreenUV
    ).xyz;

    let prevCenterWorldPosition: vec4<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        prevScreenUV
    );

    // check for disocclusion
    var bDisocclusion: bool = false;
    let worldPositionDiff: vec3<f32> = prevCenterWorldPosition.xyz - worldPosition.xyz;
    if(/*dot(worldPositionDiff, worldPositionDiff) >= 0.1f || */abs(worldPosition.w - prevCenterWorldPosition.w) >= 1.0f)
    {
        bDisocclusion = true;
    }

    var minColor: vec3<f32> = vec3<f32>(9999.0f, 9999.0f, 9999.0f);
    var maxColor: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    
    let textureSize: vec2<u32> = textureDimensions(currOutputTexture);
    let oneOverTextureSize: vec2<f32> = vec2<f32>(
        1.0f / f32(textureSize.x),
        1.0f / f32(textureSize.y)
    );

    // get min max neighbor color for clamping
    let iSampleSize: i32 = 1;
    var filteredColor: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var fSampleCount: f32 = 0.0f;
    for(var iY: i32 = -iSampleSize; iY <= iSampleSize; iY++)
    {
        for(var iX: i32 = -iSampleSize; iX <= iSampleSize; iX++)
        {
            let sampleUV: vec2<f32> = in.texCoord + vec2<f32>(oneOverTextureSize.x * f32(iX), oneOverTextureSize.y * f32(iY));
            let sampleColor: vec3<f32> = textureSample(
                currOutputTexture,
                textureSampler,
                sampleUV
            ).xyz;
            
            minColor = min(sampleColor, minColor);
            maxColor = max(sampleColor, maxColor);

            filteredColor += sampleColor;
            fSampleCount += 1.0f;

        }   // for x = -sampleSize to sampleSize
    }   // for y = -sampleSize to sampleSize

    var fHistoryPct: f32 = 1.0f / NUM_HISTORY;
    if(bDisocclusion)
    {
        prevOutput = filteredColor / fSampleCount;
    }

    // clamp previous color with current min and max color in box
    let prevClampedOutput: vec3<f32> = clamp(prevOutput, minColor, maxColor);
    var mixedOutput: vec4<f32> = mix(
        vec4<f32>(prevClampedOutput.xyz, 1.0f),
        currOutput,
        fHistoryPct);

    out.output = mixedOutput;

    return out;
}

/////
fn getPrevScreenUV(
    currWorldPosition: vec3<f32>,
    texCoord: vec2<f32>) -> vec2<f32>
{
    var clipSpaceDiff: vec4<f32> = textureSample(
        motionTexture,
        textureSampler,
        texCoord
    );
    clipSpaceDiff.x = clipSpaceDiff.x * 2.0f - 1.0f;
    clipSpaceDiff.y = clipSpaceDiff.y * 2.0f - 1.0f; 
    let prevScreenUV: vec2<f32> = vec2<f32>(
        texCoord.x - clipSpaceDiff.x,
        texCoord.y - clipSpaceDiff.y
    );

    let dUV: vec2<f32> = vec2<f32>(
        1.0f / f32(defaultUniformData.miScreenWidth),
        1.0f / f32(defaultUniformData.miScreenHeight)
    );

    var retUV: vec2<f32> = texCoord - vec2<f32>(clipSpaceDiff.x, clipSpaceDiff.y);
    var fSmallestDiff: f32 = 99999.0f;
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        var fV: f32 = retUV.y + f32(iY) * dUV.y;
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            var fU: f32 = retUV.x + f32(iX) * dUV.x;
            let sampleWorldPosition: vec3<f32> = textureSample(
                currWorldPositionTexture,
                textureSampler,
                vec2<f32>(fU, fV)
            ).xyz;

            let diff: vec3<f32> = currWorldPosition - sampleWorldPosition;
            let fDiffLength = dot(diff, diff);
            if(fDiffLength < fSmallestDiff)
            {
                retUV = vec2<f32>(fU, fV);
                fSmallestDiff = fDiffLength;
            }
        }
    }

    return retUV;
}