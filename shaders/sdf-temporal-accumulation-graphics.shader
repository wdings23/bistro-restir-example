const FLT_MAX: f32 = 1000000.0f;
const UINT32_MAX: u32 = 0xffffffff;
const PI: f32 = 3.14159f;

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
    @location(1) moment: vec4<f32>,
    @location(2) debug: vec4<f32>,
};

@group(0) @binding(0)
var inputTexture: texture_2d<f32>;

@group(0) @binding(1)
var prevOutputTexture: texture_2d<f32>;

@group(0) @binding(2)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(3)
var normalTexture: texture_2d<f32>;

@group(0) @binding(4)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(5)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(6)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(7)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(8)
var prevMomentTexture: texture_2d<f32>;

@group(0) @binding(9)
var textureSampler: sampler;

struct UniformData
{
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    mfPositionScale: f32,
};

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
    let kfPlaneThreshold: f32 = 1.0f;
    let kfPositionThreshold: f32 = 0.2f;
    let kfAngleThreshold: f32 = 0.7f;

    var out: FragmentOutput;

    var screenCoord: vec2<u32> = vec2<u32>(
        u32(in.texCoord.x * f32(defaultUniformData.miScreenWidth)),
        u32(in.texCoord.y * f32(defaultUniformData.miScreenHeight))
    );

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture, 
        textureSampler, 
        in.texCoord);

    let normal: vec3<f32> = textureSample(
        normalTexture, 
        textureSampler, 
        in.texCoord).xyz;

    if(worldPosition.w <= 0.0f)
    {
        out.output0 = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    let fPlaneD: f32 = dot(normal, worldPosition.xyz) * -1.0f;

    let inputRadiance: vec4<f32> = textureSample(
        inputTexture,
        textureSampler,
        in.texCoord
    );

    var currOutput: vec3<f32> = inputRadiance.xyz;

    // check for disocclusion for previous history pixel
    var fDisocclusion: f32 = 0.0f;
    var prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);
    if(isPrevUVOutOfBounds(in.texCoord))
    {
        fDisocclusion = 1.0f;
    }
    else
    {
        fDisocclusion = f32(isDisoccluded2(in.texCoord, prevScreenUV));
    }

    let prevOutput: vec4<f32> = textureSample(
        prevOutputTexture,
        textureSampler,
        prevScreenUV
    );

    var fNumAccumulatedFrames: f32 = prevOutput.w;

    out.debug = vec4<f32>(
        fDisocclusion,
        fDisocclusion,
        fDisocclusion,
        1.0f);

    fNumAccumulatedFrames *= (1.0f - fDisocclusion);

    fNumAccumulatedFrames += 1.0f;
    
    var fBlendPct: f32 = clamp(1.0f / fNumAccumulatedFrames, 0.0f, 1.0f / 15.0f);
    if(fDisocclusion > 0.0f || fNumAccumulatedFrames <= 10.0f)
    {
        fBlendPct = 1.0f;
    }
    
    let temporalAccumulatedOutput: vec3<f32> = mix(
        prevOutput.xyz,
        currOutput,
        fBlendPct
    );

    out.output0 = vec4<f32>(
        temporalAccumulatedOutput.x,
        temporalAccumulatedOutput.y,
        temporalAccumulatedOutput.z,
        fNumAccumulatedFrames
    );

    var currMoment: vec2<f32> = getMoment(in.texCoord);
    var prevMoment: vec2<f32> = textureSample(
        prevMomentTexture,
        textureSampler,
        prevScreenUV
    ).xy;
    let moment: vec2<f32> = mix(
        prevMoment,
        currMoment,
        fBlendPct
    );
    let fVariance: f32 = moment.y - moment.x * moment.x;

    out.moment = vec4<f32>(
        moment.x,
        moment.y,
        fVariance,
        0.0f
    );

    out.debug = vec4<f32>(
        fBlendPct,
        fBlendPct,
        fBlendPct,
        1.0f
    );

    return out;
}

/////
fn getMoment(
    screenUV: vec2<f32>) -> vec2<f32>
{
    var ret: vec2<f32> = vec2<f32>(0.0f, 0.0f);
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            let offset: vec2<f32> = vec2<f32>(
                f32(iX) / f32(defaultUniformData.miScreenWidth),
                f32(iY) / f32(defaultUniformData.miScreenHeight)
            );

            let radiance: vec3<f32> = textureSample(
                inputTexture,
                textureSampler,
                screenUV + offset
            ).xyz;

            let fLuminance: f32 = computeLuminance(radiance);
            ret.x += fLuminance;
            ret.y += fLuminance * fLuminance;
        }
    }

    ret /= 9.0f;
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
fn isDisoccluded2(
    screenUV: vec2<f32>,
    prevScreenUV: vec2<f32>
) -> bool
{
    var worldPosition: vec4<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        screenUV);

    var prevWorldPosition: vec4<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        prevScreenUV);

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

    return !(iMesh == iPrevMesh && fCheckDepth <= 0.01f && fCheckWorldPositionDistance <= 0.01f && fCheckDP >= 0.99f && prevWorldPosition.w > 0.0f && worldPosition.w > 0.0f);
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}