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
    @location(0) output0: vec4<f32>,
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

struct UniformData
{
    mfScale: f32,
};

@group(0) @binding(0)
var inputTexture: texture_2d<f32>;

@group(0) @binding(1)
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

    let fOffsetU: f32 = 1.0f / (f32(defaultUniformData.miScreenWidth) * uniformData.mfScale);
    let fOffsetV: f32 = 1.0f / (f32(defaultUniformData.miScreenHeight) * uniformData.mfScale);

    var ret: vec3<f32> = textureSample(
        inputTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    var fW: f32 = textureSample(
        inputTexture,
        textureSampler,
        in.texCoord
    ).w;

    var offset: vec2<f32> = vec2<f32>(-fOffsetU, 0.0f);
    ret += textureSample(
        inputTexture,
        textureSampler,
        in.texCoord + offset
    ).xyz;

    offset = vec2<f32>(fOffsetU, 0.0f);
    ret += textureSample(
        inputTexture,
        textureSampler,
        in.texCoord + offset
    ).xyz;

    offset = vec2<f32>(0.0f, -fOffsetV);
    ret += textureSample(
        inputTexture,
        textureSampler,
        in.texCoord + offset
    ).xyz;

    offset = vec2<f32>(0.0f, fOffsetV);
    ret += textureSample(
        inputTexture,
        textureSampler,
        in.texCoord + offset
    ).xyz;

    out.output0 = vec4<f32>(ret * 0.25f, fW);

    return out;
}

