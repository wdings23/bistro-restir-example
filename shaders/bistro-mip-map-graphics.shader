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
    @location(1) output1: vec4<f32>,
    @location(2) output2: vec4<f32>,
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
var inputTexture0: texture_2d<f32>;

@group(0) @binding(1)
var inputTexture1: texture_2d<f32>;

@group(0) @binding(2)
var inputTexture2: texture_2d<f32>;

@group(0) @binding(3)
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

    var ret0: vec3<f32> = textureSample(
        inputTexture0,
        textureSampler,
        in.texCoord
    ).xyz;

    var fW0: f32 = textureSample(
        inputTexture0,
        textureSampler,
        in.texCoord
    ).w;

    var offset0: vec2<f32> = vec2<f32>(-fOffsetU, 0.0f);
    ret0 += textureSample(
        inputTexture0,
        textureSampler,
        in.texCoord + offset0
    ).xyz;

    offset0 = vec2<f32>(fOffsetU, 0.0f);
    ret0 += textureSample(
        inputTexture0,
        textureSampler,
        in.texCoord + offset0
    ).xyz;

    offset0 = vec2<f32>(0.0f, -fOffsetV);
    ret0 += textureSample(
        inputTexture0,
        textureSampler,
        in.texCoord + offset0
    ).xyz;

    offset0 = vec2<f32>(0.0f, fOffsetV);
    ret0 += textureSample(
        inputTexture0,
        textureSampler,
        in.texCoord + offset0
    ).xyz;

    var ret1: vec3<f32> = textureSample(
        inputTexture1,
        textureSampler,
        in.texCoord
    ).xyz;

    var fW1: f32 = textureSample(
        inputTexture1,
        textureSampler,
        in.texCoord
    ).w;

    var offset1: vec2<f32> = vec2<f32>(-fOffsetU, 0.0f);
    ret1 += textureSample(
        inputTexture1,
        textureSampler,
        in.texCoord + offset1
    ).xyz;

    offset1 = vec2<f32>(fOffsetU, 0.0f);
    ret1 += textureSample(
        inputTexture1,
        textureSampler,
        in.texCoord + offset1
    ).xyz;

    offset1 = vec2<f32>(0.0f, -fOffsetV);
    ret1 += textureSample(
        inputTexture1,
        textureSampler,
        in.texCoord + offset1
    ).xyz;

    offset1 = vec2<f32>(0.0f, fOffsetV);
    ret1 += textureSample(
        inputTexture1,
        textureSampler,
        in.texCoord + offset1
    ).xyz;





    var ret2: vec3<f32> = textureSample(
        inputTexture2,
        textureSampler,
        in.texCoord
    ).xyz;

    var fW2: f32 = textureSample(
        inputTexture2,
        textureSampler,
        in.texCoord
    ).w;

    var offset2: vec2<f32> = vec2<f32>(-fOffsetU, 0.0f);
    ret2 += textureSample(
        inputTexture2,
        textureSampler,
        in.texCoord + offset2
    ).xyz;

    offset2 = vec2<f32>(fOffsetU, 0.0f);
    ret2 += textureSample(
        inputTexture2,
        textureSampler,
        in.texCoord + offset2
    ).xyz;

    offset2 = vec2<f32>(0.0f, -fOffsetV);
    ret2 += textureSample(
        inputTexture2,
        textureSampler,
        in.texCoord + offset2
    ).xyz;

    offset2 = vec2<f32>(0.0f, fOffsetV);
    ret2 += textureSample(
        inputTexture2,
        textureSampler,
        in.texCoord + offset2
    ).xyz;

    out.output0 = vec4<f32>(ret0 * 0.25f, fW0);
    out.output1 = vec4<f32>(ret1 * 0.25f, fW1);
    out.output2 = vec4<f32>(ret2 * 0.25f, fW2);


    return out;
}

