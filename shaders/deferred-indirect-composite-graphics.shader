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
    @location(0) output : vec4<f32>,
    @location(1) debug0 : vec4<f32>,
    @location(2) debug1 : vec4<f32>,
    @location(3) debug2 : vec4<f32>,
};

@group(0) @binding(0)
var lightingTexture: texture_2d<f32>;

@group(0) @binding(1)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(2)
var indirectDiffuseTexture: texture_2d<f32>;

@group(0) @binding(3)
var textureSampler: sampler;

struct UniformData
{
    maLightViewProjectionMatrix: array<mat4x4<f32>, 3>,
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
    var out: FragmentOutput;

    let lighting: vec3<f32> = textureSample(
        lightingTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    let indirectDiffuseRadiance: vec3<f32> = textureSample(
        indirectDiffuseTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    let ambientOcclusion: vec3<f32> = textureSample(
        ambientOcclusionTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    
    out.output = vec4<f32>(
        (lighting + indirectDiffuseRadiance) * ambientOcclusion.x, 
        1.0f
    );

    out.debug0 = vec4<f32>(lighting, 1.0f);
    out.debug1 = vec4<f32>(indirectDiffuseRadiance, 1.0f);
    out.debug2 = vec4<f32>(ambientOcclusion, 1.0f);

    return out;
}

