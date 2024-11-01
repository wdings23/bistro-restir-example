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


@group(0) @binding(0)
var<storage, read> aPreviousVertexPositions: array<vec4<f32>>;

struct UniformData {
    mViewProjectionMatrix: mat4x4<f32>,
    mPrevViewProjectionMatrix: mat4x4<f32>,
    mViewMatrix: mat4x4<f32>,

    mJitteredViewProjectionMatrix: mat4x4<f32>,
    mPrevJitteredViewProjectionMatrix: mat4x4<f32>,
};
@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<uniform> defaultUniformData: DefaultUniformData;

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec4<f32>,
    @location(2) normal : vec4<f32>,
};
struct VertexOutput {
    @location(0) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,

    @location(2) worldPosition: vec4<f32>,
    @location(3) previousWorldPosition: vec4<f32>,
};
struct FragmentOutput {
    @location(0) worldPosition : vec4f,
    @location(1) texCoord: vec4f,
    @location(2) normal: vec4f,
    @location(3) motionVector: vec4f,
    @location(4) clipSpace: vec4<f32>,
    @location(5) skinOnlyClipSpace: vec4<f32>,
};


@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) iVertexIndex: u32) -> VertexOutput 
{
    var out: VertexOutput;
    out.pos = vec4<f32>(in.pos.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
    out.texCoord = in.texCoord.xy;
    out.normal = in.normal;

    out.worldPosition = in.pos;
    out.previousWorldPosition = aPreviousVertexPositions[iVertexIndex];

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput 
{
    var out: FragmentOutput;
    
    out.worldPosition = in.worldPosition;
    out.texCoord = vec4<f32>(in.texCoord.xy, 0.0f, 0.0f);
    out.normal = in.normal;

    var prevClipSpacePosition: vec4<f32> = in.previousWorldPosition * defaultUniformData.mPrevViewProjectionMatrix;
    var clipSpacePosition: vec4<f32> = in.worldPosition * defaultUniformData.mViewProjectionMatrix;

    prevClipSpacePosition.x /= prevClipSpacePosition.w;
    prevClipSpacePosition.y /= prevClipSpacePosition.w;
    prevClipSpacePosition.z /= prevClipSpacePosition.w;

    clipSpacePosition.x /= clipSpacePosition.w;
    clipSpacePosition.y /= clipSpacePosition.w;
    clipSpacePosition.z /= clipSpacePosition.w;

    prevClipSpacePosition.x = prevClipSpacePosition.x * 0.5f + 0.5f;
    prevClipSpacePosition.y = prevClipSpacePosition.y * 0.5f + 0.5f;
    prevClipSpacePosition.z = prevClipSpacePosition.z * 0.5f + 0.5f;

    var clipSpace: vec3<f32> = clipSpacePosition.xyz;

    clipSpacePosition.x = clipSpacePosition.x * 0.5f + 0.5f;
    clipSpacePosition.y = clipSpacePosition.y * 0.5f + 0.5f;
    clipSpacePosition.z = clipSpacePosition.z * 0.5f + 0.5f;

    out.motionVector.x = (clipSpacePosition.x - prevClipSpacePosition.x) * 0.5f + 0.5f;
    out.motionVector.y = 1.0f - ((clipSpacePosition.y - prevClipSpacePosition.y) * 0.5f + 0.5f);
    out.motionVector.z = floor(in.worldPosition.w + 0.5f);      // mesh id
    out.motionVector.w = clipSpacePosition.z;                    // depth

    out.worldPosition.w = in.pos.z;

    out.clipSpace = vec4<f32>(clipSpace.xyz, 1.0f);
    out.skinOnlyClipSpace = vec4<f32>(clipSpace.xyz, 1.0f);

    return out;
}