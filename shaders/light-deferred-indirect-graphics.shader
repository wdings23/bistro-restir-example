const PI: f32 = 3.14159f;

struct UniformData
{
    mViewProjectionMatrix: mat4x4<f32>,
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



@group(0) @binding(0)
var<uniform> uniformData: UniformData;

@group(0) @binding(1)
var<uniform> defaultUniformData: DefaultUniformData;

struct VertexInput {
    @location(0) worldPosition : vec4<f32>,
    @location(1) texCoord: vec4<f32>,
    @location(2) normal : vec4<f32>
};
struct VertexOutput {
    @location(0) worldPosition: vec4<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) normal: vec4<f32>,
};
struct FragmentOutput {
    @location(0) depth : vec4<f32>,
    @location(1) debug: vec4<f32>,
};


@vertex
fn vs_main(in: VertexInput,
    @builtin(vertex_index) iVertexIndex: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(in.worldPosition.xyz, 1.0f) * uniformData.mViewProjectionMatrix;
    out.worldPosition = in.worldPosition;
    out.normal = in.normal;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    out.depth = vec4<f32>(in.pos.z, in.pos.z, in.pos.z, 1.0f);
    out.debug = in.normal;

    return out;
}

