@group(0) @binding(0)
var<storage, read> aCameraMatrices: array<mat4x4<f32>>;

struct UniformData {
    mViewProjectionMatrix: mat4x4<f32>,
    mLightViewProjectionMatrix: mat4x4<f32>,
    mInverseViewProjectionMatrix: mat4x4<f32>,

    miScreenWidth: u32,
    miScreenHeight: u32,

    miFrustumIndex: u32,
};
@group(1) @binding(0)
var<uniform> uniformData: UniformData;

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec4<f32>,
    @location(2) normal : vec4<f32>,
};
struct VertexOutput {
    @location(0) worldPosition: vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(2) normal: vec4<f32>,
};
struct FragmentOutput {
    @location(0) output : vec4<f32>,
};


@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) iVertexIndex: u32) -> VertexOutput 
{
    var out: VertexOutput;
    out.pos = vec4<f32>(in.pos.x, in.pos.y, in.pos.z, 1.0f) * aCameraMatrices[0];
    out.worldPosition = in.pos;
    out.texCoord = in.texCoord.xy;
    out.normal = in.normal;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput 
{
    var out: FragmentOutput;
    out.output = vec4<f32>(in.worldPosition.x, in.worldPosition.y, in.worldPosition.z, in.worldPosition.w + in.pos.z);
    
    return out;
}
