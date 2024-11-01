struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @location(2) color : vec4<f32>
};
struct VertexOutput {
    @location(0) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) color: vec4<f32>
};
struct FragmentOutput {
    @location(0) colorOutput : vec4f,
};

@group(0) @binding(0)
var indirectDiffuseOutputTexture: texture_2d<f32>;

@group(0) @binding(1)
var emissiveOutputTexture: texture_2d<f32>;

@group(0) @binding(2)
var directSunOutputTexture: texture_2d<f32>;

@group(0) @binding(3)
var specularOutputTexture: texture_2d<f32>;

@group(0) @binding(4)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(5)
var shadowTexture: texture_2d<f32>;

@group(0) @binding(6)
var textureSampler : sampler;

struct UniformData
{
    miOutputTextureID: u32,
    miScreenWidth: u32,
    miScreenHeight: u32,
    miPadding2: u32,
};



@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(in.pos.xyz, 1.0);
    out.texCoord = in.texCoord;
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;

    let indirectDiffuseOutput: vec4<f32> = textureSample(
        indirectDiffuseOutputTexture, 
        textureSampler, 
        in.texCoord);

    let shadowOutput: vec4<f32> = textureSample(
        shadowTexture,
        textureSampler,
        in.texCoord);

    let emissiveOutput: vec4<f32> = textureSample(
        emissiveOutputTexture,
        textureSampler,
        in.texCoord);

    let directSunOutput: vec4<f32> = textureSample(
        directSunOutputTexture,
        textureSampler,
        in.texCoord);

    let specularOutput: vec4<f32> = textureSample(
        specularOutputTexture,
        textureSampler,
        in.texCoord);

    let ambientOcclusionOutput: vec4<f32> = textureSample(
        ambientOcclusionTexture,
        textureSampler,
        in.texCoord);

    out.colorOutput = vec4<f32>(
        indirectDiffuseOutput.xyz * (shadowOutput.xyz + vec3<f32>(0.2f, 0.2f, 0.2f)) * ambientOcclusionOutput.x + 
        emissiveOutput.xyz * ambientOcclusionOutput.x + 
        directSunOutput.xyz * ambientOcclusionOutput.x + 
        specularOutput.xyz,
        1.0f);

    return out;
}

