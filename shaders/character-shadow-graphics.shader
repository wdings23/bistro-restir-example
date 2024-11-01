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
    @location(0) shadowOutput : vec4f,
};

@group(0) @binding(0)
var characterWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var viewWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(2)
var<storage, read> aCharacterCameraMatrices: array<mat4x4<f32>>;

@group(0) @binding(3)
var textureSampler : sampler;

struct UniformData
{
    mViewProjectionMatrix: mat4x4<f32>,
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

    out.shadowOutput = vec4<f32>(1.0f, 1.0f, 1.0f, 0.0f);

    let viewWorldPositionDepth: vec4<f32> = textureSample(
        viewWorldPositionTexture,
        textureSampler,
        in.texCoord);

    if(viewWorldPositionDepth.w <= 0.0f)
    {
        return out;
    }

    var viewClipSpace: vec4<f32> = 
        vec4<f32>(viewWorldPositionDepth.xyz, 1.0f) * 
        aCharacterCameraMatrices[0];

    viewClipSpace.x /= viewClipSpace.w;
    viewClipSpace.y /= viewClipSpace.w;
    viewClipSpace.z /= viewClipSpace.w;

    let screenCoord: vec2<f32> = vec2<f32>(
        viewClipSpace.x * 0.5f + 0.5f,
        1.0f - (viewClipSpace.y * 0.5f + 0.5f)
    );

    if(screenCoord.x < 0.0f || screenCoord.x > 1.0f || 
       screenCoord.y < 0.0f || screenCoord.y > 1.0f)
    {
        return out;
    }

    let characterWorldPositionDepth: vec4<f32> = textureSample(
        characterWorldPositionTexture,
        textureSampler,
        screenCoord
    );    

    let kfBias: f32 = 0.01f;
    if(characterWorldPositionDepth.w > 0.0f && viewClipSpace.z > fract(characterWorldPositionDepth.w) + kfBias)
    {
        out.shadowOutput = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    }

    return out;
}

