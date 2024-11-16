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
var texture0: texture_2d<f32>;

@group(0) @binding(1)
var texture1: texture_2d<f32>;

@group(0) @binding(2)
var texture2: texture_2d<f32>;

@group(0) @binding(3)
var texture3: texture_2d<f32>;

@group(0) @binding(4)
var texture4: texture_2d<f32>;

@group(0) @binding(5)
var texture5: texture_2d<f32>;

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

    var radiance: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

    let fOneOverScreenWidth: f32 = 1.0f / f32(uniformData.miScreenWidth);
    let fOneOverScreenHeight: f32 = 1.0f / f32(uniformData.miScreenHeight);

    if(uniformData.miOutputTextureID == 0u)
    {
        radiance = textureSample(
            texture0, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 1u)
    {
        radiance = textureSample(
            texture1, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 2u)
    {
        radiance = textureSample(
            texture2, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 3u)
    {
        radiance = textureSample(
            texture3, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 4u)
    {
        radiance = textureSample(
            texture4, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 5u)
    {
        radiance = textureSample(
            texture5, 
            textureSampler, 
            in.texCoord);
    }
    
    let sdrColor: vec3<f32> = ACESFilm(radiance.xyz);
    out.colorOutput.x = sdrColor.x;
    out.colorOutput.y = sdrColor.y;
    out.colorOutput.z = sdrColor.z;
    out.colorOutput.w = 1.0f;

    

    return out;
}

/////
fn ACESFilm(
    radiance: vec3<f32>) -> vec3<f32>
{
    let fA: f32 = 2.51f;
    let fB: f32 = 0.03f;
    let fC: f32 = 2.43f;
    let fD: f32 = 0.59f;
    let fE: f32 = 0.14f;

    return saturate((radiance * (fA * radiance + fB)) / (radiance * (fC * radiance + fD) + fE));
}

/////
fn linearToSRGB(
    x: vec3<f32>) -> vec3<f32>
{
    var ret: vec3<f32>;
    if(x.x < 0.0031308f && x.y < 0.0031308f && x.z < 0.0031308f)
    {
        ret = x * 12.92f;
    }
    else
    {
        ret = vec3<f32>(
            pow(x.x, 1.0f / 2.4f) * 1.055f - 0.055f,
            pow(x.y, 1.0f / 2.4f) * 1.055f - 0.055f,
            pow(x.z, 1.0f / 2.4f) * 1.055f - 0.055f
        );
    }

    return ret;
}

/////
fn convertToSRGB(
    radiance: vec3<f32>) -> vec3<f32>
{
    let maxComp: vec3<f32> = vec3<f32>(max(max(radiance.x, radiance.y), radiance.z));
    let maxRadiance: vec3<f32> = max(radiance, maxComp * 0.01f);
    let linearRadiance: vec3<f32> = ACESFilm(maxRadiance);

    return linearToSRGB(linearRadiance);
}