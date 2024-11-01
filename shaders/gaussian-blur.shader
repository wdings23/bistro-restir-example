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
    @location(0) radiance: vec4<f32>,
};

@group(0) @binding(0)
var radianceTexture: texture_2d<f32>;

@group(0) @binding(1)
var textureSampler: sampler;

struct UniformData
{
    miFrame: i32,
    miScreenWidth: i32,
    miScreenHeight: i32,
    mfRand: f32,

    mfAmbientOcclusionRadius: f32,
    mfAmbientOcclusionQuality: f32,
    mfAmbientOcclusionNumDirections: f32,
    mfPadding: f32,

    maSamples: array<vec4<f32>, 64>,

    mfRand0: f32,
    mfRand1: f32,
    mfRand2: f32,
    mfRand3: f32,

};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

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
    
    let fTwoPI: f32 = 6.283185f;
    var fOneOverScreenWidth: f32 = 1.0f / f32(uniformData.miScreenWidth);
    var fOneOverScreenHeight: f32 = 1.0f / f32(uniformData.miScreenHeight);

    var ret: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var screenUV: vec2<f32> = in.texCoord.xy;
    var fCount: f32 = 0.0f;
    for(var fD: f32 = 0.0f; fD < 3.14159f; fD += fTwoPI / uniformData.mfAmbientOcclusionNumDirections)
    {
        for(var i: f32 = 1.0f / uniformData.mfAmbientOcclusionQuality; i <= 1.0f; i += 1.0f / uniformData.mfAmbientOcclusionQuality)
        {
            var offset: vec2<f32> = vec2<f32>(
                cos(fD) * fOneOverScreenWidth,
                sin(fD) * fOneOverScreenHeight
            ) * uniformData.mfAmbientOcclusionRadius * i;

            var radiance: vec3<f32> = textureSample(
                radianceTexture,
                textureSampler,
                screenUV + offset  
            ).xyz;

            ret += radiance;

            fCount += 1.0f;
        }
    }

    //ret /= uniformData.mfAmbientOcclusionQuality * uniformData.mfAmbientOcclusionNumDirections - 15.0f;
    ret /= fCount;

    out.radiance = vec4<f32>(ret.xyz, 1.0f);

    return out;
}