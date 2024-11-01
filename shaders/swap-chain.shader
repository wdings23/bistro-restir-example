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
var texture6: texture_2d<f32>;

@group(0) @binding(7)
var texture7: texture_2d<f32>;

@group(0) @binding(8)
var texture8: texture_2d<f32>;

@group(0) @binding(9)
var texture9: texture_2d<f32>;

@group(0) @binding(10)
var texture10: texture_2d<f32>;

@group(0) @binding(11)
var texture11: texture_2d<f32>;

@group(0) @binding(12)
var texture12: texture_2d<f32>;

@group(0) @binding(13)
var texture13: texture_2d<f32>;

@group(0) @binding(14)
var texture14: texture_2d<f32>;

@group(0) @binding(15)
var texture15: texture_2d<f32>;

@group(0) @binding(16)
var texture16: texture_2d<f32>;

@group(0) @binding(17)
var texture17: texture_2d<f32>;

@group(0) @binding(18)
var texture18: texture_2d<f32>;

@group(0) @binding(19)
var texture19: texture_2d<f32>;

@group(0) @binding(20)
var texture20: texture_2d<f32>;

@group(0) @binding(21)
var texture21: texture_2d<f32>;

@group(0) @binding(22)
var texture22: texture_2d<f32>;

@group(0) @binding(23)
var texture23: texture_2d<f32>;

@group(0) @binding(24)
var texture24: texture_2d<f32>;

@group(0) @binding(25)
var texture25: texture_2d<f32>;

@group(0) @binding(26)
var texture26: texture_2d<f32>;

@group(0) @binding(27)
var texture27: texture_2d<f32>;

@group(0) @binding(28)
var texture28: texture_2d<f32>;

@group(0) @binding(29)
var texture29: texture_2d<f32>;

@group(0) @binding(30)
var texture30: texture_2d<f32>;

@group(0) @binding(31)
var texture31: texture_2d<f32>;

@group(0) @binding(32)
var texture32: texture_2d<f32>;

@group(0) @binding(33)
var texture33: texture_2d<f32>;

@group(0) @binding(34)
var texture34: texture_2d<f32>;

@group(0) @binding(35)
var texture35: texture_2d<f32>;

@group(0) @binding(36)
var texture36: texture_2d<f32>;

@group(0) @binding(37)
var texture37: texture_2d<f32>;

@group(0) @binding(38)
var texture38: texture_2d<f32>;

@group(0) @binding(39)
var texture39: texture_2d<f32>;

@group(0) @binding(40)
var texture40: texture_2d<f32>;

@group(0) @binding(41)
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
    else if(uniformData.miOutputTextureID == 6u)
    {
        radiance = textureSample(
            texture6, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 7u)
    {
        radiance = textureSample(
            texture7, 
            textureSampler, 
            in.texCoord);

        
    }
    else if(uniformData.miOutputTextureID == 8u)
    {
        radiance = textureSample(
            texture8, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 9u)
    {
        radiance = textureSample(
            texture9, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 10u)
    {
        radiance = textureSample(
            texture10, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 11u)
    {
        radiance = textureSample(
            texture11, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 12u)
    {
        radiance = textureSample(
            texture12, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 13u)
    {
        radiance = textureSample(
            texture13, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 14u)
    {
        radiance = textureSample(
            texture14, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 15u)
    {
        radiance = textureSample(
            texture15, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 16u)
    {
        radiance = textureSample(
            texture16, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 17u)
    {
        radiance = textureSample(
            texture17, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 18u)
    {
        radiance = textureSample(
            texture18, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 19u)
    {
        //radiance = textureSample(
        //    texture16,
        //    textureSampler,
        //    in.texCoord
        //);

        
        var indirectDiffuse: vec3<f32> = textureSample(
            texture12,
            textureSampler,
            in.texCoord 
        ).xyz;

        var specular: vec3<f32> = textureSample(
            texture17,
            textureSampler,
            in.texCoord
        ).xyz;

        var ao: vec3<f32> = textureSample(
            texture18,
            textureSampler,
            in.texCoord
        ).xyz;

        var dynamicShadow: vec3<f32> = textureSample(
            texture20,
            textureSampler,
            in.texCoord
        ).xyz;

        radiance.x = (indirectDiffuse.x * ao.x + specular.x) * 2.0f * dynamicShadow.x;
        radiance.y = (indirectDiffuse.y * ao.y + specular.y) * 2.0f * dynamicShadow.y;
        radiance.z = (indirectDiffuse.z * ao.z + specular.z) * 2.0f * dynamicShadow.z;
        
    }
    else if(uniformData.miOutputTextureID == 20u)
    {
        radiance = textureSample(
            texture20, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 21u)
    {
        radiance = textureSample(
            texture21, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 22u)
    {
        radiance = textureSample(
            texture22, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 23u)
    {
        radiance = textureSample(
            texture23, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 24u)
    {
        radiance = textureSample(
            texture24, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 25u)
    {
        radiance = textureSample(
            texture25, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 26u)
    {
        radiance = textureSample(
            texture26, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 27u)
    {
        radiance = textureSample(
            texture27, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 28u)
    {
        radiance = textureSample(
            texture28, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 29u)
    {
        radiance = textureSample(
            texture29, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 30u)
    {
        radiance = textureSample(
            texture30, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 31u)
    {
        radiance = textureSample(
            texture31, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 32u)
    {
        radiance = textureSample(
            texture32, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 33u)
    {
        radiance = textureSample(
            texture33, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 34u)
    {
        radiance = textureSample(
            texture34, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 35u)
    {
        radiance = textureSample(
            texture35, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 36u)
    {
        radiance = textureSample(
            texture36, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 37u)
    {
        radiance = textureSample(
            texture37, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 38u)
    {
        radiance = textureSample(
            texture38, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 39u)
    {
        radiance = textureSample(
            texture39, 
            textureSampler, 
            in.texCoord);
    }
    else if(uniformData.miOutputTextureID == 40u)
    {
        radiance = textureSample(
            texture40, 
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