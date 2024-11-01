const FLT_MAX: f32 = 1000000.0f;
const UINT32_MAX: u32 = 0xffffffff;
const PI: f32 = 3.14159f;

struct VertexInput 
{
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @location(2) normal : vec4<f32>
};
struct VertexOutput 
{
    @location(0) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) normal: vec4<f32>
};
struct FragmentOutput 
{
    @location(0) output0 : vec4<f32>,
    @location(1) output1 : vec4<f32>,
    @location(2) output2 : vec4<f32>,
};

struct BrixelRayHit
{
    miBrickIndex: u32,
    miBrixelIndex: u32
};

@group(0) @binding(0)
var inputTexture0: texture_2d<f32>;

@group(0) @binding(1)
var inputTexture1: texture_2d<f32>;

@group(0) @binding(2)
var inputTexture2: texture_2d<f32>;

@group(0) @binding(3)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(4)
var normalTexture: texture_2d<f32>;

@group(0) @binding(5)
var rayCountTexture: texture_2d<f32>;

@group(0) @binding(6)
var textureSampler: sampler;

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

    out.output0 = gaussianBlur(in.texCoord, 0u);
    out.output1 = gaussianBlur(in.texCoord, 1u);
    out.output2 = gaussianBlur(in.texCoord, 2u);

/*
    let textureSize: vec2<u32> = textureDimensions(inputTexture, 0);
    let invTexSize: vec2<f32> = vec2<f32>(
        1.0f / f32(textureSize.x), 
        1.0f / f32(textureSize.y));
   
    var texCoords: vec2<f32> = vec2<f32>(
        in.texCoord.x * f32(textureSize.x) - 0.5f,
        in.texCoord.y * f32(textureSize.y) - 0.5f);

    let fxy: vec2<f32> = fract(texCoords);
    texCoords -= fxy;

    let xcubic: vec4<f32> = cubic(fxy.x);
    let ycubic: vec4<f32> = cubic(fxy.y);

    let c: vec4<f32> = texCoords.xxyy + vec2<f32>(-0.5f, 1.5f).xyxy;
    
    let s: vec4<f32> = vec4<f32>(xcubic.xz + xcubic.yw, ycubic.xz + ycubic.yw);
    var offset: vec4<f32> = c + vec4 (xcubic.yw, ycubic.yw) / s;
    offset *= invTexSize.xxyy;
    
    let sample0: vec4<f32> = textureSample(inputTexture, textureSampler, offset.xz);
    let sample1: vec4<f32> = textureSample(inputTexture, textureSampler, offset.yz);
    let sample2: vec4<f32> = textureSample(inputTexture, textureSampler, offset.xw);
    let sample3: vec4<f32> = textureSample(inputTexture, textureSampler, offset.yw);

    let sx: f32 = s.x / (s.x + s.y);
    let sy: f32 = s.z / (s.z + s.w);

    out.output = mix(
        mix(sample3, sample2, sx), 
        mix(sample1, sample0, sx), 
        sy);
*/

    return out;
}

///// from http://www.java-gaming.org/index.php?topic=35123.0
fn cubic(v: f32) -> vec4<f32>
{
    let n: vec4<f32> = vec4<f32>(1.0f, 2.0f, 3.0f, 4.0f) - v;
    let s: vec4<f32> = n * n * n;
    let x: f32 = s.x;
    let y: f32 = s.y - 4.0f * s.x;
    let z: f32 = s.z - 4.0f * s.y + 6.0f * s.x;
    let w: f32 = 6.0f - x - y - z;
    return vec4<f32>(x, y, z, w) * (1.0f/6.0f);
}

fn gaussian(
    i: vec2<f32>,
    sigma: f32) -> f32 
{
    let iOverSigma: vec2<f32> = i / sigma;
    return exp( -0.5f* dot(iOverSigma,i) ) / ( 6.28f * sigma*sigma );
}

/////
fn gaussianBlur(
    texCoord: vec2<f32>,
    iTextureID: u32) -> vec4<f32>
{
    let fTwoPI: f32 = 6.283185f;
    let fQuality: f32 = 8.0f;
    let kfPlaneThreshold: f32 = 1.0f;
    let kfPositionThreshold: f32 = 0.1f;
    let kfAngleThreshold: f32 = 0.9f;
    var fRadius: f32 = 16.0f;

    let rayCount: vec2<f32> = textureSample(
        rayCountTexture,
        textureSampler,
        texCoord
    ).xy;

    let fMult: f32 = 4.0f - clamp(rayCount.y / 40.0f, 0.0f, 3.5f); 
    fRadius *= fMult;

    var fOneOverScreenWidth: f32 = 1.0f / 640.0f;
    var fOneOverScreenHeight: f32 = 1.0f / 480.0f;

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        texCoord
    );

    let normal: vec3<f32> = textureSample(
        normalTexture,
        textureSampler,
        texCoord
    ).xyz;

    let fPlaneD: f32 = dot(normal, worldPosition.xyz) * -1.0f;

    var ret: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    let iMesh: u32 = u32(worldPosition.w - fract(worldPosition.w));

    if(iTextureID == 0)
    {
        ret = textureSample(
            inputTexture0,
            textureSampler,
            texCoord).xyz;
    }
    else if(iTextureID == 1)
    {
        ret = textureSample(
            inputTexture1,
            textureSampler,
            texCoord).xyz;
    }
    else
    {
        ret = textureSample(
            inputTexture2,
            textureSampler,
            texCoord).xyz;

        fRadius = 7.0f;
    }

    var screenUV: vec2<f32> = texCoord.xy;
    var fCount: f32 = 1.0f;
    for(var fD: f32 = 0.0f; fD < 3.14159f; fD += fTwoPI / 16.0f)
    {
        for(var i: f32 = 1.0f / fQuality; i <= 1.0f; i += 1.0f / fQuality)
        {
            let offset: vec2<f32> = vec2<f32>(
                cos(fD) * fOneOverScreenWidth,
                sin(fD) * fOneOverScreenHeight
            ) * fRadius * i;

            let sampleUV: vec2<f32> = screenUV + offset;
            let sampleWorldPosition: vec4<f32> = textureSample(
                worldPositionTexture,
                textureSampler,
                sampleUV
            );
            let sampleNormal: vec3<f32> = textureSample(
                normalTexture,
                textureSampler,
                sampleUV
            ).xyz;
            let diff: vec3<f32> = sampleWorldPosition.xyz - worldPosition.xyz;

            let iCheckMesh: u32 = u32(sampleWorldPosition.w - fract(sampleWorldPosition.w));
            let fPlaneDistance: f32 = dot(diff, normal) + fPlaneD;
            let fDP: f32 = dot(sampleNormal, normal);
            if(length(diff) > kfPositionThreshold ||
               abs(fDP) < kfAngleThreshold ||
               iMesh != iCheckMesh)
            {
                continue;
            }

            var radiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
            if(iTextureID == 0u)
            {
                radiance = textureSample(
                    inputTexture0,
                    textureSampler,
                    screenUV + offset  
                ).xyz;
            }
            else if(iTextureID == 1u)
            {
                radiance = textureSample(
                    inputTexture1,
                    textureSampler,
                    screenUV + offset  
                ).xyz;
            }
            else
            {
                radiance = textureSample(
                    inputTexture2,
                    textureSampler,
                    screenUV + offset  
                ).xyz;
            }

            ret += radiance;

            fCount += 1.0f;
        }
    }

    ret /= fCount;

    return vec4<f32>(ret.xyz, 1.0f);
}