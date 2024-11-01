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
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var materialTexture: texture_2d<f32>;

@group(0) @binding(3)
var roughMetalTexture: texture_2d<f32>;

@group(0) @binding(4)
var skyTexture0: texture_2d<f32>;

@group(0) @binding(5)
var skyTexture1: texture_2d<f32>;

@group(0) @binding(6)
var skyTexture2: texture_2d<f32>;

@group(0) @binding(7)
var skyTexture3: texture_2d<f32>;

@group(0) @binding(8)
var skyTexture: texture_2d<f32>;

@group(0) @binding(9)
var skyLightTexture: texture_2d<f32>;

@group(0) @binding(10)
var brdfLUTTexture: texture_2d<f32>;

@group(0) @binding(11)
var lightViewDepthTexture0: texture_2d<f32>;

@group(0) @binding(12)
var lightViewDepthTexture1: texture_2d<f32>;

@group(0) @binding(13)
var lightViewDepthTexture2: texture_2d<f32>;

@group(0) @binding(14)
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

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture, 
        textureSampler, 
        in.texCoord);

    let normal: vec3<f32> = textureSample(
        normalTexture, 
        textureSampler, 
        in.texCoord).xyz;

    if(worldPosition.w <= 0.0f)
    {
        out.output = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        return out;
    }

    let skyUV: vec2<f32> = octahedronMap2(normal.xyz);
    let sky: vec4<f32> = textureSample(
        skyTexture0,
        textureSampler,
        skyUV
    );

    let skyLight: vec4<f32> = textureSample(
        skyLightTexture,
        textureSampler,
        skyUV
    );

    let material: vec4<f32> = textureSample(
        materialTexture,
        textureSampler,
        in.texCoord
    );

    let roughMetal: vec4<f32> = textureSample(
        roughMetalTexture,
        textureSampler,
        in.texCoord
    );

    let iMesh: u32 = u32(ceil(worldPosition.w - 0.5f));

    let ambient: vec3<f32> = pbrAmbientWithAlbedo(
        normal.xyz,
        defaultUniformData.mCameraPosition.xyz,
        worldPosition.xyz,
        abs(roughMetal.x - roughMetal.y),
        max(roughMetal.y + 0.2f, 1.0f),
        material.xyz);

    let lighting: vec3<f32> = pbrDirectLightWithAlbedo(
        normal.xyz,
        defaultUniformData.mCameraPosition.xyz,
        worldPosition.xyz,
        abs(roughMetal.x - roughMetal.y),
        max(roughMetal.y + 0.2f, 1.0f),
        material.xyz);

    let shadowColor: vec3<f32> = shadow(
        worldPosition, 
        normal.xyz,
        defaultUniformData.mLightDirection.xyz);

    out.output = vec4<f32>(
        (lighting * shadowColor + ambient), 
        1.0f
    );

    out.debug0 = vec4<f32>(lighting, 1.0f);
    out.debug1 = vec4<f32>(ambient, 1.0f);
    out.debug2 = vec4<f32>(shadowColor, 1.0f);

    return out;
}

/////
fn octahedronMap2(direction: vec3<f32>) -> vec2<f32>
{
    let fDP: f32 = dot(vec3<f32>(1.0f, 1.0f, 1.0f), abs(direction));
    let newDirection: vec3<f32> = vec3<f32>(direction.x, direction.z, direction.y) / fDP;

    var ret: vec2<f32> =
        vec2<f32>(
            (1.0f - abs(newDirection.z)) * sign(newDirection.x),
            (1.0f - abs(newDirection.x)) * sign(newDirection.z));
       
    if(newDirection.y < 0.0f)
    {
        ret = vec2<f32>(
            newDirection.x, 
            newDirection.z);
    }

    ret = ret * 0.5f + vec2<f32>(0.5f, 0.5f);
    ret.y = 1.0f - ret.y;

    return ret;
}

/////
fn decodeOctahedronMap(uv: vec2<f32>) -> vec3<f32>
{
    let newUV: vec2<f32> = uv * 2.0f - vec2<f32>(1.0f, 1.0f);

    let absUV: vec2<f32> = vec2<f32>(abs(newUV.x), abs(newUV.y));
    var v: vec3<f32> = vec3<f32>(newUV.x, newUV.y, 1.0f - (absUV.x + absUV.y));

    if(absUV.x + absUV.y > 1.0f) 
    {
        v.x = (abs(newUV.y) - 1.0f) * -sign(newUV.x);
        v.y = (abs(newUV.x) - 1.0f) * -sign(newUV.y);
    }

    v.y *= -1.0f;

    return v;
}

//////
fn DistributionGGX(
    N: vec3<f32>, 
    H: vec3<f32>, 
    roughness: f32) -> f32
{
    let a: f32 = roughness*roughness;
    let a2: f32 = a*a;
    let NdotH: f32 = max(dot(N, H), 0.0f);
    let NdotH2: f32 = NdotH*NdotH;

    let nom: f32   = a2;
    var denom: f32 = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;

    return nom / denom;
}
// ----------------------------------------------------------------------------
fn GeometrySchlickGGX(
    NdotV: f32, 
    roughness: f32) -> f32
{
    let r: f32 = (roughness + 1.0);
    let k: f32 = (r*r) / 8.0;

    let nom: f32   = NdotV;
    let denom: f32 = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
fn GeometrySmith(
    N: vec3<f32>, 
    V: vec3<f32>, 
    L: vec3<f32>, 
    roughness: f32) -> f32
{
    let NdotV: f32 = max(dot(N, V), 0.0f);
    let NdotL: f32 = max(dot(N, L), 0.0f);
    let ggx2: f32 = GeometrySchlickGGX(NdotV, roughness);
    let ggx1: f32 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
fn fresnelSchlick(
    cosTheta: f32, 
    F0: vec3<f32>) -> vec3<f32>
{
    return F0 + (1.0f - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}
// ----------------------------------------------------------------------------
fn fresnelSchlickRoughness(
    cosTheta: f32, 
    F0: vec3<f32>, 
    roughness: f32) -> vec3<f32>
{
    return F0 + (max(vec3<f32>(1.0f - roughness), F0) - F0) * pow(clamp(1.0f - cosTheta, 0.0f, 1.0f), 5.0f);
}   
// ----------------------------------------------------------------------------
fn pbrAmbientWithAlbedo(
    Normal: vec3<f32>,
    camPos: vec3<f32>,
    WorldPos: vec3<f32>, 
    roughness: f32, 
    metallic: f32,
    albedo: vec3<f32>) -> vec3<f32>
{		
    let N: vec3<f32> = Normal;
    let V: vec3<f32> = normalize(camPos - WorldPos);
    let R: vec3<f32> = reflect(-V, N); 

    var uv: vec2<f32> = octahedronMap2(N);
    let irradiance: vec3<f32> = 
        textureSample(
            skyTexture0, 
            textureSampler,
            uv).xyz;

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    var F0: vec3<f32> = vec3<f32>(0.04f, 0.04f, 0.04f); 
    F0 = mix(F0, albedo, metallic);

    // ambient lighting (we now use IBL as the ambient term)
    let F: vec3<f32> = fresnelSchlickRoughness(max(dot(N, V), 0.0), F0, roughness);
    
    let kS: vec3<f32> = F;
    var kD: vec3<f32> = 1.0f - kS;
    kD *= 1.0f - metallic;	  
    
    let diffuse: vec3<f32>      = irradiance; 
    
    // sample both the pre-filter map and the BRDF lut and combine them together as per the Split-Sum approximation to get the IBL specular part.
    let MAX_REFLECTION_LOD: f32 = 4.0f;
    var prefilteredColor: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    uv = octahedronMap2(R);
    if(roughness >= 0.7f)
    {
        prefilteredColor = textureSample(
            skyTexture0,
            textureSampler,
            uv
        ).xyz;
    }
    else if(roughness >= 0.5f)
    {
        prefilteredColor = textureSample(
            skyTexture1,
            textureSampler,
            uv
        ).xyz;
    }
    else if(roughness >= 0.25f)
    {
        prefilteredColor = textureSample(
            skyTexture2,
            textureSampler,
            uv
        ).xyz;
    }
    else if(roughness >= 0.1f)
    {
        prefilteredColor = textureSample(
            skyTexture3,
            textureSampler,
            uv
        ).xyz;
    }
    else
    {
        prefilteredColor = textureSample(
            skyTexture,
            textureSampler,
            uv
        ).xyz;
    }
       
    let brdf: vec2<f32> = textureSample(
        brdfLUTTexture, 
        textureSampler,
        vec2(max(dot(N, V), 0.0), roughness)).xy;
    let specular: vec3<f32> = prefilteredColor * (F * brdf.x + brdf.y);
    let ambient: vec3<f32> = (kD * diffuse + specular);
    
    return ambient;
}

/////
fn pbrDirectLightWithAlbedo(
    Normal: vec3<f32>,
    camPos: vec3<f32>,
    WorldPos: vec3<f32>, 
    roughness: f32, 
    metallic: f32,
    albedo: vec3<f32>) -> vec3<f32>
{
    let V: vec3<f32> = normalize(camPos - WorldPos);
    let skyUV: vec2<f32> = octahedronMap2(Normal.xyz);
    
    // calculate per-light radiance
    let L: vec3<f32> = defaultUniformData.mLightDirection.xyz;
    let H: vec3<f32> = normalize(V + L);
    let irradiance: vec3<f32> = textureSample(
        skyLightTexture,
        textureSampler,
        skyUV
    ).xyz;

    // calculate reflectance at normal incidence; if dia-electric (like plastic) use F0 
    // of 0.04 and if it's a metal, use the albedo color as F0 (metallic workflow)    
    var F0: vec3<f32> = vec3<f32>(0.04f, 0.04f, 0.04f); 
    F0 = mix(F0, albedo, metallic);

    // Cook-Torrance BRDF
    let NDF: f32 = DistributionGGX(Normal, H, roughness);   
    let G: f32   = GeometrySmith(Normal, V, L, roughness);    
    let F: vec3<f32>    = fresnelSchlick(max(dot(H, V), 0.0f), F0);        
    
    let numerator: vec3<f32>    = NDF * G * F;
    let denominator: f32 = 4.0f * max(dot(Normal, V), 0.0f) * max(dot(Normal, L), 0.0f) + 0.0001f; // + 0.0001 to prevent divide by zero
    let specular: vec3<f32> = numerator / denominator;
    
        // kS is equal to Fresnel
    let kS: vec3<f32> = F;
    // for energy conservation, the diffuse and specular light can't
    // be above 1.0 (unless the surface emits light); to preserve this
    // relationship the diffuse component (kD) should equal 1.0 - kS.
    var kD: vec3<f32> = vec3<f32>(1.0f) - kS;
    // multiply kD by the inverse metalness such that only non-metals 
    // have diffuse lighting, or a linear blend if partly metal (pure metals
    // have no diffuse light).
    kD *= 1.0f - metallic;	                
        
    // scale light by NdotL
    let NdotL: f32 = max(dot(Normal, L), 0.0f);        

    // add to outgoing radiance Lo
    let ret: vec3<f32> = (kD / PI + specular) * irradiance * NdotL; // note that we already multiplied the BRDF by the Fresnel (kS) so we won't multiply by kS again
    return ret;
}

/////
fn shadow(
    worldPosition: vec4<f32>,
    normal: vec3<f32>,
    lightDirection: vec3<f32>
) -> vec3<f32>
{
    let kfBias: f32 = 0.01f;
    var ret: vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f);
    let fBias: f32 = kfBias * -sign(dot(normal, lightDirection));
    
    // frustum 0
    // transform world position to light view clip space position
    var ret0: vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f);
    var clipSpacePositionFromLightView0: vec4<f32> = 
        vec4<f32>(worldPosition.xyz, 1.0f) * 
        uniformData.maLightViewProjectionMatrix[0];
    clipSpacePositionFromLightView0.x /= clipSpacePositionFromLightView0.w;
    clipSpacePositionFromLightView0.y /= clipSpacePositionFromLightView0.w;
    clipSpacePositionFromLightView0.z /= clipSpacePositionFromLightView0.w;
    let screenCoord0: vec2<f32> = vec2<f32>(
        clipSpacePositionFromLightView0.x * 0.5f + 0.5f,
        1.0f - (clipSpacePositionFromLightView0.y * 0.5f + 0.5f)
    );
    if(screenCoord0.x >= 0.0f && screenCoord0.x <= 1.0f &&
       screenCoord0.y >= 0.0f && screenCoord0.y <= 1.0f)
    {
        let lightViewDepth0: vec4<f32> = textureSample(
            lightViewDepthTexture0,
            textureSampler,
            screenCoord0
        );
        if(lightViewDepth0.w > 0.0f && clipSpacePositionFromLightView0.z + fBias * 0.5f > lightViewDepth0.x)
        {
            ret0 = vec3<f32>(0.0f, 0.0f, 0.0f);
        }
    }
    
    // frustum 1
    // transform world position to light view clip space position
    var ret1: vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f);
    var clipSpacePositionFromLightView1: vec4<f32> = 
        vec4<f32>(worldPosition.xyz, 1.0f) * 
        uniformData.maLightViewProjectionMatrix[1];
    clipSpacePositionFromLightView1.x /= clipSpacePositionFromLightView1.w;
    clipSpacePositionFromLightView1.y /= clipSpacePositionFromLightView1.w;
    clipSpacePositionFromLightView1.z /= clipSpacePositionFromLightView1.w;
    let screenCoord1: vec2<f32> = vec2<f32>(
        clipSpacePositionFromLightView1.x * 0.5f + 0.5f,
        1.0f - (clipSpacePositionFromLightView1.y * 0.5f + 0.5f)
    );
    // check against the current depth from light's view space
    if(screenCoord1.x >= 0.0f && screenCoord1.x <= 1.0f &&
       screenCoord1.y >= 0.0f && screenCoord1.y <= 1.0f)
    {
        let lightViewDepth1: vec4<f32> = textureSample(
            lightViewDepthTexture1,
            textureSampler,
            screenCoord1
        );
        if(lightViewDepth1.w > 0.0f && clipSpacePositionFromLightView1.z + fBias > lightViewDepth1.x)
        {
            ret1 = vec3<f32>(0.0f, 0.0f, 0.0f);
        }
    }
       
    // Frsutum 2
    // transform world position to light view clip space position
    var ret2: vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f);
    var clipSpacePositionFromLightView2: vec4<f32> = 
        vec4<f32>(worldPosition.xyz, 1.0f) * 
        uniformData.maLightViewProjectionMatrix[2];
    clipSpacePositionFromLightView2.x /= clipSpacePositionFromLightView2.w;
    clipSpacePositionFromLightView2.y /= clipSpacePositionFromLightView2.w;
    clipSpacePositionFromLightView2.z /= clipSpacePositionFromLightView2.w;
    let screenCoord2: vec2<f32> = vec2<f32>(
        clipSpacePositionFromLightView2.x * 0.5f + 0.5f,
        1.0f - (clipSpacePositionFromLightView2.y * 0.5f + 0.5f)
    );
    // check against the current depth from light's view space
    if(screenCoord2.x >= 0.0f && screenCoord2.x <= 1.0f &&
       screenCoord2.y >= 0.0f && screenCoord2.y <= 1.0f)
    {
        let lightViewDepth2: vec4<f32> = textureSample(
            lightViewDepthTexture2,
            textureSampler,
            screenCoord2
        );
        if(lightViewDepth2.w > 0.0f && clipSpacePositionFromLightView2.z + fBias > lightViewDepth2.x)
        {
            ret2 = vec3<f32>(0.0f, 0.0f, 0.0f);
        }
    }
        
    ret = ret0 * ret1 * ret2;
    
    return ret;
}