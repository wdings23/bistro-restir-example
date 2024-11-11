const UINT32_MAX: u32 = 0xffffffffu;
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

@group(0) @binding(0)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var localNormalTexture: texture_2d<f32>;

@group(0) @binding(3)
var textureSampler: sampler;

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
struct FragmentOutput {
    @location(0) mWorldNormal: vec4<f32>,
    @location(1) mLocalNormal : vec4<f32>,
};


@vertex
fn vs_main(in: VertexInput) -> VertexOutput 
{
    var out: VertexOutput;
    out.pos = in.pos;
    out.texCoord = in.texCoord;
    
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    
    var out: FragmentOutput;
    
    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        in.texCoord.xy
    );

    let normal: vec4<f32> = textureSample(
        normalTexture,
        textureSampler,
        in.texCoord.xy
    );

    let localNormalTextureColor: vec4<f32> = textureSample(
        localNormalTexture,
        textureSampler,
        in.texCoord.xy
    );

    var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    if(abs(normal.y) > 0.98f)
    {
        up = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    let tangent: vec3<f32> = normalize(cross(up, normal.xyz));
    let binormal: vec3<f32> = normalize(cross(normal.xyz, tangent));

    let fPhi: f32 = (localNormalTextureColor.x * 2.0f - 1.0f) * PI; 
    let fTheta: f32 = localNormalTextureColor.y * 2.0f * PI;
    
    if(localNormalTextureColor.w < 1.0f)
    {
        out.mLocalNormal = vec4<f32>(normal.xyz, 1.0f);
    }
    else
    {
        out.mLocalNormal = vec4<f32>(
            sin(fPhi) * cos(fTheta),
            sin(fPhi) * sin(fTheta),
            cos(fPhi),
            1.0f
        );
    }

    let sampleNormal: vec3<f32> = normalize(tangent * out.mLocalNormal.x + binormal * out.mLocalNormal.y + normal.xyz * out.mLocalNormal.z);
    out.mWorldNormal = vec4<f32>(
        sampleNormal.xyz,
        1.0f
    );

    if(localNormalTextureColor.w < 1.0f)
    {
        out.mWorldNormal = vec4<f32>(
            normal.xyz,
            1.0f
        );
    }
    
    return out;
}