struct UniformData
{
    mViewProjectionMatrix: mat4x4<f32>,
    mPrevViewProjectionMatrix: mat4x4<f32>,
    mViewMatrix: mat4x4<f32>,

    mJitteredViewProjectionMatrix: mat4x4<f32>,
    mPrevJitteredViewProjectionMatrix: mat4x4<f32>,
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


@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<uniform> defaultUniformData: DefaultUniformData;

struct VertexInput {
    @location(0) worldPosition : vec4<f32>,
    @location(1) texCoord: vec4<f32>,
    @location(2) normal : vec4<f32>
};
struct VertexOutput {
    @location(0) worldPosition: vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) motionVector: vec4<f32>,
};
struct FragmentOutput {
    @location(0) worldPosition : vec4<f32>,
    @location(1) texCoord : vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) motionVector: vec4<f32>,
    @location(4) clipSpace: vec4<f32>,
    @location(5) debug: vec4<f32>,
};


@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(in.worldPosition.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
    out.worldPosition = in.worldPosition;
    out.texCoord = in.texCoord.xy;
    out.normal = in.normal;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    out.worldPosition = in.worldPosition;
    out.texCoord = vec4<f32>(in.texCoord.x, in.texCoord.y, 0.0f, 1.0f);
    let normalXYZ: vec3<f32> = normalize(in.normal.xyz);
    out.normal.x = normalXYZ.x;
    out.normal.y = normalXYZ.y;
    out.normal.z = normalXYZ.z;
    out.normal.w = 1.0f;

    // store depth and mesh id in worldPosition.w
    out.worldPosition.w = clamp(in.pos.z, 0.0f, 0.999f) + floor(in.worldPosition.w + 0.5f);
    
    var currClipSpacePos: vec4<f32> = vec4<f32>(in.worldPosition.xyz, 1.0f) * defaultUniformData.mViewProjectionMatrix;
    currClipSpacePos.x /= currClipSpacePos.w;
    currClipSpacePos.y /= currClipSpacePos.w;
    currClipSpacePos.z /= currClipSpacePos.w;

    var clipSpace: vec3<f32> = currClipSpacePos.xyz;

    currClipSpacePos.x = currClipSpacePos.x * 0.5f + 0.5f;
    currClipSpacePos.y = currClipSpacePos.y * 0.5f + 0.5f;
    currClipSpacePos.z = currClipSpacePos.z * 0.5f + 0.5f;

    var prevClipSpacePos: vec4<f32> = vec4<f32>(in.worldPosition.xyz, 1.0f) * defaultUniformData.mPrevViewProjectionMatrix;
    prevClipSpacePos.x /= prevClipSpacePos.w;
    prevClipSpacePos.y /= prevClipSpacePos.w;
    prevClipSpacePos.z /= prevClipSpacePos.w;
    prevClipSpacePos.x = prevClipSpacePos.x * 0.5f + 0.5f;
    prevClipSpacePos.y = prevClipSpacePos.y * 0.5f + 0.5f;
    prevClipSpacePos.z = prevClipSpacePos.z * 0.5f + 0.5f;

    out.motionVector.x = (currClipSpacePos.x - prevClipSpacePos.x) * 0.5f + 0.5f;
    out.motionVector.y = 1.0f - ((currClipSpacePos.y - prevClipSpacePos.y) * 0.5f + 0.5f);
    out.motionVector.z = floor(in.worldPosition.w + 0.5f);      // mesh id
    out.motionVector.w = currClipSpacePos.z;                    // depth

    var xform: vec4<f32> = vec4<f32>(in.worldPosition.xyz, 1.0f) * defaultUniformData.mViewMatrix;
    var fDepth: f32 = xform.z / xform.w;

    // based on z from 0.1 to 100.0
    var frustumColor: vec4<f32> = vec4<f32>(1.0f, 1.0f, 0.0f, in.pos.z);
    if(in.pos.z < 0.5f)
    {
        // z = 0.1 to 1.0 
        frustumColor = vec4<f32>(1.0f, 0.0, 0.0, in.pos.z);
    }
    else if(in.pos.z < 0.75f)
    {
        // z = 1.0 to 4.0
        frustumColor = vec4<f32>(0.0f, 1.0f, 0.0f, in.pos.z);
    }
    else if(in.pos.z < 0.8777f)
    {
        // z = 4.0 to 8.0
        frustumColor = vec4<f32>(0.0f, 0.0f, 1.0f, in.pos.z);
    }
   
    out.clipSpace = vec4<f32>(clipSpace.xyz, 1.0f);

    return out;
}