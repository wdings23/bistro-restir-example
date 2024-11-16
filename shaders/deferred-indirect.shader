const PI: f32 = 3.14159f;

struct UniformData
{
    miNumMeshes: u32,
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

struct Material
{
    mDiffuse: vec4<f32>,
    mSpecular: vec4<f32>,
    mEmissive: vec4<f32>,

    miID: u32,
    miAlbedoTextureID: u32,
    miNormalTextureID: u32,
    miEmissiveTextureID: u32,
};

struct Range
{
    miStart: i32,
    miEnd: i32,
};

@group(0) @binding(0)
var<uniform> uniformData: UniformData;

@group(0) @binding(1)
var<storage> aMaterials: array<Material>;

@group(0) @binding(2)
var<storage> aMeshMaterialID: array<u32>;

@group(0) @binding(3)
var<storage, read> aMeshTriangleIndexRanges: array<Range>;

@group(0) @binding(4)
var<uniform> defaultUniformData: DefaultUniformData;

//@group(0) @binding(5)
//var textureSampler: sampler;

//@group(0) @binding(6)
//var linearTextureSampler: sampler;

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
    @location(3) currClipSpacePos: vec4<f32>,
    @location(4) prevClipSpacePos: vec4<f32>,
};
struct FragmentOutput {
    @location(0) worldPosition : vec4<f32>,
    @location(1) texCoord : vec4<f32>,
    @location(2) normal: vec4<f32>,
    @location(3) motionVector: vec4<f32>,
    @location(4) clipSpace: vec4<f32>,
    @location(5) material: vec4<f32>,
    @location(6) roughMetal: vec4<f32>,
    @location(7) debug: vec4<f32>,
};


@vertex
fn vs_main(in: VertexInput,
    @builtin(vertex_index) iVertexIndex: u32) -> VertexOutput {
    var out: VertexOutput;
    out.pos = vec4<f32>(in.worldPosition.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
    out.worldPosition = in.worldPosition;
    out.texCoord = in.texCoord.xy;
    out.normal = in.normal;
    out.currClipSpacePos = vec4<f32>(in.worldPosition.xyz, 1.0f) * defaultUniformData.mViewProjectionMatrix;
    out.prevClipSpacePos = vec4<f32>(in.worldPosition.xyz, 1.0f) * defaultUniformData.mPrevJitteredViewProjectionMatrix;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    var out: FragmentOutput;
    
    let iMesh: u32 = u32(ceil(in.worldPosition.w - 0.5f));
    out.worldPosition = vec4<f32>(in.worldPosition.xyz, f32(iMesh));
    out.texCoord = vec4<f32>(in.texCoord.x, in.texCoord.y, 0.0f, 1.0f);
    let normalXYZ: vec3<f32> = normalize(in.normal.xyz);
    out.normal.x = normalXYZ.x;
    out.normal.y = normalXYZ.y;
    out.normal.z = normalXYZ.z;
    out.normal.w = 1.0f;

    var texCoord: vec2<f32> = in.texCoord.xy;
    if(texCoord.x < 0.0f)
    {
        texCoord.x = fract(abs(1.0f - texCoord.x));
    }
    else if(texCoord.x > 1.0f)
    {
        texCoord.x = fract(texCoord.x);
    }

    if(texCoord.y < 0.0f)
    {
        texCoord.y = fract(abs(1.0f - texCoord.y));
    }
    else if(texCoord.y > 1.0f)
    {
        texCoord.y = fract(texCoord.y);
    }

    out.texCoord = vec4<f32>(texCoord.x, texCoord.y, 0.0f, 1.0f);

    // store depth and mesh id in worldPosition.w
    out.worldPosition.w = clamp(in.pos.z, 0.0f, 0.999f) + floor(in.worldPosition.w + 0.5f);

    var currClipSpacePos: vec3<f32> = vec3<f32>(
        in.currClipSpacePos.x / in.currClipSpacePos.w,
        in.currClipSpacePos.y / in.currClipSpacePos.w,
        in.currClipSpacePos.z / in.currClipSpacePos.w
    ) * 0.5f + 
    vec3<f32>(0.5f);
    currClipSpacePos.y = 1.0f - currClipSpacePos.y;

    var prevClipSpacePos: vec3<f32> = vec3<f32>(
        in.prevClipSpacePos.x / in.prevClipSpacePos.w,
        in.prevClipSpacePos.y / in.prevClipSpacePos.w,
        in.prevClipSpacePos.z / in.prevClipSpacePos.w
    ) * 0.5f + 
    vec3<f32>(0.5f);
    prevClipSpacePos.y = 1.0f - prevClipSpacePos.y;

    out.motionVector.x = (currClipSpacePos.x - prevClipSpacePos.x); // * 0.5f + 0.5f;
    out.motionVector.y = (currClipSpacePos.y - prevClipSpacePos.y); // * 0.5f + 0.5f;
    out.motionVector.z = floor(in.worldPosition.w + 0.5f);      // mesh id
    out.motionVector.w = currClipSpacePos.z;                    // depth

    var xform: vec4<f32> = vec4<f32>(in.worldPosition.xyz, 1.0f) * defaultUniformData.mJitteredViewProjectionMatrix;
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
   
    out.clipSpace = vec4<f32>(currClipSpacePos.xyz, 1.0f);

    let iMaterialID: u32 = aMeshMaterialID[iMesh];
    let material: Material = aMaterials[iMaterialID-1];
    var textureColor: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f); 
    var normalTextureColor: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);

    var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    if(abs(normalXYZ.y) > 0.98f)
    {
        up = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    let tangent: vec3<f32> = normalize(cross(up, normalXYZ));
    let binormal: vec3<f32> = normalize(cross(normalXYZ, tangent));
    
    //let fPhi: f32 = ((normalTextureColor.x * 2.0f - 1.0f) * PI + PI) * 0.5f;
    //let fTheta: f32 =  (normalTextureColor.y * 2.0f - 1.0f) * PI;
    
    let fPhi: f32 = (normalTextureColor.x * 2.0f - 1.0f) * PI; // acos(1.0f - 2.0f * normalTextureColor.x);
    let fTheta: f32 = normalTextureColor.y * 2.0f * PI;

    let localNormal: vec3<f32> = vec3<f32>(
        sin(fPhi) * cos(fTheta),
        sin(fPhi) * sin(fTheta),
        cos(fPhi));
    
    let sampleNormal: vec3<f32> = normalize(tangent * localNormal.x + binormal * localNormal.y + normalXYZ * localNormal.z);

    //out.normal = vec4<f32>(sampleNormal.xyz, 1.0f);
    out.normal = vec4<f32>(normalXYZ, 1.0f);
    out.roughMetal = vec4<f32>(material.mSpecular.x, material.mSpecular.y, 0.0f, 0.0f);

    let fDP: f32 = max(dot(sampleNormal, normalize(vec3<f32>(1.0f, 1.0f, 1.0f))), 0.1f);
    out.debug = vec4<f32>(fDepth, fDepth, fDepth, 1.0f);

    return out;
}

