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

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage> aMaterials: array<Material>;

@group(1) @binding(2)
var<storage> aMeshMaterialID: array<u32>;

@group(1) @binding(3)
var<storage, read> aMeshTriangleIndexRanges: array<Range>;

@group(1) @binding(4)
var<uniform> defaultUniformData: DefaultUniformData;

@group(0) @binding(1)
var textureSampler: sampler;

@group(0) @binding(2)
var linearTextureSampler: sampler;

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
   
    out.clipSpace = vec4<f32>(currClipSpacePos.xyz, 1.0f);

    let iMaterialID: u32 = aMeshMaterialID[iMesh];
    let material: Material = aMaterials[iMaterialID-1];
    var textureColor: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f); 
    var normalTextureColor: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    if(material.miAlbedoTextureID >= 0 && material.miAlbedoTextureID < 100)
    {
        textureColor = sampleTexture0(
            material.miAlbedoTextureID, 
            in.texCoord);

        normalTextureColor = sampleNormalTexture0(
            material.miNormalTextureID,
            in.texCoord
        );
    }
    else 
    {
        textureColor = sampleTexture1(
            material.miAlbedoTextureID, 
            in.texCoord);

        normalTextureColor = sampleNormalTexture1(
            material.miNormalTextureID,
            in.texCoord
        );
    }

    out.material = textureColor;

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

    out.normal = vec4<f32>(sampleNormal.xyz, 1.0f);
    out.roughMetal = vec4<f32>(material.mSpecular.x, material.mSpecular.y, 0.0f, 0.0f);

    let fDP: f32 = max(dot(sampleNormal, normalize(vec3<f32>(1.0f, 1.0f, 1.0f))), 0.1f);
    out.debug = vec4<f32>(fDP * textureColor.x, fDP * textureColor.y, fDP * textureColor.z, 1.0f);

    return out;
}

@group(2) @binding(0)
var texture0: texture_2d<f32>;
@group(2) @binding(1)
var texture1: texture_2d<f32>;
@group(2) @binding(2)
var texture2: texture_2d<f32>;
@group(2) @binding(3)
var texture3: texture_2d<f32>;
@group(2) @binding(4)
var texture4: texture_2d<f32>;
@group(2) @binding(5)
var texture5: texture_2d<f32>;
@group(2) @binding(6)
var texture6: texture_2d<f32>;
@group(2) @binding(7)
var texture7: texture_2d<f32>;
@group(2) @binding(8)
var texture8: texture_2d<f32>;
@group(2) @binding(9)
var texture9: texture_2d<f32>;
@group(2) @binding(10)
var texture10: texture_2d<f32>;
@group(2) @binding(11)
var texture11: texture_2d<f32>;
@group(2) @binding(12)
var texture12: texture_2d<f32>;
@group(2) @binding(13)
var texture13: texture_2d<f32>;
@group(2) @binding(14)
var texture14: texture_2d<f32>;
@group(2) @binding(15)
var texture15: texture_2d<f32>;
@group(2) @binding(16)
var texture16: texture_2d<f32>;
@group(2) @binding(17)
var texture17: texture_2d<f32>;
@group(2) @binding(18)
var texture18: texture_2d<f32>;
@group(2) @binding(19)
var texture19: texture_2d<f32>;
@group(2) @binding(20)
var texture20: texture_2d<f32>;
@group(2) @binding(21)
var texture21: texture_2d<f32>;
@group(2) @binding(22)
var texture22: texture_2d<f32>;
@group(2) @binding(23)
var texture23: texture_2d<f32>;
@group(2) @binding(24)
var texture24: texture_2d<f32>;
@group(2) @binding(25)
var texture25: texture_2d<f32>;
@group(2) @binding(26)
var texture26: texture_2d<f32>;
@group(2) @binding(27)
var texture27: texture_2d<f32>;
@group(2) @binding(28)
var texture28: texture_2d<f32>;
@group(2) @binding(29)
var texture29: texture_2d<f32>;
@group(2) @binding(30)
var texture30: texture_2d<f32>;
@group(2) @binding(31)
var texture31: texture_2d<f32>;
@group(2) @binding(32)
var texture32: texture_2d<f32>;
@group(2) @binding(33)
var texture33: texture_2d<f32>;
@group(2) @binding(34)
var texture34: texture_2d<f32>;
@group(2) @binding(35)
var texture35: texture_2d<f32>;
@group(2) @binding(36)
var texture36: texture_2d<f32>;
@group(2) @binding(37)
var texture37: texture_2d<f32>;
@group(2) @binding(38)
var texture38: texture_2d<f32>;
@group(2) @binding(39)
var texture39: texture_2d<f32>;
@group(2) @binding(40)
var texture40: texture_2d<f32>;
@group(2) @binding(41)
var texture41: texture_2d<f32>;
@group(2) @binding(42)
var texture42: texture_2d<f32>;
@group(2) @binding(43)
var texture43: texture_2d<f32>;
@group(2) @binding(44)
var texture44: texture_2d<f32>;
@group(2) @binding(45)
var texture45: texture_2d<f32>;
@group(2) @binding(46)
var texture46: texture_2d<f32>;
@group(2) @binding(47)
var texture47: texture_2d<f32>;
@group(2) @binding(48)
var texture48: texture_2d<f32>;
@group(2) @binding(49)
var texture49: texture_2d<f32>;
@group(2) @binding(50)
var texture50: texture_2d<f32>;
@group(2) @binding(51)
var texture51: texture_2d<f32>;
@group(2) @binding(52)
var texture52: texture_2d<f32>;
@group(2) @binding(53)
var texture53: texture_2d<f32>;
@group(2) @binding(54)
var texture54: texture_2d<f32>;
@group(2) @binding(55)
var texture55: texture_2d<f32>;
@group(2) @binding(56)
var texture56: texture_2d<f32>;
@group(2) @binding(57)
var texture57: texture_2d<f32>;
@group(2) @binding(58)
var texture58: texture_2d<f32>;
@group(2) @binding(59)
var texture59: texture_2d<f32>;
@group(2) @binding(60)
var texture60: texture_2d<f32>;
@group(2) @binding(61)
var texture61: texture_2d<f32>;
@group(2) @binding(62)
var texture62: texture_2d<f32>;
@group(2) @binding(63)
var texture63: texture_2d<f32>;
@group(2) @binding(64)
var texture64: texture_2d<f32>;
@group(2) @binding(65)
var texture65: texture_2d<f32>;
@group(2) @binding(66)
var texture66: texture_2d<f32>;
@group(2) @binding(67)
var texture67: texture_2d<f32>;
@group(2) @binding(68)
var texture68: texture_2d<f32>;
@group(2) @binding(69)
var texture69: texture_2d<f32>;
@group(2) @binding(70)
var texture70: texture_2d<f32>;
@group(2) @binding(71)
var texture71: texture_2d<f32>;
@group(2) @binding(72)
var texture72: texture_2d<f32>;
@group(2) @binding(73)
var texture73: texture_2d<f32>;
@group(2) @binding(74)
var texture74: texture_2d<f32>;
@group(2) @binding(75)
var texture75: texture_2d<f32>;
@group(2) @binding(76)
var texture76: texture_2d<f32>;
@group(2) @binding(77)
var texture77: texture_2d<f32>;
@group(2) @binding(78)
var texture78: texture_2d<f32>;
@group(2) @binding(79)
var texture79: texture_2d<f32>;
@group(2) @binding(80)
var texture80: texture_2d<f32>;
@group(2) @binding(81)
var texture81: texture_2d<f32>;
@group(2) @binding(82)
var texture82: texture_2d<f32>;
@group(2) @binding(83)
var texture83: texture_2d<f32>;
@group(2) @binding(84)
var texture84: texture_2d<f32>;
@group(2) @binding(85)
var texture85: texture_2d<f32>;
@group(2) @binding(86)
var texture86: texture_2d<f32>;
@group(2) @binding(87)
var texture87: texture_2d<f32>;
@group(2) @binding(88)
var texture88: texture_2d<f32>;
@group(2) @binding(89)
var texture89: texture_2d<f32>;
@group(2) @binding(90)
var texture90: texture_2d<f32>;
@group(2) @binding(91)
var texture91: texture_2d<f32>;
@group(2) @binding(92)
var texture92: texture_2d<f32>;
@group(2) @binding(93)
var texture93: texture_2d<f32>;
@group(2) @binding(94)
var texture94: texture_2d<f32>;
@group(2) @binding(95)
var texture95: texture_2d<f32>;
@group(2) @binding(96)
var texture96: texture_2d<f32>;
@group(2) @binding(97)
var texture97: texture_2d<f32>;
@group(2) @binding(98)
var texture98: texture_2d<f32>;
@group(2) @binding(99)
var texture99: texture_2d<f32>;
@group(3) @binding(0)
var texture100: texture_2d<f32>;
@group(3) @binding(1)
var texture101: texture_2d<f32>;
@group(3) @binding(2)
var texture102: texture_2d<f32>;
@group(3) @binding(3)
var texture103: texture_2d<f32>;
@group(3) @binding(4)
var texture104: texture_2d<f32>;
@group(3) @binding(5)
var texture105: texture_2d<f32>;
@group(3) @binding(6)
var texture106: texture_2d<f32>;
@group(3) @binding(7)
var texture107: texture_2d<f32>;
@group(3) @binding(8)
var texture108: texture_2d<f32>;
@group(3) @binding(9)
var texture109: texture_2d<f32>;
@group(3) @binding(10)
var texture110: texture_2d<f32>;
@group(3) @binding(11)
var texture111: texture_2d<f32>;
@group(3) @binding(12)
var texture112: texture_2d<f32>;
@group(3) @binding(13)
var texture113: texture_2d<f32>;
@group(3) @binding(14)
var texture114: texture_2d<f32>;
@group(3) @binding(15)
var texture115: texture_2d<f32>;
@group(3) @binding(16)
var texture116: texture_2d<f32>;
@group(3) @binding(17)
var texture117: texture_2d<f32>;
@group(3) @binding(18)
var texture118: texture_2d<f32>;
@group(3) @binding(19)
var texture119: texture_2d<f32>;
@group(3) @binding(20)
var texture120: texture_2d<f32>;
@group(3) @binding(21)
var texture121: texture_2d<f32>;
@group(3) @binding(22)
var texture122: texture_2d<f32>;
@group(3) @binding(23)
var texture123: texture_2d<f32>;
@group(3) @binding(24)
var texture124: texture_2d<f32>;
@group(3) @binding(25)
var texture125: texture_2d<f32>;
@group(3) @binding(26)
var texture126: texture_2d<f32>;
@group(3) @binding(27)
var texture127: texture_2d<f32>;
@group(3) @binding(28)
var texture128: texture_2d<f32>;
@group(3) @binding(29)
var texture129: texture_2d<f32>;
@group(3) @binding(30)
var texture130: texture_2d<f32>;
@group(3) @binding(31)
var texture131: texture_2d<f32>;

/////
fn sampleTexture0(
    iTextureID: u32,
    uv: vec2<f32>) -> vec4<f32>
{
    var ret: vec4<f32> = vec4<f32>(1.0f, 0.0f, 0.0f, 1.0f);
    if(iTextureID == 0u)
    {
        ret = textureSample(texture0, linearTextureSampler, uv);
    }
    else if(iTextureID == 1u)
    {
        ret = textureSample(texture1, linearTextureSampler, uv);
    }
    else if(iTextureID == 2u)
    {
        ret = textureSample(texture2, linearTextureSampler, uv);
    }
    else if(iTextureID == 3u)
    {
        ret = textureSample(texture3, linearTextureSampler, uv);
    }
    else if(iTextureID == 4u)
    {
        ret = textureSample(texture4, linearTextureSampler, uv);
    }
    else if(iTextureID == 5u)
    {
        ret = textureSample(texture5, linearTextureSampler, uv);
    }
    else if(iTextureID == 6u)
    {
        ret = textureSample(texture6, linearTextureSampler, uv);
    }
    else if(iTextureID == 7u)
    {
        ret = textureSample(texture7, linearTextureSampler, uv);
    }
    else if(iTextureID == 8u)
    {
        ret = textureSample(texture8, linearTextureSampler, uv);
    }
    else if(iTextureID == 9u)
    {
        ret = textureSample(texture9, linearTextureSampler, uv);
    }
    else if(iTextureID == 10u)
    {
        ret = textureSample(texture10, linearTextureSampler, uv);
    }
    else if(iTextureID == 11u)
    {
        ret = textureSample(texture11, linearTextureSampler, uv);
    }
    else if(iTextureID == 12u)
    {
        ret = textureSample(texture12, linearTextureSampler, uv);
    }
    else if(iTextureID == 13u)
    {
        ret = textureSample(texture13, linearTextureSampler, uv);
    }
    else if(iTextureID == 14u)
    {
        ret = textureSample(texture14, linearTextureSampler, uv);
    }
    else if(iTextureID == 15u)
    {
        ret = textureSample(texture15, linearTextureSampler, uv);
    }
    else if(iTextureID == 16u)
    {
        ret = textureSample(texture16, linearTextureSampler, uv);
    }
    else if(iTextureID == 17u)
    {
        ret = textureSample(texture17, linearTextureSampler, uv);
    }
    else if(iTextureID == 18u)
    {
        ret = textureSample(texture18, linearTextureSampler, uv);
    }
    else if(iTextureID == 19u)
    {
        ret = textureSample(texture19, linearTextureSampler, uv);
    }
    else if(iTextureID == 20u)
    {
        ret = textureSample(texture20, linearTextureSampler, uv);
    }
    else if(iTextureID == 21u)
    {
        ret = textureSample(texture21, linearTextureSampler, uv);
    }
    else if(iTextureID == 22u)
    {
        ret = textureSample(texture22, linearTextureSampler, uv);
    }
    else if(iTextureID == 23u)
    {
        ret = textureSample(texture23, linearTextureSampler, uv);
    }
    else if(iTextureID == 24u)
    {
        ret = textureSample(texture24, linearTextureSampler, uv);
    }
    else if(iTextureID == 25u)
    {
        ret = textureSample(texture25, linearTextureSampler, uv);
    }
    else if(iTextureID == 26u)
    {
        ret = textureSample(texture26, linearTextureSampler, uv);
    }
    else if(iTextureID == 27u)
    {
        ret = textureSample(texture27, linearTextureSampler, uv);
    }
    else if(iTextureID == 28u)
    {
        ret = textureSample(texture28, linearTextureSampler, uv);
    }
    else if(iTextureID == 29u)
    {
        ret = textureSample(texture29, linearTextureSampler, uv);
    }
    else if(iTextureID == 30u)
    {
        ret = textureSample(texture30, linearTextureSampler, uv);
    }
    else if(iTextureID == 31u)
    {
        ret = textureSample(texture31, linearTextureSampler, uv);
    }
    else if(iTextureID == 32u)
    {
        ret = textureSample(texture32, linearTextureSampler, uv);
    }
    else if(iTextureID == 33u)
    {
        ret = textureSample(texture33, linearTextureSampler, uv);
    }
    else if(iTextureID == 34u)
    {
        ret = textureSample(texture34, linearTextureSampler, uv);
    }
    else if(iTextureID == 35u)
    {
        ret = textureSample(texture35, linearTextureSampler, uv);
    }
    else if(iTextureID == 36u)
    {
        ret = textureSample(texture36, linearTextureSampler, uv);
    }
    else if(iTextureID == 37u)
    {
        ret = textureSample(texture37, linearTextureSampler, uv);
    }
    else if(iTextureID == 38u)
    {
        ret = textureSample(texture38, linearTextureSampler, uv);
    }
    else if(iTextureID == 39u)
    {
        ret = textureSample(texture39, linearTextureSampler, uv);
    }
    else if(iTextureID == 40u)
    {
        ret = textureSample(texture40, linearTextureSampler, uv);
    }
    else if(iTextureID == 41u)
    {
        ret = textureSample(texture41, linearTextureSampler, uv);
    }
    else if(iTextureID == 42u)
    {
        ret = textureSample(texture42, linearTextureSampler, uv);
    }
    else if(iTextureID == 43u)
    {
        ret = textureSample(texture43, linearTextureSampler, uv);
    }
    else if(iTextureID == 44u)
    {
        ret = textureSample(texture44, linearTextureSampler, uv);
    }
    else if(iTextureID == 45u)
    {
        ret = textureSample(texture45, linearTextureSampler, uv);
    }
    else if(iTextureID == 46u)
    {
        ret = textureSample(texture46, linearTextureSampler, uv);
    }
    else if(iTextureID == 47u)
    {
        ret = textureSample(texture47, linearTextureSampler, uv);
    }
    else if(iTextureID == 48u)
    {
        ret = textureSample(texture48, linearTextureSampler, uv);
    }
    else if(iTextureID == 49u)
    {
        ret = textureSample(texture49, linearTextureSampler, uv);
    }
    else if(iTextureID == 50u)
    {
        ret = textureSample(texture50, linearTextureSampler, uv);
    }
    else if(iTextureID == 51u)
    {
        ret = textureSample(texture51, linearTextureSampler, uv);
    }
    else if(iTextureID == 52u)
    {
        ret = textureSample(texture52, linearTextureSampler, uv);
    }
    else if(iTextureID == 53u)
    {
        ret = textureSample(texture53, linearTextureSampler, uv);
    }
    else if(iTextureID == 54u)
    {
        ret = textureSample(texture54, linearTextureSampler, uv);
    }
    else if(iTextureID == 55u)
    {
        ret = textureSample(texture55, linearTextureSampler, uv);
    }
    else if(iTextureID == 56u)
    {
        ret = textureSample(texture56, linearTextureSampler, uv);
    }
    else if(iTextureID == 57u)
    {
        ret = textureSample(texture57, linearTextureSampler, uv);
    }
    else if(iTextureID == 58u)
    {
        ret = textureSample(texture58, linearTextureSampler, uv);
    }
    else if(iTextureID == 59u)
    {
        ret = textureSample(texture59, linearTextureSampler, uv);
    }
    else if(iTextureID == 60u)
    {
        ret = textureSample(texture60, linearTextureSampler, uv);
    }
    else if(iTextureID == 61u)
    {
        ret = textureSample(texture61, linearTextureSampler, uv);
    }
    else if(iTextureID == 62u)
    {
        ret = textureSample(texture62, linearTextureSampler, uv);
    }
    else if(iTextureID == 63u)
    {
        ret = textureSample(texture63, linearTextureSampler, uv);
    }
    else if(iTextureID == 64u)
    {
        ret = textureSample(texture64, linearTextureSampler, uv);
    }
    else if(iTextureID == 65u)
    {
        ret = textureSample(texture65, linearTextureSampler, uv);
    }
    else if(iTextureID == 66u)
    {
        ret = textureSample(texture66, linearTextureSampler, uv);
    }
    else if(iTextureID == 67u)
    {
        ret = textureSample(texture67, linearTextureSampler, uv);
    }
    else if(iTextureID == 68u)
    {
        ret = textureSample(texture68, linearTextureSampler, uv);
    }
    else if(iTextureID == 69u)
    {
        ret = textureSample(texture69, linearTextureSampler, uv);
    }
    else if(iTextureID == 70u)
    {
        ret = textureSample(texture70, linearTextureSampler, uv);
    }
    else if(iTextureID == 71u)
    {
        ret = textureSample(texture71, linearTextureSampler, uv);
    }
    else if(iTextureID == 72u)
    {
        ret = textureSample(texture72, linearTextureSampler, uv);
    }
    else if(iTextureID == 73u)
    {
        ret = textureSample(texture73, linearTextureSampler, uv);
    }
    else if(iTextureID == 74u)
    {
        ret = textureSample(texture74, linearTextureSampler, uv);
    }
    else if(iTextureID == 75u)
    {
        ret = textureSample(texture75, linearTextureSampler, uv);
    }
    else if(iTextureID == 76u)
    {
        ret = textureSample(texture76, linearTextureSampler, uv);
    }
    else if(iTextureID == 77u)
    {
        ret = textureSample(texture77, linearTextureSampler, uv);
    }
    else if(iTextureID == 78u)
    {
        ret = textureSample(texture78, linearTextureSampler, uv);
    }
    else if(iTextureID == 79u)
    {
        ret = textureSample(texture79, linearTextureSampler, uv);
    }
    else if(iTextureID == 80u)
    {
        ret = textureSample(texture80, linearTextureSampler, uv);
    }
    else if(iTextureID == 81u)
    {
        ret = textureSample(texture81, linearTextureSampler, uv);
    }
    else if(iTextureID == 82u)
    {
        ret = textureSample(texture82, linearTextureSampler, uv);
    }
    else if(iTextureID == 83u)
    {
        ret = textureSample(texture83, linearTextureSampler, uv);
    }
    else if(iTextureID == 84u)
    {
        ret = textureSample(texture84, linearTextureSampler, uv);
    }
    else if(iTextureID == 85u)
    {
        ret = textureSample(texture85, linearTextureSampler, uv);
    }
    else if(iTextureID == 86u)
    {
        ret = textureSample(texture86, linearTextureSampler, uv);
    }
    else if(iTextureID == 87u)
    {
        ret = textureSample(texture87, linearTextureSampler, uv);
    }
    else if(iTextureID == 88u)
    {
        ret = textureSample(texture88, linearTextureSampler, uv);
    }
    else if(iTextureID == 89u)
    {
        ret = textureSample(texture89, linearTextureSampler, uv);
    }
    else if(iTextureID == 90u)
    {
        ret = textureSample(texture90, linearTextureSampler, uv);
    }
    else if(iTextureID == 91u)
    {
        ret = textureSample(texture91, linearTextureSampler, uv);
    }
    else if(iTextureID == 92u)
    {
        ret = textureSample(texture92, linearTextureSampler, uv);
    }
    else if(iTextureID == 93u)
    {
        ret = textureSample(texture93, linearTextureSampler, uv);
    }
    else if(iTextureID == 94u)
    {
        ret = textureSample(texture94, linearTextureSampler, uv);
    }
    else if(iTextureID == 95u)
    {
        ret = textureSample(texture95, linearTextureSampler, uv);
    }
    else if(iTextureID == 96u)
    {
        ret = textureSample(texture96, linearTextureSampler, uv);
    }
    else if(iTextureID == 97u)
    {
        ret = textureSample(texture97, linearTextureSampler, uv);
    }
    else if(iTextureID == 98u)
    {
        ret = textureSample(texture98, linearTextureSampler, uv);
    }
    else if(iTextureID == 99u)
    {
        ret = textureSample(texture99, linearTextureSampler, uv);
    }
    else 
    {
        ret = textureSample(texture0, linearTextureSampler, uv);
    }

    return ret;
}


/////
fn sampleTexture1(
    iTextureID: u32,
    uv: vec2<f32>) -> vec4<f32>
{
    var ret: vec4<f32> = vec4<f32>(1.0f, 0.0f, 0.0f, 1.0f);
    if(iTextureID == 100u)
    {
        ret = textureSample(texture100, linearTextureSampler, uv);
    }
    else if(iTextureID == 101u)
    {
        ret = textureSample(texture101, linearTextureSampler, uv);
    }
    else if(iTextureID == 102u)
    {
        ret = textureSample(texture102, linearTextureSampler, uv);
    }
    else if(iTextureID == 103u)
    {
        ret = textureSample(texture103, linearTextureSampler, uv);
    }
    else if(iTextureID == 104u)
    {
        ret = textureSample(texture104, linearTextureSampler, uv);
    }
    else if(iTextureID == 105u)
    {
        ret = textureSample(texture105, linearTextureSampler, uv);
    }
    else if(iTextureID == 106u)
    {
        ret = textureSample(texture106, linearTextureSampler, uv);
    }
    else if(iTextureID == 107u)
    {
        ret = textureSample(texture107, linearTextureSampler, uv);
    }
    else if(iTextureID == 108u)
    {
        ret = textureSample(texture108, linearTextureSampler, uv);
    }
    else if(iTextureID == 109u)
    {
        ret = textureSample(texture109, linearTextureSampler, uv);
    }
    else if(iTextureID == 110u)
    {
        ret = textureSample(texture110, linearTextureSampler, uv);
    }
    else if(iTextureID == 111u)
    {
        ret = textureSample(texture111, linearTextureSampler, uv);
    }
    else if(iTextureID == 112u)
    {
        ret = textureSample(texture112, linearTextureSampler, uv);
    }
    else if(iTextureID == 113u)
    {
        ret = textureSample(texture113, linearTextureSampler, uv);
    }
    else if(iTextureID == 114u)
    {
        ret = textureSample(texture114, linearTextureSampler, uv);
    }
    else if(iTextureID == 115u)
    {
        ret = textureSample(texture115, linearTextureSampler, uv);
    }
    else if(iTextureID == 116u)
    {
        ret = textureSample(texture116, linearTextureSampler, uv);
    }
    else if(iTextureID == 117u)
    {
        ret = textureSample(texture117, linearTextureSampler, uv);
    }
    else if(iTextureID == 118u)
    {
        ret = textureSample(texture118, linearTextureSampler, uv);
    }
    else if(iTextureID == 119u)
    {
        ret = textureSample(texture119, linearTextureSampler, uv);
    }
    else if(iTextureID == 120u)
    {
        ret = textureSample(texture120, linearTextureSampler, uv);
    }
    else if(iTextureID == 121u)
    {
        ret = textureSample(texture121, linearTextureSampler, uv);
    }
    else if(iTextureID == 122u)
    {
        ret = textureSample(texture122, linearTextureSampler, uv);
    }
    else if(iTextureID == 123u)
    {
        ret = textureSample(texture123, linearTextureSampler, uv);
    }
    else if(iTextureID == 124u)
    {
        ret = textureSample(texture124, linearTextureSampler, uv);
    }
    else if(iTextureID == 125u)
    {
        ret = textureSample(texture125, linearTextureSampler, uv);
    }
    else if(iTextureID == 126u)
    {
        ret = textureSample(texture126, linearTextureSampler, uv);
    }
    else if(iTextureID == 127u)
    {
        ret = textureSample(texture127, linearTextureSampler, uv);
    }
    else if(iTextureID == 128u)
    {
        ret = textureSample(texture128, linearTextureSampler, uv);
    }
    else if(iTextureID == 129u)
    {
        ret = textureSample(texture129, linearTextureSampler, uv);
    }
    else if(iTextureID == 130u)
    {
        ret = textureSample(texture130, linearTextureSampler, uv);
    }
    else if(iTextureID == 131u)
    {
        ret = textureSample(texture131, linearTextureSampler, uv);
    }
    else 
    {
        ret = textureSample(texture0, linearTextureSampler, uv);
    }

    return ret;
}

@group(4) @binding(0)
var normalTexture0: texture_2d<f32>;
@group(4) @binding(1)
var normalTexture1: texture_2d<f32>;
@group(4) @binding(2)
var normalTexture2: texture_2d<f32>;
@group(4) @binding(3)
var normalTexture3: texture_2d<f32>;
@group(4) @binding(4)
var normalTexture4: texture_2d<f32>;
@group(4) @binding(5)
var normalTexture5: texture_2d<f32>;
@group(4) @binding(6)
var normalTexture6: texture_2d<f32>;
@group(4) @binding(7)
var normalTexture7: texture_2d<f32>;
@group(4) @binding(8)
var normalTexture8: texture_2d<f32>;
@group(4) @binding(9)
var normalTexture9: texture_2d<f32>;
@group(4) @binding(10)
var normalTexture10: texture_2d<f32>;
@group(4) @binding(11)
var normalTexture11: texture_2d<f32>;
@group(4) @binding(12)
var normalTexture12: texture_2d<f32>;
@group(4) @binding(13)
var normalTexture13: texture_2d<f32>;
@group(4) @binding(14)
var normalTexture14: texture_2d<f32>;
@group(4) @binding(15)
var normalTexture15: texture_2d<f32>;
@group(4) @binding(16)
var normalTexture16: texture_2d<f32>;
@group(4) @binding(17)
var normalTexture17: texture_2d<f32>;
@group(4) @binding(18)
var normalTexture18: texture_2d<f32>;
@group(4) @binding(19)
var normalTexture19: texture_2d<f32>;
@group(4) @binding(20)
var normalTexture20: texture_2d<f32>;
@group(4) @binding(21)
var normalTexture21: texture_2d<f32>;
@group(4) @binding(22)
var normalTexture22: texture_2d<f32>;
@group(4) @binding(23)
var normalTexture23: texture_2d<f32>;
@group(4) @binding(24)
var normalTexture24: texture_2d<f32>;
@group(4) @binding(25)
var normalTexture25: texture_2d<f32>;
@group(4) @binding(26)
var normalTexture26: texture_2d<f32>;
@group(4) @binding(27)
var normalTexture27: texture_2d<f32>;
@group(4) @binding(28)
var normalTexture28: texture_2d<f32>;
@group(4) @binding(29)
var normalTexture29: texture_2d<f32>;
@group(4) @binding(30)
var normalTexture30: texture_2d<f32>;
@group(4) @binding(31)
var normalTexture31: texture_2d<f32>;
@group(4) @binding(32)
var normalTexture32: texture_2d<f32>;
@group(4) @binding(33)
var normalTexture33: texture_2d<f32>;
@group(4) @binding(34)
var normalTexture34: texture_2d<f32>;
@group(4) @binding(35)
var normalTexture35: texture_2d<f32>;
@group(4) @binding(36)
var normalTexture36: texture_2d<f32>;
@group(4) @binding(37)
var normalTexture37: texture_2d<f32>;
@group(4) @binding(38)
var normalTexture38: texture_2d<f32>;
@group(4) @binding(39)
var normalTexture39: texture_2d<f32>;
@group(4) @binding(40)
var normalTexture40: texture_2d<f32>;
@group(4) @binding(41)
var normalTexture41: texture_2d<f32>;
@group(4) @binding(42)
var normalTexture42: texture_2d<f32>;
@group(4) @binding(43)
var normalTexture43: texture_2d<f32>;
@group(4) @binding(44)
var normalTexture44: texture_2d<f32>;
@group(4) @binding(45)
var normalTexture45: texture_2d<f32>;
@group(4) @binding(46)
var normalTexture46: texture_2d<f32>;
@group(4) @binding(47)
var normalTexture47: texture_2d<f32>;
@group(4) @binding(48)
var normalTexture48: texture_2d<f32>;
@group(4) @binding(49)
var normalTexture49: texture_2d<f32>;
@group(4) @binding(50)
var normalTexture50: texture_2d<f32>;
@group(4) @binding(51)
var normalTexture51: texture_2d<f32>;
@group(4) @binding(52)
var normalTexture52: texture_2d<f32>;
@group(4) @binding(53)
var normalTexture53: texture_2d<f32>;
@group(4) @binding(54)
var normalTexture54: texture_2d<f32>;
@group(4) @binding(55)
var normalTexture55: texture_2d<f32>;
@group(4) @binding(56)
var normalTexture56: texture_2d<f32>;
@group(4) @binding(57)
var normalTexture57: texture_2d<f32>;
@group(4) @binding(58)
var normalTexture58: texture_2d<f32>;
@group(4) @binding(59)
var normalTexture59: texture_2d<f32>;
@group(4) @binding(60)
var normalTexture60: texture_2d<f32>;
@group(4) @binding(61)
var normalTexture61: texture_2d<f32>;
@group(4) @binding(62)
var normalTexture62: texture_2d<f32>;
@group(4) @binding(63)
var normalTexture63: texture_2d<f32>;
@group(4) @binding(64)
var normalTexture64: texture_2d<f32>;
@group(4) @binding(65)
var normalTexture65: texture_2d<f32>;
@group(4) @binding(66)
var normalTexture66: texture_2d<f32>;
@group(4) @binding(67)
var normalTexture67: texture_2d<f32>;
@group(4) @binding(68)
var normalTexture68: texture_2d<f32>;
@group(4) @binding(69)
var normalTexture69: texture_2d<f32>;
@group(4) @binding(70)
var normalTexture70: texture_2d<f32>;
@group(4) @binding(71)
var normalTexture71: texture_2d<f32>;
@group(4) @binding(72)
var normalTexture72: texture_2d<f32>;
@group(4) @binding(73)
var normalTexture73: texture_2d<f32>;
@group(4) @binding(74)
var normalTexture74: texture_2d<f32>;
@group(4) @binding(75)
var normalTexture75: texture_2d<f32>;
@group(4) @binding(76)
var normalTexture76: texture_2d<f32>;
@group(4) @binding(77)
var normalTexture77: texture_2d<f32>;
@group(4) @binding(78)
var normalTexture78: texture_2d<f32>;
@group(4) @binding(79)
var normalTexture79: texture_2d<f32>;
@group(4) @binding(80)
var normalTexture80: texture_2d<f32>;
@group(4) @binding(81)
var normalTexture81: texture_2d<f32>;
@group(4) @binding(82)
var normalTexture82: texture_2d<f32>;
@group(4) @binding(83)
var normalTexture83: texture_2d<f32>;
@group(4) @binding(84)
var normalTexture84: texture_2d<f32>;
@group(4) @binding(85)
var normalTexture85: texture_2d<f32>;
@group(4) @binding(86)
var normalTexture86: texture_2d<f32>;
@group(4) @binding(87)
var normalTexture87: texture_2d<f32>;
@group(4) @binding(88)
var normalTexture88: texture_2d<f32>;
@group(4) @binding(89)
var normalTexture89: texture_2d<f32>;
@group(4) @binding(90)
var normalTexture90: texture_2d<f32>;
@group(4) @binding(91)
var normalTexture91: texture_2d<f32>;
@group(4) @binding(92)
var normalTexture92: texture_2d<f32>;
@group(4) @binding(93)
var normalTexture93: texture_2d<f32>;
@group(4) @binding(94)
var normalTexture94: texture_2d<f32>;
@group(4) @binding(95)
var normalTexture95: texture_2d<f32>;
@group(4) @binding(96)
var normalTexture96: texture_2d<f32>;
@group(4) @binding(97)
var normalTexture97: texture_2d<f32>;
@group(4) @binding(98)
var normalTexture98: texture_2d<f32>;
@group(4) @binding(99)
var normalTexture99: texture_2d<f32>;
@group(5) @binding(0)
var normalTexture100: texture_2d<f32>;
@group(5) @binding(1)
var normalTexture101: texture_2d<f32>;
@group(5) @binding(2)
var normalTexture102: texture_2d<f32>;
@group(5) @binding(3)
var normalTexture103: texture_2d<f32>;
@group(5) @binding(4)
var normalTexture104: texture_2d<f32>;
@group(5) @binding(5)
var normalTexture105: texture_2d<f32>;
@group(5) @binding(6)
var normalTexture106: texture_2d<f32>;
@group(5) @binding(7)
var normalTexture107: texture_2d<f32>;
@group(5) @binding(8)
var normalTexture108: texture_2d<f32>;
@group(5) @binding(9)
var normalTexture109: texture_2d<f32>;
@group(5) @binding(10)
var normalTexture110: texture_2d<f32>;
@group(5) @binding(11)
var normalTexture111: texture_2d<f32>;
@group(5) @binding(12)
var normalTexture112: texture_2d<f32>;
@group(5) @binding(13)
var normalTexture113: texture_2d<f32>;
@group(5) @binding(14)
var normalTexture114: texture_2d<f32>;
@group(5) @binding(15)
var normalTexture115: texture_2d<f32>;
@group(5) @binding(16)
var normalTexture116: texture_2d<f32>;
@group(5) @binding(17)
var normalTexture117: texture_2d<f32>;
@group(5) @binding(18)
var normalTexture118: texture_2d<f32>;
@group(5) @binding(19)
var normalTexture119: texture_2d<f32>;
@group(5) @binding(20)
var normalTexture120: texture_2d<f32>;
@group(5) @binding(21)
var normalTexture121: texture_2d<f32>;
@group(5) @binding(22)
var normalTexture122: texture_2d<f32>;
@group(5) @binding(23)
var normalTexture123: texture_2d<f32>;
@group(5) @binding(24)
var normalTexture124: texture_2d<f32>;
@group(5) @binding(25)
var normalTexture125: texture_2d<f32>;
@group(5) @binding(26)
var normalTexture126: texture_2d<f32>;
@group(5) @binding(27)
var normalTexture127: texture_2d<f32>;
@group(5) @binding(28)
var normalTexture128: texture_2d<f32>;
@group(5) @binding(29)
var normalTexture129: texture_2d<f32>;
@group(5) @binding(30)
var normalTexture130: texture_2d<f32>;
@group(5) @binding(31)
var normalTexture131: texture_2d<f32>;

/////
fn sampleNormalTexture0(
    iTextureID: u32,
    uv: vec2<f32>) -> vec4<f32>
{
    var ret: vec4<f32> = vec4<f32>(1.0f, 0.0f, 0.0f, 1.0f);
        if(iTextureID == 0u)
    {
        ret = textureSample(normalTexture0, textureSampler, uv);
    }
    else if(iTextureID == 1u)
    {
        ret = textureSample(normalTexture1, textureSampler, uv);
    }
    else if(iTextureID == 2u)
    {
        ret = textureSample(normalTexture2, textureSampler, uv);
    }
    else if(iTextureID == 3u)
    {
        ret = textureSample(normalTexture3, textureSampler, uv);
    }
    else if(iTextureID == 4u)
    {
        ret = textureSample(normalTexture4, textureSampler, uv);
    }
    else if(iTextureID == 5u)
    {
        ret = textureSample(normalTexture5, textureSampler, uv);
    }
    else if(iTextureID == 6u)
    {
        ret = textureSample(normalTexture6, textureSampler, uv);
    }
    else if(iTextureID == 7u)
    {
        ret = textureSample(normalTexture7, textureSampler, uv);
    }
    else if(iTextureID == 8u)
    {
        ret = textureSample(normalTexture8, textureSampler, uv);
    }
    else if(iTextureID == 9u)
    {
        ret = textureSample(normalTexture9, textureSampler, uv);
    }
    else if(iTextureID == 10u)
    {
        ret = textureSample(normalTexture10, textureSampler, uv);
    }
    else if(iTextureID == 11u)
    {
        ret = textureSample(normalTexture11, textureSampler, uv);
    }
    else if(iTextureID == 12u)
    {
        ret = textureSample(normalTexture12, textureSampler, uv);
    }
    else if(iTextureID == 13u)
    {
        ret = textureSample(normalTexture13, textureSampler, uv);
    }
    else if(iTextureID == 14u)
    {
        ret = textureSample(normalTexture14, textureSampler, uv);
    }
    else if(iTextureID == 15u)
    {
        ret = textureSample(normalTexture15, textureSampler, uv);
    }
    else if(iTextureID == 16u)
    {
        ret = textureSample(normalTexture16, textureSampler, uv);
    }
    else if(iTextureID == 17u)
    {
        ret = textureSample(normalTexture17, textureSampler, uv);
    }
    else if(iTextureID == 18u)
    {
        ret = textureSample(normalTexture18, textureSampler, uv);
    }
    else if(iTextureID == 19u)
    {
        ret = textureSample(normalTexture19, textureSampler, uv);
    }
    else if(iTextureID == 20u)
    {
        ret = textureSample(normalTexture20, textureSampler, uv);
    }
    else if(iTextureID == 21u)
    {
        ret = textureSample(normalTexture21, textureSampler, uv);
    }
    else if(iTextureID == 22u)
    {
        ret = textureSample(normalTexture22, textureSampler, uv);
    }
    else if(iTextureID == 23u)
    {
        ret = textureSample(normalTexture23, textureSampler, uv);
    }
    else if(iTextureID == 24u)
    {
        ret = textureSample(normalTexture24, textureSampler, uv);
    }
    else if(iTextureID == 25u)
    {
        ret = textureSample(normalTexture25, textureSampler, uv);
    }
    else if(iTextureID == 26u)
    {
        ret = textureSample(normalTexture26, textureSampler, uv);
    }
    else if(iTextureID == 27u)
    {
        ret = textureSample(normalTexture27, textureSampler, uv);
    }
    else if(iTextureID == 28u)
    {
        ret = textureSample(normalTexture28, textureSampler, uv);
    }
    else if(iTextureID == 29u)
    {
        ret = textureSample(normalTexture29, textureSampler, uv);
    }
    else if(iTextureID == 30u)
    {
        ret = textureSample(normalTexture30, textureSampler, uv);
    }
    else if(iTextureID == 31u)
    {
        ret = textureSample(normalTexture31, textureSampler, uv);
    }
    else if(iTextureID == 32u)
    {
        ret = textureSample(normalTexture32, textureSampler, uv);
    }
    else if(iTextureID == 33u)
    {
        ret = textureSample(normalTexture33, textureSampler, uv);
    }
    else if(iTextureID == 34u)
    {
        ret = textureSample(normalTexture34, textureSampler, uv);
    }
    else if(iTextureID == 35u)
    {
        ret = textureSample(normalTexture35, textureSampler, uv);
    }
    else if(iTextureID == 36u)
    {
        ret = textureSample(normalTexture36, textureSampler, uv);
    }
    else if(iTextureID == 37u)
    {
        ret = textureSample(normalTexture37, textureSampler, uv);
    }
    else if(iTextureID == 38u)
    {
        ret = textureSample(normalTexture38, textureSampler, uv);
    }
    else if(iTextureID == 39u)
    {
        ret = textureSample(normalTexture39, textureSampler, uv);
    }
    else if(iTextureID == 40u)
    {
        ret = textureSample(normalTexture40, textureSampler, uv);
    }
    else if(iTextureID == 41u)
    {
        ret = textureSample(normalTexture41, textureSampler, uv);
    }
    else if(iTextureID == 42u)
    {
        ret = textureSample(normalTexture42, textureSampler, uv);
    }
    else if(iTextureID == 43u)
    {
        ret = textureSample(normalTexture43, textureSampler, uv);
    }
    else if(iTextureID == 44u)
    {
        ret = textureSample(normalTexture44, textureSampler, uv);
    }
    else if(iTextureID == 45u)
    {
        ret = textureSample(normalTexture45, textureSampler, uv);
    }
    else if(iTextureID == 46u)
    {
        ret = textureSample(normalTexture46, textureSampler, uv);
    }
    else if(iTextureID == 47u)
    {
        ret = textureSample(normalTexture47, textureSampler, uv);
    }
    else if(iTextureID == 48u)
    {
        ret = textureSample(normalTexture48, textureSampler, uv);
    }
    else if(iTextureID == 49u)
    {
        ret = textureSample(normalTexture49, textureSampler, uv);
    }
    else if(iTextureID == 50u)
    {
        ret = textureSample(normalTexture50, textureSampler, uv);
    }
    else if(iTextureID == 51u)
    {
        ret = textureSample(normalTexture51, textureSampler, uv);
    }
    else if(iTextureID == 52u)
    {
        ret = textureSample(normalTexture52, textureSampler, uv);
    }
    else if(iTextureID == 53u)
    {
        ret = textureSample(normalTexture53, textureSampler, uv);
    }
    else if(iTextureID == 54u)
    {
        ret = textureSample(normalTexture54, textureSampler, uv);
    }
    else if(iTextureID == 55u)
    {
        ret = textureSample(normalTexture55, textureSampler, uv);
    }
    else if(iTextureID == 56u)
    {
        ret = textureSample(normalTexture56, textureSampler, uv);
    }
    else if(iTextureID == 57u)
    {
        ret = textureSample(normalTexture57, textureSampler, uv);
    }
    else if(iTextureID == 58u)
    {
        ret = textureSample(normalTexture58, textureSampler, uv);
    }
    else if(iTextureID == 59u)
    {
        ret = textureSample(normalTexture59, textureSampler, uv);
    }
    else if(iTextureID == 60u)
    {
        ret = textureSample(normalTexture60, textureSampler, uv);
    }
    else if(iTextureID == 61u)
    {
        ret = textureSample(normalTexture61, textureSampler, uv);
    }
    else if(iTextureID == 62u)
    {
        ret = textureSample(normalTexture62, textureSampler, uv);
    }
    else if(iTextureID == 63u)
    {
        ret = textureSample(normalTexture63, textureSampler, uv);
    }
    else if(iTextureID == 64u)
    {
        ret = textureSample(normalTexture64, textureSampler, uv);
    }
    else if(iTextureID == 65u)
    {
        ret = textureSample(normalTexture65, textureSampler, uv);
    }
    else if(iTextureID == 66u)
    {
        ret = textureSample(normalTexture66, textureSampler, uv);
    }
    else if(iTextureID == 67u)
    {
        ret = textureSample(normalTexture67, textureSampler, uv);
    }
    else if(iTextureID == 68u)
    {
        ret = textureSample(normalTexture68, textureSampler, uv);
    }
    else if(iTextureID == 69u)
    {
        ret = textureSample(normalTexture69, textureSampler, uv);
    }
    else if(iTextureID == 70u)
    {
        ret = textureSample(normalTexture70, textureSampler, uv);
    }
    else if(iTextureID == 71u)
    {
        ret = textureSample(normalTexture71, textureSampler, uv);
    }
    else if(iTextureID == 72u)
    {
        ret = textureSample(normalTexture72, textureSampler, uv);
    }
    else if(iTextureID == 73u)
    {
        ret = textureSample(normalTexture73, textureSampler, uv);
    }
    else if(iTextureID == 74u)
    {
        ret = textureSample(normalTexture74, textureSampler, uv);
    }
    else if(iTextureID == 75u)
    {
        ret = textureSample(normalTexture75, textureSampler, uv);
    }
    else if(iTextureID == 76u)
    {
        ret = textureSample(normalTexture76, textureSampler, uv);
    }
    else if(iTextureID == 77u)
    {
        ret = textureSample(normalTexture77, textureSampler, uv);
    }
    else if(iTextureID == 78u)
    {
        ret = textureSample(normalTexture78, textureSampler, uv);
    }
    else if(iTextureID == 79u)
    {
        ret = textureSample(normalTexture79, textureSampler, uv);
    }
    else if(iTextureID == 80u)
    {
        ret = textureSample(normalTexture80, textureSampler, uv);
    }
    else if(iTextureID == 81u)
    {
        ret = textureSample(normalTexture81, textureSampler, uv);
    }
    else if(iTextureID == 82u)
    {
        ret = textureSample(normalTexture82, textureSampler, uv);
    }
    else if(iTextureID == 83u)
    {
        ret = textureSample(normalTexture83, textureSampler, uv);
    }
    else if(iTextureID == 84u)
    {
        ret = textureSample(normalTexture84, textureSampler, uv);
    }
    else if(iTextureID == 85u)
    {
        ret = textureSample(normalTexture85, textureSampler, uv);
    }
    else if(iTextureID == 86u)
    {
        ret = textureSample(normalTexture86, textureSampler, uv);
    }
    else if(iTextureID == 87u)
    {
        ret = textureSample(normalTexture87, textureSampler, uv);
    }
    else if(iTextureID == 88u)
    {
        ret = textureSample(normalTexture88, textureSampler, uv);
    }
    else if(iTextureID == 89u)
    {
        ret = textureSample(normalTexture89, textureSampler, uv);
    }
    else if(iTextureID == 90u)
    {
        ret = textureSample(normalTexture90, textureSampler, uv);
    }
    else if(iTextureID == 91u)
    {
        ret = textureSample(normalTexture91, textureSampler, uv);
    }
    else if(iTextureID == 92u)
    {
        ret = textureSample(normalTexture92, textureSampler, uv);
    }
    else if(iTextureID == 93u)
    {
        ret = textureSample(normalTexture93, textureSampler, uv);
    }
    else if(iTextureID == 94u)
    {
        ret = textureSample(normalTexture94, textureSampler, uv);
    }
    else if(iTextureID == 95u)
    {
        ret = textureSample(normalTexture95, textureSampler, uv);
    }
    else if(iTextureID == 96u)
    {
        ret = textureSample(normalTexture96, textureSampler, uv);
    }
    else if(iTextureID == 97u)
    {
        ret = textureSample(normalTexture97, textureSampler, uv);
    }
    else if(iTextureID == 98u)
    {
        ret = textureSample(normalTexture98, textureSampler, uv);
    }
    else if(iTextureID == 99u)
    {
        ret = textureSample(normalTexture99, textureSampler, uv);
    }

    return ret;
}

/////
fn sampleNormalTexture1(
    iTextureID: u32,
    uv: vec2<f32>) -> vec4<f32>
{
    var ret: vec4<f32> = vec4<f32>(1.0f, 0.0f, 0.0f, 1.0f);
    if(iTextureID == 100u)
    {
        ret = textureSample(normalTexture100, textureSampler, uv);
    }
    else if(iTextureID == 101u)
    {
        ret = textureSample(normalTexture101, textureSampler, uv);
    }
    else if(iTextureID == 102u)
    {
        ret = textureSample(normalTexture102, textureSampler, uv);
    }
    else if(iTextureID == 103u)
    {
        ret = textureSample(normalTexture103, textureSampler, uv);
    }
    else if(iTextureID == 104u)
    {
        ret = textureSample(normalTexture104, textureSampler, uv);
    }
    else if(iTextureID == 105u)
    {
        ret = textureSample(normalTexture105, textureSampler, uv);
    }
    else if(iTextureID == 106u)
    {
        ret = textureSample(normalTexture106, textureSampler, uv);
    }
    else if(iTextureID == 107u)
    {
        ret = textureSample(normalTexture107, textureSampler, uv);
    }
    else if(iTextureID == 108u)
    {
        ret = textureSample(normalTexture108, textureSampler, uv);
    }
    else if(iTextureID == 109u)
    {
        ret = textureSample(normalTexture109, textureSampler, uv);
    }
    else if(iTextureID == 110u)
    {
        ret = textureSample(normalTexture110, textureSampler, uv);
    }
    else if(iTextureID == 111u)
    {
        ret = textureSample(normalTexture111, textureSampler, uv);
    }
    else if(iTextureID == 112u)
    {
        ret = textureSample(normalTexture112, textureSampler, uv);
    }
    else if(iTextureID == 113u)
    {
        ret = textureSample(normalTexture113, textureSampler, uv);
    }
    else if(iTextureID == 114u)
    {
        ret = textureSample(normalTexture114, textureSampler, uv);
    }
    else if(iTextureID == 115u)
    {
        ret = textureSample(normalTexture115, textureSampler, uv);
    }
    else if(iTextureID == 116u)
    {
        ret = textureSample(normalTexture116, textureSampler, uv);
    }
    else if(iTextureID == 117u)
    {
        ret = textureSample(normalTexture117, textureSampler, uv);
    }
    else if(iTextureID == 118u)
    {
        ret = textureSample(normalTexture118, textureSampler, uv);
    }
    else if(iTextureID == 119u)
    {
        ret = textureSample(normalTexture119, textureSampler, uv);
    }
    else if(iTextureID == 120u)
    {
        ret = textureSample(normalTexture120, textureSampler, uv);
    }
    else if(iTextureID == 121u)
    {
        ret = textureSample(normalTexture121, textureSampler, uv);
    }
    else if(iTextureID == 122u)
    {
        ret = textureSample(normalTexture122, textureSampler, uv);
    }
    else if(iTextureID == 123u)
    {
        ret = textureSample(normalTexture123, textureSampler, uv);
    }
    else if(iTextureID == 124u)
    {
        ret = textureSample(normalTexture124, textureSampler, uv);
    }
    else if(iTextureID == 125u)
    {
        ret = textureSample(normalTexture125, textureSampler, uv);
    }
    else if(iTextureID == 126u)
    {
        ret = textureSample(normalTexture126, textureSampler, uv);
    }
    else if(iTextureID == 127u)
    {
        ret = textureSample(normalTexture127, textureSampler, uv);
    }
    else if(iTextureID == 128u)
    {
        ret = textureSample(normalTexture128, textureSampler, uv);
    }
    else if(iTextureID == 129u)
    {
        ret = textureSample(normalTexture129, textureSampler, uv);
    }
    else if(iTextureID == 130u)
    {
        ret = textureSample(normalTexture130, textureSampler, uv);
    }
    else if(iTextureID == 131u)
    {
        ret = textureSample(normalTexture131, textureSampler, uv);
    }

    return ret;
}

