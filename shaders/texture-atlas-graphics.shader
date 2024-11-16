const UINT32_MAX: u32 = 0xffffffffu;
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

struct TexturePage
{
    //mPageUV: vec2<i32>,
    miPageUV: i32,
    miTextureID: i32,
    miHashIndex: i32,
    
    miMIP: i32,
};

struct HashEntry
{
    miPageCoordinate: u32,
    miPageIndex: u32,
    miTextureIDAndMIP: u32,
    miUpdateFrame: u32,
};

struct TextureAtlasOutput
{
    mColor: vec4<f32>,
    mbLoaded: bool
};

@group(0) @binding(0)
var atlasTexture0: texture_2d<f32>;

@group(0) @binding(1)
var atlasTexture1: texture_2d<f32>;

@group(0) @binding(2)
var atlasTexture2: texture_2d<f32>;

@group(0) @binding(3)
var atlasTexture3: texture_2d<f32>;

@group(0) @binding(4)
var pageInfoTextureMIP: texture_2d<f32>;

@group(0) @binding(5)
var pageInfoNormalTextureMIP: texture_2d<f32>;

@group(0) @binding(6)
var texCoordTexture: texture_2d<f32>;

@group(0) @binding(7)
var<storage, read_write> aPageHashTableMIP: array<HashEntry>;

@group(0) @binding(8)
var initialTextureAtlas: texture_2d<f32>;

@group(0) @binding(9)
var textureSampler: sampler;

@group(0) @binding(10)
var linearTextureSampler: sampler;

@group(1) @binding(0)
var<storage, read> aTextureDimension: array<vec2<i32>>;

@group(1) @binding(1)
var<storage, read> aNormalTextureDimension: array<vec2<i32>>;

@group(1) @binding(2)
var<uniform> defaultUniformData: DefaultUniformData;

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
    @location(0) mAlbedo : vec4<f32>,
    @location(1) mDebug0: vec4<f32>,
    @location(2) mDebug1: vec4<f32>,
    @location(3) mDebug2: vec4<f32>,
    @location(4) mDebug3: vec4<f32>,
    @location(5) mDebug4: vec4<f32>,
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
    
    out.mAlbedo = vec4<f32>(1.0f, 0.0f, 0.0f, 1.0f);
    out.mDebug0 = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    out.mDebug1 = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    out.mDebug2 = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    out.mDebug3 = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    out.mDebug4 = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

    out.mAlbedo = outputTextureMIP(
        in.texCoord, 
        0u
    ).mColor;

    out.mDebug3 = out.mAlbedo;

    let retNormal: TextureAtlasOutput  = outputTextureMIP(
        in.texCoord, 
        1u
    );

    out.mDebug0 = retNormal.mColor;
    
    return out;
}

/////
fn outputTextureMIP(
    texCoord: vec2<f32>, 
    iTextureType: u32) -> TextureAtlasOutput
{
    var ret: TextureAtlasOutput;

    let fPageSize: f32 = 64.0f;
    let fTextureAtlasSize: f32 = 8192.0f;
    let fNumPagePerRow: f32 = fTextureAtlasSize / fPageSize;
    let iNumPagesPerRow: i32 = i32(fNumPagePerRow);
    let fOneOverNumPagePerRow = 1.0f / fNumPagePerRow;

    // texture page info 
    var pageInfoMIP: vec4<f32> = textureSample(
        pageInfoTextureMIP,
        textureSampler,
        texCoord
    );
    if(iTextureType == 1u)
    {
        pageInfoMIP = textureSample(
            pageInfoNormalTextureMIP,
            textureSampler,
            texCoord
        );
    }

    let uv: vec2<f32> = textureSample(
        texCoordTexture,
        textureSampler,
        texCoord
    ).xy;

    // texture id, hash id, mip and page uv
    let iTextureIDMIP: i32 = i32(ceil(pageInfoMIP.z - 0.5f));
    let iHashIndexMIP: i32 = i32(ceil(pageInfoMIP.w - 0.5f));
    var iPageIndexMIP: i32 = i32(aPageHashTableMIP[iHashIndexMIP].miPageIndex);
    let iMIP: u32 = aPageHashTableMIP[iHashIndexMIP].miTextureIDAndMIP & 0xf;

    if(u32(iPageIndexMIP) == 0xffffffffu)
    {
        let fInitialTextureAtlasSize: f32 = 512.0f;
        let iInitialPageSize: i32 = 8;
        let iNumInitialPagesPerRow: i32 = i32(fInitialTextureAtlasSize) / 8; 

        // get the uv difference from start of the page to current sample uv
        var initialTextureDimension: vec2<i32> = vec2<i32>(iInitialPageSize, iInitialPageSize);
        let initialImageCoord: vec2<i32> = vec2<i32>(
            i32(uv.x * f32(initialTextureDimension.x)),
            i32(uv.y * f32(initialTextureDimension.y))
        );
        let initialImagePageIndex: vec2<i32> = vec2<i32>(
            initialImageCoord.x / iInitialPageSize,
            initialImageCoord.y / iInitialPageSize
        );
        let initialStartPageImageCoord: vec2<i32> = vec2<i32>(
            initialImagePageIndex.x * iInitialPageSize,
            initialImagePageIndex.y * iInitialPageSize
        ); 
        let initialImageCoordDiff: vec2<i32> = initialImageCoord - initialStartPageImageCoord;
        
        // uv of the page in the atlas
        // page x, y
        let iInitialAtlasPageX: i32 = iTextureIDMIP % iNumInitialPagesPerRow;
        let iInitialAtlasPageY: i32 = iTextureIDMIP / iNumInitialPagesPerRow;
        
        // atlas uv
        let iInitialAtlasX: i32 = iInitialAtlasPageX * iInitialPageSize;
        let iInitialAtlasY: i32 = iInitialAtlasPageY * iInitialPageSize;
        let initialAtlasUV: vec2<f32> = vec2<f32>(
            f32(iInitialAtlasX + initialImageCoordDiff.x) / fInitialTextureAtlasSize,
            f32(iInitialAtlasY + initialImageCoordDiff.y) / fInitialTextureAtlasSize 
        );

        ret.mbLoaded = false;
        ret.mColor = textureSample(
            initialTextureAtlas,
            textureSampler,
            initialAtlasUV);

        if(iTextureType == 1u)
        {
            ret.mColor = vec4<f32>(0.5f, 0.5f, 1.0f, 1.0f);
        }

        return ret; // * vec4<f32>(1.0f, 0.2f, 0.2f, 1.0f);
    }

    ret.mColor = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    ret.mbLoaded = true;

    // atlas texture index
    iPageIndexMIP = iPageIndexMIP % (iNumPagesPerRow * iNumPagesPerRow);
    if(iPageIndexMIP > 0)
    {
        iPageIndexMIP -= 1;

        let iMIPDenom: i32 = i32(pow(2.0f, f32(iMIP)));

        // get the uv difference from start of the page to current sample uv
        var textureDimension: vec2<i32> = aTextureDimension[iTextureIDMIP];
        textureDimension[0] /= iMIPDenom;
        textureDimension[1] /= iMIPDenom;
        let imageCoord: vec2<i32> = vec2<i32>(
            i32(uv.x * f32(textureDimension.x)),
            i32(uv.y * f32(textureDimension.y))
        );
        let imagePageIndex: vec2<i32> = vec2<i32>(
            imageCoord.x / i32(fPageSize),
            imageCoord.y / i32(fPageSize)
        );
        let startPageImageCoord: vec2<i32> = vec2<i32>(
            imagePageIndex.x * i32(fPageSize),
            imagePageIndex.y * i32(fPageSize)
        ); 
        let imageCoordDiff: vec2<i32> = imageCoord - startPageImageCoord;
        
        // uv of the page in the atlas
        // page x, y
        let iAtlasPageX: i32 = iPageIndexMIP % iNumPagesPerRow;
        let iAtlasPageY: i32 = iPageIndexMIP / iNumPagesPerRow;
        
        // atlas uv
        let iAtlasX: i32 = iAtlasPageX * i32(fPageSize);
        let iAtlasY: i32 = iAtlasPageY * i32(fPageSize);
        let atlasUV: vec2<f32> = vec2<f32>(
            f32(iAtlasX + imageCoordDiff.x) / fTextureAtlasSize,
            f32(iAtlasY + imageCoordDiff.y) / fTextureAtlasSize 
        );
        let atlasUV0: vec2<f32> = vec2<f32>(
            f32(max(iAtlasX + imageCoordDiff.x - 1, iAtlasX)) / fTextureAtlasSize,
            f32(iAtlasY + imageCoordDiff.y) / fTextureAtlasSize 
        );
        let atlasUV1: vec2<f32> = vec2<f32>(
            f32(iAtlasX + imageCoordDiff.x) / fTextureAtlasSize,
            f32(max(iAtlasY + imageCoordDiff.y - 1, iAtlasY)) / fTextureAtlasSize 
        );
        let atlasUV2: vec2<f32> = vec2<f32>(
            f32(min(iAtlasX + imageCoordDiff.x + 1, iAtlasX + i32(fPageSize) - 1)) / fTextureAtlasSize,
            f32(iAtlasY + imageCoordDiff.y) / fTextureAtlasSize 
        );
        let atlasUV3: vec2<f32> = vec2<f32>(
            f32(iAtlasX + imageCoordDiff.x) / fTextureAtlasSize,
            f32(min(iAtlasY + imageCoordDiff.y + 1, iAtlasY + i32(fPageSize) - 1)) / fTextureAtlasSize 
        );

        // sample atlas texture based on MIP index
        ret.mColor = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        if(iMIP == 0)
        {
            ret.mColor = textureSample(
                atlasTexture0,
                textureSampler,
                atlasUV
            );

/*
            ret.mColor += textureSample(
                atlasTexture0,
                textureSampler,
                atlasUV0
            );

            ret.mColor += textureSample(
                atlasTexture0,
                textureSampler,
                atlasUV1
            );

            ret.mColor += textureSample(
                atlasTexture0,
                textureSampler,
                atlasUV2
            );

            ret.mColor += textureSample(
                atlasTexture0,
                textureSampler,
                atlasUV3
            );

            ret.mColor *= 0.25f;
*/
        }
        else if(iMIP == 1)
        {
            ret.mColor = textureSample(
                atlasTexture1,
                textureSampler,
                atlasUV
            );
/*
            ret.mColor += textureSample(
                atlasTexture1,
                textureSampler,
                atlasUV0
            );

            ret.mColor += textureSample(
                atlasTexture1,
                textureSampler,
                atlasUV1
            );

            ret.mColor += textureSample(
                atlasTexture1,
                textureSampler,
                atlasUV2
            );

            ret.mColor += textureSample(
                atlasTexture1,
                textureSampler,
                atlasUV3
            );

            ret.mColor *= 0.25f;
*/
        }
        else if(iMIP == 2)
        {
            ret.mColor = textureSample(
                atlasTexture2,
                textureSampler,
                atlasUV
            );

/*
            ret.mColor += textureSample(
                atlasTexture2,
                textureSampler,
                atlasUV0
            );

            ret.mColor += textureSample(
                atlasTexture2,
                textureSampler,
                atlasUV1
            );

            ret.mColor += textureSample(
                atlasTexture2,
                textureSampler,
                atlasUV2
            );

            ret.mColor += textureSample(
                atlasTexture2,
                textureSampler,
                atlasUV3
            );

            ret.mColor *= 0.25f;
*/
        }
        else if(iMIP == 3)
        {
            ret.mColor = textureSample(
                atlasTexture3,
                textureSampler,
                atlasUV
            );

/*
            ret.mColor += textureSample(
                atlasTexture3,
                textureSampler,
                atlasUV0
            );

            ret.mColor += textureSample(
                atlasTexture3,
                textureSampler,
                atlasUV1
            );

            ret.mColor += textureSample(
                atlasTexture3,
                textureSampler,
                atlasUV2
            );

            ret.mColor += textureSample(
                atlasTexture3,
                textureSampler,
                atlasUV3
            );

            ret.mColor *= 0.25f;
*/
        }

        // mark page as active
        aPageHashTableMIP[iHashIndexMIP].miUpdateFrame = u32(defaultUniformData.miFrame);
    }

    return ret;
}

