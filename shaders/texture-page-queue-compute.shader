const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const PROBE_IMAGE_SIZE: u32 = 8u;
const TEXTURE_PAGE_SIZE: u32 = 64u;

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
    miMIP: u32,
    miUpdateFrame: u32,
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

@group(0) @binding(0)
var<storage, read_write> aTexturePageQueueMIP: array<TexturePage>;

@group(0) @binding(1)
var<storage, read_write> aiCounters: array<atomic<i32>>;

@group(0) @binding(2)
var<storage, read_write> aiUsedHashPages: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read_write> aPageHashTableMIP: array<HashEntry>;

@group(0) @binding(4)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(5)
var texCoordTexture: texture_2d<f32>;

@group(0) @binding(6)
var textureAtlas0: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(7)
var textureAtlas1: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(8)
var textureAtlas2: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(9)
var textureAtlas3: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(10)
var initialTextureAtlas: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(11)
var texturePageInfoTexture: texture_storage_2d<rgba32float, write>;

@group(0) @binding(12)
var normalTexturePageInfoTexture: texture_storage_2d<rgba32float, write>;

@group(0) @binding(13)
var textureSampler: sampler;

@group(1) @binding(0)
var<storage, read> aTextureSizes: array<vec2<i32>>; 

@group(1) @binding(1)
var<storage> aMeshMaterialID: array<u32>;

@group(1) @binding(2)
var<storage> aMeshMaterials: array<Material>;

@group(1) @binding(3)
var<storage, read> aNormalTextureSizes: array<vec2<i32>>; 

@group(1) @binding(4)
var<uniform> defaultUniformData: DefaultUniformData;

const iNumThreadX: u32 = 16u;
const iNumThreadY: u32 = 16u;

/////
@compute
@workgroup_size(iNumThreadX, iNumThreadY)
fn cs_main(
    @builtin(workgroup_id) workGroupIndex: vec3<u32>,
    @builtin(local_invocation_id) localIndex: vec3<u32>,
    @builtin(num_workgroups) workGroupSize: vec3<u32>) 
{
    let fFar: f32 = 100.0f;
    let fNear: f32 = 1.0f;
    let iMaxNumCachePerFrame: i32 = 200;

    let iThreadID: u32 = u32(localIndex.y) + iNumThreadX + u32(localIndex.x);
    
    // total threads with workgroup and local workgroup
    let iNumTotalThreadX: u32 = iNumThreadX * workGroupSize.x;
    let iNumTotalThreadY: u32 = iNumThreadY * workGroupSize.y; 

    let iTileWidth: u32 = u32(defaultUniformData.miScreenWidth) / iNumTotalThreadX;
    let iTileHeight: u32 = u32(defaultUniformData.miScreenHeight) / iNumTotalThreadY;

    let iTotalTileX: u32 = workGroupIndex.x * iNumThreadX + localIndex.x;
    let iTotalTileY: u32 = workGroupIndex.y * iNumThreadY + localIndex.y;

    for(var iTileY: u32 = 0; iTileY < iTileHeight; iTileY++)
    {
        var fY: f32 = f32(iTotalTileY * iTileHeight + iTileY) / f32(defaultUniformData.miScreenHeight);
        for(var iTileX: u32 = 0; iTileX < iTileWidth; iTileX++)
        {
            var fX: f32 = f32(iTotalTileX * iTileWidth + iTileX) / f32(defaultUniformData.miScreenWidth);
            var uv: vec2<f32> = vec2<f32>(fX, fY);
            
            let worldPosition: vec4<f32> = textureSampleLevel(
                worldPositionTexture,
                textureSampler,
                uv,
                0.0f
            );
            
            var texCoord: vec4<f32> = textureSampleLevel(
                texCoordTexture,
                textureSampler,
                uv,
                0.0f
            );

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

            // TODO: use depth for MIP level
            var fDepth: f32 = fract(worldPosition.w);

            // linearize depth
            let fLinearDepth: f32 = (fNear * fFar) / (fFar - fDepth * (fFar - fNear));
            var iMIP: u32 = 0u;
            if(fLinearDepth >= 3.0f && fLinearDepth < 10.0f)
            {
                iMIP = 1u;
            }
            else if(fLinearDepth >= 10.0f && fLinearDepth < 40.0f)
            {
                iMIP = 2u;
            }
            else if(fLinearDepth >= 40.0f)
            {
                iMIP = 3u;
            }

            // image coordinate to fetch
            let iMesh: u32 = u32(ceil(worldPosition.w - fDepth - 0.5f));
            let iMaterialID: u32 = aMeshMaterialID[iMesh] - 1u;
            let iAlbedoTextureID: u32 = aMeshMaterials[iMaterialID].miAlbedoTextureID;
            let iNormalTextureID: u32 = aMeshMaterials[iMaterialID].miNormalTextureID;

            var textureSize: vec2<f32> = vec2<f32>(
                f32(aTextureSizes[iMaterialID].x),
                f32(aTextureSizes[iMaterialID].y)
            );
            var normalTextureSize: vec2<f32> = vec2<f32>(
                f32(aNormalTextureSizes[iMaterialID].x),
                f32(aNormalTextureSizes[iMaterialID].y)
            );

            // mip dimension, division by power of 2
            let mipTextureSize = textureSize / pow(2.0f, f32(iMIP));
            let mipNormalTextureSize = normalTextureSize / pow(2.0f, f32(iMIP));    

            // get the start uv in image space for the page
            let mipImageCoord: vec2<f32> = texCoord.xy * mipTextureSize;
            let mipStartPageUV: vec2<i32> = vec2<i32>(
                i32(floor(mipImageCoord.x / f32(TEXTURE_PAGE_SIZE))),
                i32(floor(mipImageCoord.y / f32(TEXTURE_PAGE_SIZE)))
            );
            let mipNormalImageCoord: vec2<f32> = texCoord.xy * mipNormalTextureSize;
            let mipNormalStartPageUV: vec2<i32> = vec2<i32>(
                i32(floor(mipNormalImageCoord.x / f32(TEXTURE_PAGE_SIZE))),
                i32(floor(mipNormalImageCoord.y / f32(TEXTURE_PAGE_SIZE)))
            );
            
            // hash index to check for existence
            var iAlbedoHashIndex: u32 = hash13(
                vec3<f32>(
                    f32(mipStartPageUV.x),
                    f32(mipStartPageUV.y) * 32.0f, 
                    f32(iAlbedoTextureID) + f32(iMIP + 1) * 1024.0f 
                ),
                80000u
            );

            var iNormalHashIndex: u32 = hash13(
                vec3<f32>(
                    f32(mipNormalStartPageUV.x),
                    f32(mipNormalStartPageUV.y) * 32.0f, 
                    f32(iNormalTextureID + 65536) + f32(iMIP + 1) * 1024.0f 
                ),
                80000u
            );

            iAlbedoHashIndex = registerTexturePage(
                mipStartPageUV,
                iAlbedoHashIndex,
                iAlbedoTextureID,
                iMIP
            );

            iNormalHashIndex = registerTexturePage(
                mipNormalStartPageUV,
                iNormalHashIndex,
                iNormalTextureID + 65536,
                iMIP
            );

            // output page info
            textureStore(
                texturePageInfoTexture,
                vec2<i32>(i32(uv.x * f32(defaultUniformData.miScreenWidth)), i32(uv.y * f32(defaultUniformData.miScreenHeight))),
                vec4<f32>(
                    f32(mipStartPageUV.x),
                    f32(mipStartPageUV.y),
                    f32(iAlbedoTextureID) + f32(iMIP) * 0.1f,
                    f32(iAlbedoHashIndex)
                )
            );

            // output page info
            textureStore(
                normalTexturePageInfoTexture,
                vec2<i32>(i32(uv.x * f32(defaultUniformData.miScreenWidth)), i32(uv.y * f32(defaultUniformData.miScreenHeight))),
                vec4<f32>(
                    f32(mipNormalStartPageUV.x),
                    f32(mipNormalStartPageUV.y),
                    f32(iNormalTextureID) + f32(iMIP) * 0.1f,
                    f32(iNormalHashIndex)
                )
            );
        }
    }
}

/////
fn registerTexturePage(
    startPageUV: vec2<i32>,
    iHashIndex: u32,
    iTextureID: u32,
    iMIP: u32) -> u32
{
    var iHashIndexCopy: u32 = iHashIndex;

    let iBefore: u32 = atomicExchange(&aiUsedHashPages[iHashIndexCopy], 1u);
    if(iBefore <= 0u)
    {
        let iQueueIndex: i32 = atomicAdd(&aiCounters[0], 1); 
        atomicAdd(&aiCounters[iMIP + 1], 1);
        
        let iSignedEncodedCoordinateMIP: i32 = i32(i32(startPageUV.x) | i32(i32(startPageUV.y) << 16));
        aTexturePageQueueMIP[iQueueIndex].miPageUV = iSignedEncodedCoordinateMIP;
        aTexturePageQueueMIP[iQueueIndex].miTextureID = i32(iTextureID);
        aTexturePageQueueMIP[iQueueIndex].miHashIndex = i32(iHashIndexCopy);
        aTexturePageQueueMIP[iQueueIndex].miMIP = i32(iMIP);

        aPageHashTableMIP[iHashIndexCopy].miPageCoordinate = u32(u32(startPageUV.x) | u32(u32(startPageUV.y) << 16u));
        aPageHashTableMIP[iHashIndexCopy].miPageIndex = UINT32_MAX;
        aPageHashTableMIP[iHashIndexCopy].miMIP = iMIP;
    }
    else 
    {
        let iMIPEncodedCoordinate: u32 = u32(u32(startPageUV.x) | u32(u32(startPageUV.y) << 16u));
        if(aPageHashTableMIP[iHashIndexCopy].miPageCoordinate != iMIPEncodedCoordinate)
        {
            for(var i: u32 = 0u; i < 10000u; i++)
            {
                iHashIndexCopy = (iHashIndexCopy + 1u) % 80000u;
                
                // found an existing entry 
                if(aPageHashTableMIP[iHashIndexCopy].miPageCoordinate == iMIPEncodedCoordinate)
                {
                    break;
                }

                // register new entry if not found
                let iBefore: u32 = atomicExchange(&aiUsedHashPages[iHashIndexCopy], 1u);
                if(iBefore <= 0u)
                {
                    let iQueueIndex: i32 = atomicAdd(&aiCounters[0], 1); 
                    atomicAdd(&aiCounters[iMIP + 1], 1);

                    let iSignedEncodedCoordinateMIP: i32 = i32(i32(startPageUV.x) | i32(i32(startPageUV.y) << 16));
                    aTexturePageQueueMIP[iQueueIndex].miPageUV = i32(iSignedEncodedCoordinateMIP);
                    aTexturePageQueueMIP[iQueueIndex].miTextureID = i32(iTextureID);
                    aTexturePageQueueMIP[iQueueIndex].miHashIndex = i32(iHashIndexCopy);
                    aTexturePageQueueMIP[iQueueIndex].miMIP = i32(iMIP);

                    aPageHashTableMIP[iHashIndexCopy].miPageCoordinate = iMIPEncodedCoordinate;
                    aPageHashTableMIP[iHashIndexCopy].miPageIndex = UINT32_MAX;
                    aPageHashTableMIP[iHashIndexCopy].miMIP = iMIP;

                    break;
                }
            }
        }
    }

    return iHashIndexCopy;
}


/////
fn murmurHash13(
    src: vec3<u32>) -> u32
{
    var srcCopy: vec3<u32> = src;
    var M: u32 = 0x5bd1e995u;
    var h: u32 = 1190494759u;
    srcCopy *= M; srcCopy.x ^= srcCopy.x >> 24u; srcCopy.y ^= srcCopy.y >> 24u; srcCopy.z ^= srcCopy.z >> 24u; srcCopy *= M;
    h *= M; h ^= srcCopy.x; h *= M; h ^= srcCopy.y; h *= M; h ^= srcCopy.z;
    h ^= h >> 13u; h *= M; h ^= h >> 15u;
    return h;
}

/////
fn hash13(
    src: vec3<f32>,
    iNumSlots: u32) -> u32
{
    let srcU32: vec3<u32> = vec3<u32>(
        bitcast<u32>(src.x),
        bitcast<u32>(src.y),
        bitcast<u32>(src.z)
    );

    let h: u32 = u32(murmurHash13(srcU32));
    var fValue: f32 = bitcast<f32>((h & 0x007ffffffu) | 0x3f800000u) - 1.0f;
    let iRet: u32 = clamp(u32(fValue * f32(iNumSlots - 1)), 0u, iNumSlots - 1);
    return iRet;
}
