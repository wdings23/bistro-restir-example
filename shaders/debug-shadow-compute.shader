const kfShadowColor: f32 = 0.05f;

@group(0) @binding(0)
var viewShadowTexture: texture_storage_2d<rgba32float, write>;

@group(0) @binding(1)
var viewDepthTexture: texture_2d<f32>;

@group(0) @binding(2)
var lightViewDepthTexture0: texture_2d<f32>;

@group(0) @binding(3)
var lightViewDepthTexture1: texture_2d<f32>;

@group(0) @binding(4)
var lightViewDepthTexture2: texture_2d<f32>;

@group(0) @binding(5)
var lightViewDepthTexture3: texture_2d<f32>;

@group(0) @binding(6)
var lightViewDepthTexture4: texture_2d<f32>;

@group(0) @binding(7)
var lightViewDepthTexture5: texture_2d<f32>;

@group(0) @binding(8)
var lightViewDepthTexture6: texture_2d<f32>;

@group(0) @binding(9)
var lightViewDepthTexture7: texture_2d<f32>;

@group(0) @binding(10)
var lightViewDepthTexture8: texture_2d<f32>;

@group(0) @binding(11)
var lightViewDepthTexture9: texture_2d<f32>;

@group(0) @binding(12)
var lightViewDepthTexture10: texture_2d<f32>;

@group(0) @binding(13)
var lightViewDepthTexture11: texture_2d<f32>;

@group(0) @binding(14)
var <storage, read_write> aiVisibleTiles: array<atomic<u32>>;

@group(0) @binding(15)
var <storage, read_write> aiCounters: array<atomic<u32>>;

struct TilePageInfo
{
    miX: u32,
    miY: u32,
    miPartition: u32,
    miPadding: u32,
};

@group(0) @binding(16)
var <storage, read_write> aTilePageInfo: array<TilePageInfo>;

@group(0) @binding(17)
var closeUpLightViewTexture: texture_2d<f32>;

@group(0) @binding(18)
var <storage, read> aCloseUpViewMatrices: array<mat4x4<f32>>;

struct UniformData 
{
    mViewProjectionMatrix: mat4x4<f32>,  
    mInverseViewProjectionMatrix: mat4x4<f32>,

    maLightViewProjectionMatrices: array<mat4x4<f32>, 12>,

    miScreenWidth: u32,
    miScreenHeight: u32,
};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

const iLocalThreadSize: u32 = 16u;

var<workgroup> saTilePages: array<vec2<u32>, 256>;

@compute
@workgroup_size(iLocalThreadSize, iLocalThreadSize, 1)
fn cs_main(
    @builtin(workgroup_id) workGroupIndex: vec3<u32>, 
    @builtin(local_invocation_id) localIndex: vec3<u32>) 
{
    let iTotalWorkgroup: u32 = workGroupIndex.y * 40 + workGroupIndex.x;
    let iLocalThread: u32 = localIndex.y * iLocalThreadSize + localIndex.x;

    let iTotalThreadIndex: u32 = (iTotalWorkgroup * iLocalThreadSize * iLocalThreadSize) + iLocalThread;

    let iOutputTextureSize: u32 = 8192u;
    let iTileSize: u32 = 128u;
    let fOneOverNumTiles: f32 = f32(iTileSize) / f32(iOutputTextureSize);
    let iNumTilesPerRow: u32 = iOutputTextureSize / iTileSize;
    let iNumPartitions: u32 = 3u;
    let iNumTotalVisibleEntries: u32 = iNumPartitions * (iNumTilesPerRow * (iNumTilesPerRow / 32));

    let iSampleX: u32 = workGroupIndex.x * iLocalThreadSize + localIndex.x;
    let iSampleY: u32 = workGroupIndex.y * iLocalThreadSize + localIndex.y;

    var fViewDepth: f32 = textureLoad(
        viewDepthTexture,
        vec2<u32>(iSampleX, iSampleY),
        0
    ).x;
    //fViewDepth = fract(fViewDepth);

    var texCoord: vec2<f32> = vec2<f32>(
        f32(iSampleX) / f32(uniformData.miScreenWidth),
        f32(iSampleY) / f32(uniformData.miScreenHeight) 
    );

    var screenPosition: vec4<f32> = vec4<f32>(
        texCoord.x * 2.0f - 1.0f,
        (texCoord.y * 2.0f - 1.0f) * -1.0f,
        fViewDepth,
        1.0f
    );

    // convert screen space to world space
    var worldPosition: vec4<f32> = screenPosition * uniformData.mInverseViewProjectionMatrix;
    worldPosition.x /= worldPosition.w;
    worldPosition.y /= worldPosition.w;
    worldPosition.z /= worldPosition.w;

    let normal: vec3<f32> = getNormal(
        iSampleX, 
        iSampleY);

    //var retColor: vec4<f32> = vec4<f32>(worldPosition.x, worldPosition.y, worldPosition.z, 1.0f);
    var retColor: vec4<f32> = vec4<f32>(1.0f, 1.0f, 1.0f, 1.0f);
    var debugColor: vec4<f32> = vec4<f32>(1.0f, 1.0f, 1.0f, 1.0f);
    for(var iLightFrustumPartition: u32 = 0u; iLightFrustumPartition < 12u; iLightFrustumPartition++)
    {
        // convert from world space to light space
        var lightClipSpace: vec4<f32> = vec4<f32>(
            worldPosition.x,
            worldPosition.y,
            worldPosition.z,
            1.0f) * uniformData.maLightViewProjectionMatrices[iLightFrustumPartition]; 

        lightClipSpace.x /= lightClipSpace.w;
        lightClipSpace.y /= lightClipSpace.w;
        lightClipSpace.z /= lightClipSpace.w;

        let lightSpaceUV: vec2<f32> = vec2<f32>(
            lightClipSpace.x * 0.5f + 0.5f,
            1.0f - (lightClipSpace.y * 0.5f + 0.5f)
        );

        let fTextureWidth: f32 = f32(textureDimensions(lightViewDepthTexture0, 0).x);
        let fTextureHeight: f32 = f32(textureDimensions(lightViewDepthTexture0, 0).y);

        let iOutputX: u32 = u32(lightSpaceUV.x * fTextureWidth);
        let iOutputY: u32 = u32(lightSpaceUV.y * fTextureHeight);

        if(lightSpaceUV.x >= 0.0f && lightSpaceUV.x <= 1.0f &&
           lightSpaceUV.y >= 0.0f && lightSpaceUV.y <= 1.0f)
        {
           var fLightViewDepth: f32 = 0.0f;
            if(iLightFrustumPartition == 0u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture0,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(1.0f, 0.0f, 0.0f, 1.0f);
            }
            else if(iLightFrustumPartition == 1u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture1,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(0.0f, 1.0f, 0.0f, 1.0f);
            }
            else if(iLightFrustumPartition == 2u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture2,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(0.0f, 0.0f, 1.0f, 1.0f);
            }
            else if(iLightFrustumPartition == 3u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture3,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(1.0f, 1.0f, 0.0f, 1.0f);
            }
            else if(iLightFrustumPartition == 4u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture4,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(1.0f, 0.5f, 0.0f, 1.0f);
            }
            else if(iLightFrustumPartition == 5u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture5,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(1.0f, 0.0f, 1.0f, 1.0f);
            }
            else if(iLightFrustumPartition == 6u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture6,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(1.0f, 0.0f, 0.5f, 1.0f);
            }
            else if(iLightFrustumPartition == 7u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture7,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(0.0f, 0.5f, 1.0f, 1.0f);
            }
            else if(iLightFrustumPartition == 8u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture8,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(0.0f, 0.5f, 1.0f, 1.0f);
            }
            else if(iLightFrustumPartition == 9u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture9,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(1.0f, 0.5f, 0.5f, 1.0f);
            }
            else if(iLightFrustumPartition == 10u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture10,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(0.5f, 0.5f, 0.0f, 1.0f);
            }
            else if(iLightFrustumPartition == 11u)
            {
                fLightViewDepth = textureLoad(
                    lightViewDepthTexture11,
                    vec2<u32>(iOutputX, iOutputY),
                    0
                ).x;

                debugColor = vec4<f32>(0.5f, 0.5f, 1.0f, 1.0f);
            }

            let lightDirection: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
            let fDP: f32 = dot(normal.xyz, lightDirection);
            var fBias: f32 = 0.001f;
            if(fDP < 0.0f)
            {
                fBias = -0.001f;
            }
            
            if(fLightViewDepth + fBias < lightClipSpace.z)
            {
                retColor = vec4<f32>(kfShadowColor, kfShadowColor, kfShadowColor, 1.0f);
            }

            let iTileX: u32 = u32((lightSpaceUV.x * f32(iOutputTextureSize)) / f32(iTileSize));
            let iTileY: u32 = u32((lightSpaceUV.y * f32(iOutputTextureSize)) / f32(iTileSize));
            
            // register the tile to be rendered
            let iRetBits: u32 = setBit(
                iTileX, 
                iTileY,
                iNumTilesPerRow,
                iLightFrustumPartition / 4u);

            // close up texture
            var fCloseupLightViewDepth: f32 = clamp(1.0f - inCloseUpLightViewShadow(
                worldPosition.xyz, 
                normal.xyz,
                0u), kfShadowColor, 1.0f);
            retColor.x = fCloseupLightViewDepth;
            retColor.y = fCloseupLightViewDepth;
            retColor.z = fCloseupLightViewDepth;
            
            break;
        }
    }

    //retColor.x *= debugColor.x;
    //retColor.y *= debugColor.y;
    //retColor.z *= debugColor.z;

    textureStore(
        viewShadowTexture,
        vec2<u32>(iSampleX, iSampleY),
        retColor
    );

    atomicAdd(&aiCounters[0], 1u);


    workgroupBarrier();

    if(iTotalThreadIndex <= iNumTotalVisibleEntries)
    {
        for(var i: u32 = 0u; i < 1000000u; i++)
        {
            if(aiCounters[0] >= uniformData.miScreenWidth * uniformData.miScreenHeight)
            {
                break;
            }
        }

        var iBits: u32 = aiVisibleTiles[iTotalThreadIndex];
        let iCurrPartition: u32 = iTotalThreadIndex / (iNumTilesPerRow * (iNumTilesPerRow / 32u));
        let iY: u32 = iTotalThreadIndex / (iNumTilesPerRow / 32u);
        for(var i: u32 = 0; i < 32u; i++)
        {
            let iCheckBit: u32 = (1u << i);
            let iBit: u32 = (iBits & iCheckBit);
            if(iBit > 0u)
            {
                let iX: u32 = i + (iTotalThreadIndex % 2u) * 32u;

                let iTileInfoIndex: u32 = atomicAdd(&aiCounters[1], 1u);
                aTilePageInfo[iTileInfoIndex].miX = iX;
                aTilePageInfo[iTileInfoIndex].miY = iY;
                aTilePageInfo[iTileInfoIndex].miPartition = iCurrPartition;
            }
        }
    }
}

/////
fn setBit(
    iX: u32,
    iY: u32,
    iTotalDimension: u32,
    iPartitionIndex: u32) -> u32
{
    // 32 bits per int
    let iBitDimensions: u32 = iTotalDimension / 32;

    let iIndexX: u32 = iX / 32;          
    let iIndexY: u32 = iY;

    let iTotalPartitionindex: u32 = iPartitionIndex * iTotalDimension * iBitDimensions;
    let iTotalIndex: u32 = iTotalPartitionindex + iIndexY * iBitDimensions + iIndexX;
    let iBitIndex: u32 = u32(iX % 32);
    let iOrValue: u32 = u32(1u << iBitIndex);

    var ret: u32 = atomicOr(&aiVisibleTiles[iTotalIndex], iOrValue);
    ret = atomicLoad(&aiVisibleTiles[iTotalIndex]);
    return ret;
}

/////
fn getBit(
    iX: u32,
    iY: u32,
    iTotalDimension: u32,
    iPartitionIndex: u32) -> u32
{
    // 32 bits per int
    let iBitDimensions: u32 = iTotalDimension / 32u;

    let iIndexX: u32 = iX / 32u;          
    let iIndexY: u32 = iY;

    let iTotalPartitionindex: u32 = iPartitionIndex * iTotalDimension * iBitDimensions;
    let iTotalIndex: u32 = iTotalPartitionindex + iIndexY * iBitDimensions + iIndexX;
    let iBitIndex: u32 = u32(iX % 32u);
    let iBitMask: u32 = u32(1u << iBitIndex);
    
    var ret = aiVisibleTiles[iTotalIndex] & iBitMask;
    ret = ret >> iBitIndex;

    return ret;
}

/////
fn inCloseUpLightViewShadow(
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    iMatrixIndex: u32) -> f32 
{
    // convert from world space to light space
    var lightClipSpace: vec4<f32> = vec4<f32>(
        worldPosition.x,
        worldPosition.y,
        worldPosition.z,
        1.0f) * aCloseUpViewMatrices[iMatrixIndex];

    lightClipSpace.x /= lightClipSpace.w;
    lightClipSpace.y /= lightClipSpace.w;
    lightClipSpace.z /= lightClipSpace.w;

    let lightSpaceUV: vec2<f32> = vec2<f32>(
        lightClipSpace.x * 0.5f + 0.5f,
        1.0f - (lightClipSpace.y * 0.5f + 0.5f)
    );

    var fLightViewDepth: f32 = 1000.0f;
    if(lightSpaceUV.x >= 0.0f && lightSpaceUV.x <= 1.0f &&
       lightSpaceUV.y >= 0.0f && lightSpaceUV.y <= 1.0f)
    {
        let fTextureWidth: f32 = f32(textureDimensions(lightViewDepthTexture0, 0).x);
        let fTextureHeight: f32 = f32(textureDimensions(lightViewDepthTexture0, 0).y);

        let iOutputX: u32 = u32(lightSpaceUV.x * fTextureWidth);
        let iOutputY: u32 = u32(lightSpaceUV.y * fTextureHeight);

        fLightViewDepth = textureLoad(
            closeUpLightViewTexture,
            vec2<u32>(iOutputX, iOutputY),
            0
        ).x;
    }

    let lightDirection: vec3<f32> = normalize(vec3<f32>(1.0f, -0.7f, 0.0f));
    let fDP: f32 = dot(normal, lightDirection * -1.0f);
    var fBias: f32 = 0.001f;
    if(fDP < 0.0f)
    {
        fBias = -0.001f;
    }

    return f32(fLightViewDepth + fBias < lightClipSpace.z);
}

/////
fn getWorldPosition(
    iSampleX: u32,
    iSampleY: u32) -> vec4<f32>
{
    var texCoord: vec2<f32> = vec2<f32>(
        f32(iSampleX) / f32(uniformData.miScreenWidth),
        f32(iSampleY) / f32(uniformData.miScreenHeight) 
    );

    var fViewDepth: f32 = textureLoad(
        viewDepthTexture,
        vec2<u32>(iSampleX, iSampleY),
        0
    ).x;

    var screenPosition: vec4<f32> = vec4<f32>(
        texCoord.x * 2.0f - 1.0f,
        (texCoord.y * 2.0f - 1.0f) * -1.0f,
        fViewDepth,
        1.0f
    );

    // convert screen space to world space
    var worldPosition: vec4<f32> = screenPosition * uniformData.mInverseViewProjectionMatrix;
    worldPosition.x /= worldPosition.w;
    worldPosition.y /= worldPosition.w;
    worldPosition.z /= worldPosition.w;

    return worldPosition;
}

/////
fn getNormal(
    iSampleX: u32,
    iSampleY: u32) -> vec3<f32>
{
    
    // convert screen space to world space
    var worldPosition0: vec4<f32> = getWorldPosition(
        iSampleX, 
        iSampleY);

    var worldPosition1: vec4<f32> = getWorldPosition(
        clamp(iSampleX + 1, 0u, uniformData.miScreenWidth), 
        iSampleY);

    var worldPosition2: vec4<f32> = getWorldPosition(
        iSampleX, 
        clamp(iSampleY + 1, 0u, uniformData.miScreenHeight));

    let diff0: vec3<f32> = normalize(worldPosition1.xyz - worldPosition0.xyz);
    let diff1: vec3<f32> = normalize(worldPosition2.xyz - worldPosition0.xyz);

    let normal: vec3<f32> = cross(diff1, diff0);

    return normal;
}