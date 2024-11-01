
@group(0) @binding(0)
var lightViewClipSpaceTexture: texture_storage_2d<rgba32float, write>;

@group(0) @binding(1)
var<storage, read_write> aiNeededTiles: array<vec2<u32>>;

@group(0) @binding(2)
var <storage, read_write> aiVisibleTiles: array<atomic<u32>>;

@group(0) @binding(3)
var <storage, read_write> aiCounters: array<atomic<i32>>;

@group(0) @binding(4)
var <storage, read_write> aValidTiles: array<vec2<u32>>;

@group(0) @binding(5)
var depthTexture: texture_2d<f32>;


struct UniformData 
{
    mViewProjectionMatrix: mat4x4<f32>,
    mLightViewProjectionMatrix: mat4x4<f32>,
    mInverseViewProjectionMatrix: mat4x4<f32>,

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

    let iStartGlobalIndex: u32 = iTotalWorkgroup * 16;

    let iSampleX: u32 = workGroupIndex.x * iLocalThreadSize + localIndex.x;
    let iSampleY: u32 = workGroupIndex.y * iLocalThreadSize + localIndex.y;

    var depthValue: vec4<f32> = textureLoad(
        depthTexture,
        vec2<u32>(iSampleX, iSampleY),
        0
    );

    var texCoord: vec2<f32> = vec2<f32>(
        f32(iSampleX) / f32(uniformData.miScreenWidth),
        f32(iSampleY) / f32(uniformData.miScreenHeight) 
    );

    var screenPosition: vec4<f32> = vec4<f32>(
        texCoord.x * 2.0f - 1.0f,
        (texCoord.y * 2.0f - 1.0f) * -1.0f,
        fract(depthValue.w),
        1.0f
    );

    var worldPosition: vec4<f32> = screenPosition * uniformData.mInverseViewProjectionMatrix;
    worldPosition.x /= worldPosition.w;
    worldPosition.y /= worldPosition.w;
    worldPosition.z /= worldPosition.w;

    var lightClipSpace: vec4<f32> = vec4<f32>(
        worldPosition.x,
        worldPosition.y,
        worldPosition.z,
        1.0f) * uniformData.mLightViewProjectionMatrix; 

    let lightSpaceUV: vec2<f32> = vec2<f32>(
        lightClipSpace.x * 0.5f + 0.5f,
        1.0f - (lightClipSpace.y * 0.5f + 0.5f)
    );

    let iOutputX: u32 = u32(lightSpaceUV.x * f32(uniformData.miScreenWidth));
    let iOutputY: u32 = u32(lightSpaceUV.y * f32(uniformData.miScreenHeight));

    if(iOutputX >= 0u && iOutputX < uniformData.miScreenWidth &&
       iOutputY >= 0u && iOutputY < uniformData.miScreenHeight)
    {
        textureStore(
            lightViewClipSpaceTexture,
            vec2<u32>(iOutputX, iOutputY),
            lightClipSpace
        );

        let iOutputTextureSize: u32 = 8192u;
        let iTileSize: u32 = 128u;
        let fOneOverNumTiles: f32 = f32(iTileSize) / f32(iOutputTextureSize);
        let iNumTilesPerRow: u32 = iOutputTextureSize / iTileSize;

        let iTileX: u32 = u32((lightSpaceUV.x * f32(iOutputTextureSize)) / f32(iTileSize));
        let iTileY: u32 = u32((lightSpaceUV.y * f32(iOutputTextureSize)) / f32(iTileSize));
        
        saTilePages[iLocalThread] = vec2<u32>(iTileX, iTileY);

        // register the tile to be rendered
        let iRetBits: u32 = setBit(
            i32(iTileX), 
            i32(iTileY),
            i32(iNumTilesPerRow));

    }

    workgroupBarrier();

    var iNumTiles: u32 = 0u;
    if(iLocalThread == 0)
    {

        for(var i: u32 = 0u; i < 16u; i++)
        {
            aiNeededTiles[iStartGlobalIndex + i] = vec2<u32>(9999u, 9999u);
        }

        for(var i: u32 = 0u; i < 256u; i++)
        {
            var bFound: bool = false;
            for(var j: u32 = 0u; j < iNumTiles; j++)
            {
                if(saTilePages[i].x == aiNeededTiles[iStartGlobalIndex + j].x && saTilePages[i].y == aiNeededTiles[iStartGlobalIndex + j].y)
                {
                    bFound = true;
                    break;
                }
            }       

            if(!bFound && saTilePages[i].x != 0 && saTilePages[i].y != 0)
            {
                aiNeededTiles[iStartGlobalIndex + iNumTiles] = saTilePages[i];
                iNumTiles += 1u;
            }
        }

        atomicAdd(&aiCounters[0], 1);
    }

    workgroupBarrier();

/*
    if(iLocalThread == 0 && iTotalWorkgroup == 0)
    {
        for(var iLoop: u32 = 0u; iLoop < 10000000u; iLoop++)
        {
            if(aiCounters[0] >= 1200)
            {
                var iNumEntries: u32 = 1u;
                var i: u32 = 0u;
                aValidTiles[0] = aiNeededTiles[i];
                for(i = 0u;; i += 1u)
                {
                    if(aiNeededTiles[i].x == 0u && aiNeededTiles[i].y == 0u)
                    {
                        break;
                    }

                    if(aiNeededTiles[i].x == 9999)
                    {
                        continue;
                    }

                    var bFound: bool = false;
                    for(var j: u32 = 0u; j < iNumEntries; j++)
                    {
                        if(aiNeededTiles[i].x == aValidTiles[j].x && aiNeededTiles[i].y == aValidTiles[j].y)
                        {
                            bFound = true;
                            break;
                        }
                    }

                    if(!bFound)
                    {
                        aValidTiles[iNumEntries] = aiNeededTiles[i];
                        iNumEntries += 1u;
                    }
                    
                }

                break;
            }
        }
    }
*/
}

/////
fn setBit(
    iX: i32,
    iY: i32,
    iTotalDimension: i32) -> u32
{
    // 32 bits per int
    let iBitDimensions: i32 = iTotalDimension / 32;

    let iIndexX: i32 = iX / 32;          
    let iIndexY: i32 = iY;

    let iTotalIndex: i32 = iIndexY * iBitDimensions + iIndexX;
    let iBitIndex: u32 = u32(iX % 32);
    let iOrValue: u32 = u32(1u << iBitIndex);

    var ret: u32 = atomicOr(&aiVisibleTiles[iTotalIndex], iOrValue);
    ret = atomicLoad(&aiVisibleTiles[iTotalIndex]);
    return ret;
}

/////
fn getBit(
    iX: i32,
    iY: i32,
    iTotalDimension: i32) -> u32
{
    // 32 bits per int
    let iBitDimensions: i32 = iTotalDimension / 32;

    let iIndexX: i32 = iX / 32;          
    let iIndexY: i32 = iY;

    let iTotalIndex: i32 = iIndexY * iBitDimensions + iIndexX;
    let iBitIndex: u32 = u32(iX % 32);
    let iBitMask: u32 = u32(1u << iBitIndex);
    
    var ret = aiVisibleTiles[iTotalIndex] & iBitMask;
    ret = ret >> iBitIndex;

    return ret;
}