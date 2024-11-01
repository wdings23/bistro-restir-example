struct BrickAndBrixelInfo
{
    mBrickIndex: vec3<i32>,
    mBrixelIndex: vec3<i32>,

    miBrickArrayIndex: u32,
    miBrixelArrayIndex: u32,
};

struct BrickInfo
{
    mPosition: vec3<f32>,
    miBrixelIndex: u32,
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


@group(0) @binding(0)
var<storage, read_write> aBrixelRadiance: array<vec4<f32>>;

@group(0) @binding(1)
var debugTexture0: texture_storage_2d<rgba32float, write>;

@group(0) @binding(2)
var debugTexture1: texture_storage_2d<rgba32float, write>;

@group(0) @binding(3)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(4)
var radianceTexture: texture_2d<f32>;

@group(0) @binding(5)
var<storage, read> aBBox: array<i32>;

@group(0) @binding(6)
var<storage, read> aiBrickToBrixelMapping: array<i32>;

@group(0) @binding(7)
var<storage, read> aBricks: array<BrickInfo>;

@group(0) @binding(8)
var<storage, read_write> debugBuffer: array<vec4<f32>>;

struct UniformData
{
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    mfPositionScale: f32,
};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<uniform> defaultUniformData: DefaultUniformData;

const iNumThreads: u32 = 256u;

/////
@compute
@workgroup_size(iNumThreads)
fn cs_main(
    @builtin(num_workgroups) numWorkGroups: vec3<u32>,
    @builtin(local_invocation_index) iLocalIndex: u32,
    @builtin(workgroup_id) workGroup: vec3<u32>)
{
    let iNumTotalScreenPixels: u32 = u32(defaultUniformData.miScreenWidth * defaultUniformData.miScreenHeight);
    let iNumTotalThreads: u32 = numWorkGroups.x * iNumThreads;
    let iTotalThreadIndex: u32 = workGroup.x * iNumThreads + iLocalIndex;

    // mesh's bounding box
    let meshBBox0: vec3<f32> = vec3<f32>(
        f32(aBBox[3]) * 0.001f,
        f32(aBBox[4]) * 0.001f,
        f32(aBBox[5]) * 0.001f
    ) * uniformData.mfPositionScale;

    let meshBBox1: vec3<f32> = vec3<f32>(
        f32(aBBox[0]) * 0.001f,
        f32(aBBox[1]) * 0.001f,
        f32(aBBox[2]) * 0.001f
    ) * uniformData.mfPositionScale;

    let bboxDimension: vec3<f32> = vec3<f32>(
        ceil(meshBBox1.x) - floor(meshBBox0.x),
        ceil(meshBBox1.y) - floor(meshBBox0.y),
        ceil(meshBBox1.z) - floor(meshBBox0.z)
    );

    // number of bricks from bounding box
    let numBricks: vec3<i32> = vec3<i32>(bboxDimension / vec3<f32>(uniformData.mfBrickDimension, uniformData.mfBrickDimension, uniformData.mfBrickDimension)); 
    
    // set radiance for brixels
    for(var iPixel: i32 = i32(iTotalThreadIndex); iPixel < i32(iNumTotalScreenPixels); iPixel += i32(iNumTotalThreads))
    {
        let iScreenX: i32 = iPixel % i32(defaultUniformData.miScreenWidth);
        let iScreenY: i32 = iPixel / i32(defaultUniformData.miScreenWidth);

        let worldPosition: vec4<f32> = textureLoad(
            worldPositionTexture, 
            vec2<i32>(iScreenX, iScreenY),
            0);

        let radiance: vec4<f32> = textureLoad(
            radianceTexture,
            vec2<i32>(iScreenX, iScreenY),
            0);
   
        //setBrixelRadiance(
        //    worldPosition.xyz,
        //    radiance.xyz,
        //    numBricks,
        //    meshBBox0,
        //    iScreenX,
        //    iScreenY);

        let brickAndBrixelIndices: BrickAndBrixelInfo = getBrickAndBrickIndexFromPosition(
            worldPosition.xyz,
            numBricks,
            meshBBox0);
        
        let bInsideBrickBoundingBox: bool = (
            brickAndBrixelIndices.mBrickIndex.x >= 0 && brickAndBrixelIndices.mBrickIndex.x < numBricks.x &&
            brickAndBrixelIndices.mBrickIndex.y >= 0 && brickAndBrixelIndices.mBrickIndex.y < numBricks.y &&
            brickAndBrixelIndices.mBrickIndex.z >= 0 && brickAndBrixelIndices.mBrickIndex.z < numBricks.z
        );

        if(bInsideBrickBoundingBox)
        {
            let iBrixelArrayIndex: i32 = aiBrickToBrixelMapping[brickAndBrixelIndices.miBrickArrayIndex];
            //if(iBrixelArrayIndex > 0)
            {
                let brickPosition: vec3<f32> = aBricks[brickAndBrixelIndices.miBrickArrayIndex].mPosition;

                textureStore(
                    debugTexture0,
                    vec2<i32>(iScreenX, iScreenY),
                    vec4<f32>(
                        brickPosition.x,
                        brickPosition.y,
                        brickPosition.z,
                        1.0f)
                );

                textureStore(
                    debugTexture1,
                    vec2<i32>(iScreenX, iScreenY),
                    vec4<f32>(
                        f32(brickAndBrixelIndices.miBrickArrayIndex),
                        f32(iBrixelArrayIndex),
                        0.0f,
                        0.0f)
                );

                aBrixelRadiance[u32(iBrixelArrayIndex) * 512u + brickAndBrixelIndices.miBrixelArrayIndex] = vec4<f32>(
                    radiance.x,
                    radiance.y,
                    radiance.z,
                    f32(brickAndBrixelIndices.miBrickArrayIndex)
                    );

                if(brickAndBrixelIndices.miBrickArrayIndex == 273u && iScreenY <= 150)
                {
                    if(radiance.z >= 0.2f)
                    {
                        debugBuffer[0] = radiance;
                        debugBuffer[1] = vec4<f32>(
                            f32(iScreenX),
                            f32(iScreenY),
                            0.0f,
                            0.0f
                        );
                        debugBuffer[2] = vec4<f32>(
                            f32(brickAndBrixelIndices.mBrickIndex.x),
                            f32(brickAndBrixelIndices.mBrickIndex.y),
                            f32(brickAndBrixelIndices.mBrickIndex.z),
                            f32(brickAndBrixelIndices.miBrickArrayIndex)
                        );
                        debugBuffer[3] = vec4<f32>(
                            worldPosition.x,
                            worldPosition.y,
                            worldPosition.z,
                            1.0f
                        );
                        debugBuffer[4] = vec4<f32>(
                            f32(iBrixelArrayIndex),
                            0.0f,
                            0.0f,
                            0.0f
                        );

                    }
                }
                
            }
        }
        
    }
}

/////
fn setBrixelRadiance(
    worldPosition: vec3<f32>,
    radiance: vec3<f32>,
    numTotalBricks: vec3<i32>,
    meshBBox0: vec3<f32>,
    iScreenX: i32,
    iScreenY: i32) -> bool
{
    var bRet: bool = false;
    let brickAndBrixelIndices: BrickAndBrixelInfo = getBrickAndBrickIndexFromPosition(
        worldPosition,
        numTotalBricks,
        meshBBox0);
    
    if(brickAndBrixelIndices.mBrickIndex.x >= 0 && brickAndBrixelIndices.mBrickIndex.x < numTotalBricks.x &&
       brickAndBrixelIndices.mBrickIndex.y >= 0 && brickAndBrixelIndices.mBrickIndex.y < numTotalBricks.y &&
       brickAndBrixelIndices.mBrickIndex.z >= 0 && brickAndBrixelIndices.mBrickIndex.z < numTotalBricks.z)
    {
        let brickPosition: vec3<f32> = aBricks[brickAndBrixelIndices.miBrickArrayIndex].mPosition;

        let iBrixelArrayIndex: i32 = aiBrickToBrixelMapping[brickAndBrixelIndices.miBrickArrayIndex];
        if(iBrixelArrayIndex > 0)
        {
            let iTotalBrixelArrayIndex: u32 = u32(iBrixelArrayIndex) * 512u + brickAndBrixelIndices.miBrixelArrayIndex;
            aBrixelRadiance[iTotalBrixelArrayIndex] = vec4<f32>(
                brickPosition.x,
                brickPosition.y,
                brickPosition.z,
                f32(iScreenY));

            bRet = true;
        }
    }

    return bRet;
}

// get brixel 
fn getBrickAndBrickIndexFromPosition(
    worldPosition: vec3<f32>,
    numTotalBricks: vec3<i32>,
    meshBBox0: vec3<f32>) -> BrickAndBrixelInfo
{
    var ret: BrickAndBrixelInfo;

    let positionDiff: vec3<f32> = worldPosition.xyz * uniformData.mfPositionScale - meshBBox0;
    let brickIndex: vec3<f32> = floor(positionDiff / uniformData.mfBrickDimension);
    ret.mBrickIndex = vec3<i32>(
        i32(brickIndex.x),
        i32(brickIndex.y),
        i32(brickIndex.z)
    );
    ret.miBrickArrayIndex = u32(
        brickIndex.z * f32(numTotalBricks.x * numTotalBricks.y) +
        brickIndex.y * f32(numTotalBricks.x) +
        brickIndex.x);

    let fracHitGlobalBrickIndexFloat: vec3<f32> = fract(positionDiff) * 8.0f;
    ret.mBrixelIndex = vec3<i32>(
        i32(fracHitGlobalBrickIndexFloat.x),
        i32(fracHitGlobalBrickIndexFloat.y),
        i32(fracHitGlobalBrickIndexFloat.z));
    ret.miBrixelArrayIndex = u32(ret.mBrixelIndex.x + ret.mBrixelIndex.y * 8 + ret.mBrixelIndex.z * 64);

    return ret;
}