const FLT_MAX: f32 = 1000000.0f;
const UINT32_MAX: u32 = 0xffffffff;
const PI: f32 = 3.14159f;

struct RandomResult 
{
    mfNum: f32,
    miSeed: u32,
};

struct Ray
{
    mOrigin: vec4<f32>,
    mDirection: vec4<f32>,
    mfT: vec4<f32>,
};

struct SphereTraceInfo
{
    mHitPosition: vec3<f32>,
    mfClosestLengthFromStart: f32,
    mfDistanceToSeed: f32,
    miBrickIndex: u32,
    miBrixelIndex: u32,
    mDebug: vec4<f32>,
};

struct UniformData
{
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    mfPositionScale: f32,
};

struct DirectionalShadowInfo
{
    mRadiance: vec3<f32>,
    mRandomResult: RandomResult,
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
struct FragmentOutput 
{
    @location(0) radiance : vec4<f32>,
    @location(1) rayCounts: vec4<f32>,
    @location(2) directionalShadow: vec4<f32>,
    @location(3) rayDirection: vec4<f32>, 
    @location(4) hitBrickAndBrixels: vec4<f32>,
    @location(5) shadowMoment: vec4<f32>,
    @location(6) debug: vec4<f32>,
};

struct BrixelRayHit
{
    miBrickIndex: u32,
    miBrixelIndex: u32
};

@group(0) @binding(0)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(3)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(4)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(5)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(6)
var prevRayCountTexture: texture_2d<f32>;

@group(0) @binding(7)
var temporalRestirRayDirectionTexture: texture_2d<f32>;

@group(0) @binding(8)
var<storage, read> aBrixelDistances: array<f32>;

@group(0) @binding(9)
var<storage, read> aiBrickToBrixelMapping: array<i32>;

@group(0) @binding(10)
var<storage, read> aBBox: array<i32>;

@group(0) @binding(11)
var<storage, read_write> aiCounters: array<atomic<i32>>;

@group(0) @binding(12)
var<storage, read_write> aHitBrickQueue: array<BrixelRayHit>;

@group(0) @binding(13)
var<storage, read_write> aiQueueCounter: array<atomic<u32>>;

@group(0) @binding(14)
var<storage, read> aTrianglePositions: array<vec4<f32>>;

@group(0) @binding(15)
var<storage, read> aBrixelBarycentricCoordinates: array<vec3<f32>>;

@group(0) @binding(16)
var<storage, read_write> debugBuffer: array<f32>;

@group(0) @binding(17)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var blueNoiseTexture: texture_2d<f32>;

@group(1) @binding(2)
var<storage, read> aiTriangleIndices: array<u32>;

@group(1) @binding(3)
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

    out.rayCounts = textureSample(
        prevRayCountTexture,
        textureSampler,
        in.texCoord
    );

    var randomResult: RandomResult = initRand(
        u32(in.texCoord.x * 100.0f + in.texCoord.y * 200.0f + defaultUniformData.mfRand0 * 100.0f),
        u32(in.pos.x * 10.0f + in.pos.z * 20.0f + defaultUniformData.mfRand1 * 100.0f),
        10u);

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
        out.radiance = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        return out;
    }

    // check for disocclusion for previous history pixel
    var fDisocclusion: f32 = 0.0f;
    var prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);
    if(isPrevUVOutOfBounds(in.texCoord))
    {
        fDisocclusion = 1.0f;
    }
    else
    {
        fDisocclusion = f32(isDisoccluded2(in.texCoord, in.texCoord));
    }
    out.rayCounts *= (1.0f - fDisocclusion);

out.rayCounts = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

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

    let brickBBox0: vec3<f32> = vec3<f32>(
        floor(meshBBox0.x),
        floor(meshBBox0.y),
        floor(meshBBox0.z)
    );

    let brickBBox1: vec3<f32> = vec3<f32>(
        ceil(meshBBox1.x),
        ceil(meshBBox1.y),
        ceil(meshBBox1.z)
    );

    // number of bricks from bounding box
    let numBricks: vec3<i32> = vec3<i32>(bboxDimension / vec3<f32>(uniformData.mfBrickDimension, uniformData.mfBrickDimension, uniformData.mfBrickDimension)); 

    var screenCoord: vec2<u32> = vec2<u32>(
        u32(in.texCoord.x * f32(defaultUniformData.miScreenWidth)),
        u32(in.texCoord.y * f32(defaultUniformData.miScreenHeight))
    );

    let textureSize: vec2<u32> = textureDimensions(blueNoiseTexture, 0);
    let iTotalTextureSize: u32 = textureSize.x * textureSize.y;

    //var iOffsetX: u32 = defaultUniformData.miFrameIndex % textureSize.x;
    //var iOffsetY: u32 = defaultUniformData.miFrameIndex / textureSize.y; 

    let iTileSize: u32 = 32u;
    let iNumTilesPerRow: u32 = textureSize.x / iTileSize;
    let iNumTotalTiles: u32 = iNumTilesPerRow * iNumTilesPerRow;
    

    out.hitBrickAndBrixels = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

    let scaledWorldPosition: vec3<f32> = worldPosition.xyz * uniformData.mfPositionScale;

    // ray samples
    let iNumSamples: i32 = 20;
    var iCurrIndex: u32 = u32(defaultUniformData.miFrame * iNumSamples);
    var ret: SphereTraceInfo;
    for(var iSample: i32 = 0; iSample < iNumSamples; iSample++)
    {
        let iTileX: u32 = (iCurrIndex + u32(iNumSamples)) % iNumTilesPerRow;
        let iTileY: u32 = ((iCurrIndex + u32(iNumSamples)) / iNumTilesPerRow) % (iNumTilesPerRow * iNumTilesPerRow);

        let iTileOffsetX: u32 = (iCurrIndex + u32(iNumSamples)) % iTileSize;
        let iTileOffsetY: u32 = ((iCurrIndex + u32(iNumSamples)) / iTileSize) % (iTileSize * iTileSize);

        let iOffsetX: u32 = iTileOffsetX + iTileX * iTileSize;
        let iOffsetY: u32 = iTileOffsetY + iTileY * iTileSize; 

        //let iOffsetX: u32 = defaultUniformData.miFrame % textureSize.x;
        //let iOffsetY: u32 = (defaultUniformData.miFrame / textureSize.x) % iTotalTextureSize;

        screenCoord.x = (screenCoord.x + iOffsetX) % textureSize.x;
        screenCoord.y = (screenCoord.y + iOffsetY) % textureSize.y;
        let sampleUV: vec2<f32> = vec2<f32>(
            f32(screenCoord.x) / f32(textureSize.x),
            f32(screenCoord.y) / f32(textureSize.y) 
        );

        var blueNoise: vec3<f32> = textureSample(
            blueNoiseTexture,
            textureSampler,
            sampleUV
        ).xyz;

        //randomResult = nextRand(randomResult.miSeed);
        //let fRand0: f32 = randomResult.mfNum;
        //randomResult = nextRand(randomResult.miSeed);
        //let fRand1: f32 = randomResult.mfNum;

        let fRand0: f32 = blueNoise.x;
        let fRand1: f32 = blueNoise.y;

        // random ray direction
        var ray: Ray = uniformSampling(
            scaledWorldPosition,
            normal.xyz,
            fRand0,
            fRand1);

        let tMinMax: vec2<f32> = rayBoxIntersect2(
            scaledWorldPosition,
            ray.mDirection.xyz,
            meshBBox0,
            meshBBox1
        );
        let bIntersect: bool = (tMinMax.x <= tMinMax.y);
        let bInsideBBox: bool = (
            scaledWorldPosition.x >= meshBBox0.x && scaledWorldPosition.x <= meshBBox1.x &&
            scaledWorldPosition.y >= meshBBox0.y && scaledWorldPosition.y <= meshBBox1.y &&
            scaledWorldPosition.z >= meshBBox0.z && scaledWorldPosition.z <= meshBBox1.z);

        // sphere trace against sdf
        if(bInsideBBox || bIntersect)
        {
            // only apply ray direction position on pixels outside of the bounding box
            var startPosition: vec3<f32> = scaledWorldPosition; 
            if(bIntersect && !bInsideBBox)
            {
                startPosition += ray.mDirection.xyz * tMinMax.x;
            }

            startPosition += ray.mDirection.xyz * uniformData.mfBrixelDimension * 2.25f;
            ret = sphereTraceSDFDistance(
                startPosition,
                ray.mDirection.xyz,
                brickBBox0,
                numBricks,
                uniformData.mfBrickDimension,
                uniformData.mfBrixelDimension,
                8u,
                30.0f,
                0.0f,
                true
            );

            let fMaxAmbientOcclusionDistance: f32= 0.8f * uniformData.mfPositionScale;
            if(ret.miBrickIndex != UINT32_MAX)
            {
                let fClosestLengthFromStart: f32 = ret.mfClosestLengthFromStart;
                let fPct: f32 = fClosestLengthFromStart / fMaxAmbientOcclusionDistance;
                
                out.rayCounts.x += (1.0f - fPct);

                out.rayDirection = ray.mDirection;
                out.hitBrickAndBrixels.x = f32(ret.miBrickIndex);
                out.hitBrickAndBrixels.y = f32(ret.miBrixelIndex);

                let iQueueIndex: u32 = atomicAdd(&aiQueueCounter[0], 1u);
                aHitBrickQueue[iQueueIndex].miBrickIndex = ret.miBrickIndex;
                aHitBrickQueue[iQueueIndex].miBrixelIndex = ret.miBrixelIndex;

                // get brixel 
                let positionDiff: vec3<f32> = worldPosition.xyz * uniformData.mfPositionScale - meshBBox0;
                var brickIndex: vec3<f32> = floor(positionDiff / uniformData.mfBrickDimension);
                let iGlobalBrickArrayIndex: u32 = u32(
                    brickIndex.z * f32(numBricks.x * numBricks.y) +
                    brickIndex.y * f32(numBricks.x) +
                    brickIndex.x);
                let fracHitGlobalBrickIndexFloat: vec3<f32> = fract(positionDiff) * 8.0f;
                let hitBrixelIndex: vec3<i32> = vec3<i32>(
                    i32(fracHitGlobalBrickIndexFloat.x),
                    i32(fracHitGlobalBrickIndexFloat.y),
                    i32(fracHitGlobalBrickIndexFloat.z));

            }
        }

        out.rayCounts.y += 1.0f;
    }

    let posDiff: vec3<f32> = worldPosition.xyz * uniformData.mfPositionScale - meshBBox0;
    var brickIndex: vec3<f32> = floor(posDiff / uniformData.mfBrickDimension); 

    let fAO: f32 = smoothstep(
        0.0f, 
        0.9f, 
        1.0f - out.rayCounts.x / out.rayCounts.y);

    let iGlobalBrickArrayIndex: u32 = u32(
        brickIndex.z * f32(numBricks.x * numBricks.y) +
        brickIndex.y * f32(numBricks.x) +
        brickIndex.x);

    //out.radiance = vec4<f32>(brickIndex.x, brickIndex.y, brickIndex.z, f32(iGlobalBrickArrayIndex));
    out.radiance = vec4<f32>(fAO, fAO, fAO, 1.0f);

    let iBrixeDistanceArraylIndex: i32 = aiBrickToBrixelMapping[iGlobalBrickArrayIndex];
    let fracHitGlobalBrickIndexFloat: vec3<f32> = fract(posDiff) * 8.0f;
    let hitBrixelIndex: vec3<i32> = vec3<i32>(
        i32(fracHitGlobalBrickIndexFloat.x),
        i32(fracHitGlobalBrickIndexFloat.y),
        i32(fracHitGlobalBrickIndexFloat.z));
    
    let iCheckBrixelIndex: u32 = u32(hitBrixelIndex.z) * 64u + u32(hitBrixelIndex.y) * 8u + u32(hitBrixelIndex.x);
    let fBrixelDistance: f32 = aBrixelDistances[u32(iBrixeDistanceArraylIndex) * 512u + iCheckBrixelIndex];

    //out.debug = vec4<f32>(fBrixelDistance, f32(iGlobalBrickArrayIndex), f32(iCheckBrixelIndex), f32(iBrixeDistanceArraylIndex));
    //out.debug = vec4<f32>(posDiff.x, posDiff.y, posDiff.z, 1.0f);

    let lightDirection: vec3<f32> = normalize(vec3<f32>(-1.0f, 0.8f, 0.0f));

    let rayDirection: vec3<f32> = textureSample(
        temporalRestirRayDirectionTexture,
        textureSampler,
        in.texCoord).xyz;

    //let shadowInfo: DirectionalShadowInfo = directionalShadow(
    //    worldPosition.xyz,
    //    normal.xyz,
    //    lightDirection,
    //    brickBBox0,
    //    brickBBox1,
    //    numBricks,
    //    screenCoord,
    //    randomResult);
    //out.directionalShadow = vec4<f32>(
    //    shadowInfo.mRadiance.x,
    //    shadowInfo.mRadiance.y,
    //    shadowInfo.mRadiance.z,
    //    1.0f);
    //out.shadowMoment.x = computeLuminance(shadowInfo.mRadiance.xyz);
    //out.shadowMoment.y = out.shadowMoment.x * out.shadowMoment.x;

    out.directionalShadow = vec4<f32>(1.0f, 1.0f, 1.0f, 1.0f);
    out.shadowMoment.x = 0.0f;
    out.shadowMoment.y = 0.0f;

    return out;
}

/////
fn sphereTraceSDFDistance(
    startPosition: vec3<f32>,
    rayDirection: vec3<f32>,
    minGlobalBrickPosition: vec3<f32>,
    numGlobalBricks: vec3<i32>,
    fBrickDimension: f32,
    fBrixelDimension: f32,
    iNumBrixelsPerBrickRow: u32,
    fNumSteps: f32,
    fDistanceToDraw: f32,
    bEarlyExitOutOfBounds: bool) -> SphereTraceInfo
{
    var ret: SphereTraceInfo;
    ret.mHitPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    ret.miBrickIndex = UINT32_MAX;
    ret.miBrixelIndex = UINT32_MAX;

    let kfScatter: f32 = 64.0f;   
    var fRes: f32 = 1024.0f;
    var fTotalDistance: f32 = 0.0f;

    let iNumBrixelsPerBrickRowSquared: u32 = iNumBrixelsPerBrickRow * iNumBrixelsPerBrickRow;

    var closestPosition: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    var fClosestBrixelDistanceToSeed = FLT_MAX;
    var iClosestBrickIndex: u32 = UINT32_MAX;
    var iClosestBrixelIndex: u32 = UINT32_MAX;
    var fClosestLengthFromStart: f32 = FLT_MAX;

    let fOneOverHitBrickSize: f32 = 1.0f / fBrickDimension;
    
    // step through the mesh brixels along the ray
    var fPrevDistanceToSeed: f32 = FLT_MAX;
    var fDistanceToSeed: f32 = FLT_MAX;
    var currPosition: vec3<f32> = startPosition;
    var lastPosition: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);

    let fNumBrixelsPerBrickRow: f32 = f32(iNumBrixelsPerBrickRow);
    let maxGlobalBrickPosition: vec3<f32> = minGlobalBrickPosition + vec3<f32>(f32(numGlobalBricks.x), f32(numGlobalBricks.y), f32(numGlobalBricks.z)) * fBrickDimension;
    
    let oneOverRayDirection: vec3<f32> = 1.0f / rayDirection;

    let iNumBrickXY: i32 = numGlobalBricks.x * numGlobalBricks.y;
    let iNumBrickX: i32 = numGlobalBricks.x;
    for(var fInc: f32 = 0.0f; fInc < fNumSteps; fInc += 1.0f)
    {
        if(bEarlyExitOutOfBounds && 
           (currPosition.x < minGlobalBrickPosition.x || currPosition.x > maxGlobalBrickPosition.x ||
            currPosition.x < minGlobalBrickPosition.y || currPosition.x > maxGlobalBrickPosition.y ||
            currPosition.x < minGlobalBrickPosition.z || currPosition.x > maxGlobalBrickPosition.z))
        {
            break;
        }

        var positionDiff: vec3<f32> = currPosition - minGlobalBrickPosition;
        var hitGlobalBrickIndexFloat: vec3<f32> = floor(positionDiff / fBrickDimension);
        var iGlobalBrickArrayIndex: u32 = u32(
            hitGlobalBrickIndexFloat.z * f32(numGlobalBricks.x * numGlobalBricks.y) +
            hitGlobalBrickIndexFloat.y * f32(numGlobalBricks.x) +
            hitGlobalBrickIndexFloat.x
        );
        
        lastPosition = currPosition;
        
        if(hitGlobalBrickIndexFloat.x < 0.0f || hitGlobalBrickIndexFloat.x >= f32(numGlobalBricks.x) ||
           hitGlobalBrickIndexFloat.y < 0.0f || hitGlobalBrickIndexFloat.y >= f32(numGlobalBricks.y) ||
           hitGlobalBrickIndexFloat.z < 0.0f || hitGlobalBrickIndexFloat.z >= f32(numGlobalBricks.z))
        {
            let tMinMax: vec2<f32> = rayBoxIntersect2(
                currPosition,
                rayDirection,
                minGlobalBrickPosition,
                minGlobalBrickPosition + vec3<f32>(
                    f32(numGlobalBricks.x) * fBrickDimension,
                    f32(numGlobalBricks.y) * fBrickDimension,
                    f32(numGlobalBricks.z) * fBrickDimension)
            );

            if(tMinMax.x > tMinMax.y)
            {
                break;
            }

            // compute the smallest increment along the ray direction to get to the next brick
            var fDistance: f32 = clamp(tMinMax.x, fBrixelDimension, fBrickDimension);
            currPosition = currPosition + rayDirection * fDistance;

            fTotalDistance += fDistance;
            continue;
        }

        var iBrixelDistanceArrayIndex: i32 = aiBrickToBrixelMapping[iGlobalBrickArrayIndex];
        if(iBrixelDistanceArrayIndex <= 0)
        {
            // didn't hit any valid brick, continue in ray direction

            let fDistance: f32 = fBrixelDimension; //(hitMeshBrick.mfShortestDistanceToOthers < fBrixelDimension) ? fBrixelDimension : hitMeshBrick.mfShortestDistanceToOthers;
            currPosition += rayDirection * fDistance;
            fTotalDistance += fDistance;

            continue;
        }

        // brixel index from the fractional part of the position difference
        let fracHitGlobalBrickIndexFloat: vec3<f32> = fract(positionDiff) * fNumBrixelsPerBrickRow;
        let hitBrixelIndex: vec3<i32> = vec3<i32>(
            i32(fracHitGlobalBrickIndexFloat.x),
            i32(fracHitGlobalBrickIndexFloat.y),
            i32(fracHitGlobalBrickIndexFloat.z));
        
        let iCheckBrixelIndex: u32 = u32(hitBrixelIndex.z) * iNumBrixelsPerBrickRowSquared + u32(hitBrixelIndex.y) * iNumBrixelsPerBrickRow + u32(hitBrixelIndex.x);
        fPrevDistanceToSeed = fDistanceToSeed;

        // fetch the distance to the seed
        fDistanceToSeed = aBrixelDistances[u32(iBrixelDistanceArrayIndex) * 512u + iCheckBrixelIndex];

        // check whether to stop after hitting a valid brixel
        if(fDistanceToSeed <= fDistanceToDraw)
        {
            // store the closest brixel from the starting point
            let fLength: f32 = length(currPosition - startPosition);
            if(fLength < fClosestLengthFromStart)
            {
                closestPosition = clamp(currPosition, minGlobalBrickPosition, maxGlobalBrickPosition);
                fClosestBrixelDistanceToSeed = fDistanceToSeed;
                iClosestBrickIndex = iGlobalBrickArrayIndex;
                iClosestBrixelIndex = iCheckBrixelIndex;
                fClosestLengthFromStart = fLength;
            }

            break;

        }   // if within the draw distance

        var fDistance: f32 = fDistanceToSeed * 0.7f;       // apply scale to distance to avoid overshooting in the ray direction
        if(fDistanceToSeed > 7.0f)
        {
            fDistance = 1.0f;
        }
        currPosition += rayDirection * fDistance * fBrixelDimension;

        let fTotalDistance: f32 = length(currPosition - startPosition);
        fRes = min(fRes, (kfScatter * fDistanceToSeed * fBrixelDimension) / fTotalDistance);

    }   // for inc = 0 to num steps

    // return valid intersecting brixel, distance, position, etc.
    if(fClosestLengthFromStart != FLT_MAX && fClosestBrixelDistanceToSeed <= fDistanceToDraw)
    {
        ret.mHitPosition = closestPosition;
        ret.mfClosestLengthFromStart = fClosestLengthFromStart;
        ret.mfDistanceToSeed = fClosestBrixelDistanceToSeed;
        ret.miBrickIndex = iClosestBrickIndex;
        ret.miBrixelIndex = iClosestBrixelIndex;
    }
    else 
    {
        ret.mDebug.x = fRes * fBrixelDimension;
    }
    
    return ret;
}

/////
fn initRand(
    val0: u32, 
    val1: u32, 
    backoff: u32) -> RandomResult
{
    var retResult: RandomResult;

    var v0: u32 = val0;
    var v1: u32 = val1;
    var s0: u32 = 0u;

    for(var n: u32 = 0; n < backoff; n++)
    {
        s0 += u32(0x9e3779b9);
        v0 += ((v1 << 4) + 0xa341316c) ^ (v1 + s0) ^ ((v1 >> 5) + 0xc8013ea4);
        v1 += ((v0 << 4) + 0xad90777d) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7e95761e);
    }

    retResult.miSeed = v0;
    retResult.mfNum = 0.0f;

    return retResult;
}

/////
fn nextRand(s: u32) -> RandomResult
{
    var retResult: RandomResult;

    var sCopy: u32 = s;
    sCopy = (1664525u * sCopy + 1013904223u);
    retResult.mfNum = f32(sCopy & 0x00FFFFFF) / f32(0x01000000);
    retResult.miSeed = sCopy;

    return retResult;
}

/////
fn uniformSampling(
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    fRand0: f32,
    fRand1: f32) -> Ray
{
    let fPhi: f32 = 2.0f * PI * fRand0;
    let fCosTheta: f32 = 1.0f - fRand1;
    let fSinTheta: f32 = sqrt(1.0f - fCosTheta * fCosTheta);
    let h: vec3<f32> = vec3<f32>(
        cos(fPhi) * fSinTheta,
        sin(fPhi) * fSinTheta,
        fCosTheta);

    var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    if(abs(normal.y) > 0.999f)
    {
        up = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    let tangent: vec3<f32> = normalize(cross(up, normal));
    let binormal: vec3<f32> = normalize(cross(normal, tangent));
    let rayDirection: vec3<f32> = normalize(tangent * h.x + binormal * h.y + normal * h.z);

    var ray: Ray;
    ray.mOrigin = vec4<f32>(worldPosition, 1.0f);
    ray.mDirection = vec4<f32>(rayDirection, 1.0f);
    ray.mfT = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

    return ray;
}

/////
fn getPreviousScreenUV(
    screenUV: vec2<f32>) -> vec2<f32>
{
    var motionVector: vec2<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        screenUV).xy;
    motionVector = motionVector * 2.0f - 1.0f;
    var prevScreenUV: vec2<f32> = screenUV - motionVector;

    var worldPosition: vec3<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        screenUV
    ).xyz;
    var normal: vec3<f32> = textureSample(
        normalTexture,
        textureSampler,
        screenUV
    ).xyz;

    var fOneOverScreenWidth: f32 = 1.0f / f32(defaultUniformData.miScreenWidth);
    var fOneOverScreenHeight: f32 = 1.0f / f32(defaultUniformData.miScreenHeight);

    var fShortestWorldDistance: f32 = FLT_MAX;
    var closestScreenUV: vec2<f32> = prevScreenUV;
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            let sampleUV: vec2<f32> = prevScreenUV + vec2<f32>(
                clamp(f32(iX) * fOneOverScreenWidth, 0.0f, 1.0f),
                clamp(f32(iY) * fOneOverScreenHeight, 0.0f, 1.0f) 
            );

            var checkWorldPosition: vec3<f32> = textureSample(
                prevWorldPositionTexture,
                textureSampler,
                sampleUV
            ).xyz;
            var checkNormal: vec3<f32> = textureSample(
                prevNormalTexture,
                textureSampler,
                sampleUV
            ).xyz;
            var fNormalDP: f32 = abs(dot(checkNormal, normal));

            var worldPositionDiff: vec3<f32> = checkWorldPosition - worldPosition;
            var fLengthSquared: f32 = dot(worldPositionDiff, worldPositionDiff);
            if(fNormalDP >= 0.99f && fShortestWorldDistance > fLengthSquared)
            {
                fShortestWorldDistance = fLengthSquared;
                closestScreenUV = sampleUV;
            }
        }
    }

    return closestScreenUV;
}

///// 
fn isPrevUVOutOfBounds(inputTexCoord: vec2<f32>) -> bool
{
    var motionVector: vec4<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        inputTexCoord);
    motionVector.x = motionVector.x * 2.0f - 1.0f;
    motionVector.y = motionVector.y * 2.0f - 1.0f;
    let backProjectedScreenUV: vec2<f32> = inputTexCoord - motionVector.xy;

    return (backProjectedScreenUV.x < 0.0f || backProjectedScreenUV.x > 1.0 || backProjectedScreenUV.y < 0.0f || backProjectedScreenUV.y > 1.0f);
}

/////
fn isDisoccluded2(
    screenUV: vec2<f32>,
    prevScreenUV: vec2<f32>
) -> bool
{
    var worldPosition: vec4<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        screenUV);

    var prevWorldPosition: vec4<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        prevScreenUV);

    var normal: vec3<f32> = textureSample(
        normalTexture,
        textureSampler,
        screenUV).xyz;

    var prevNormal: vec3<f32> = textureSample(
        prevNormalTexture,
        textureSampler,
        prevScreenUV).xyz;

    var motionVector: vec4<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        screenUV);

    var prevMotionVectorAndMeshIDAndDepth: vec4<f32> = textureSample(
        prevMotionVectorTexture,
        textureSampler,
        prevScreenUV);

    let iMesh = u32(ceil(motionVector.z - 0.5f)) - 1;
    var fDepth: f32 = motionVector.w;
    var fPrevDepth: f32 = prevMotionVectorAndMeshIDAndDepth.w;
    var fCheckDepth: f32 = abs(fDepth - fPrevDepth);
    var worldPositionDiff: vec3<f32> = prevWorldPosition.xyz - worldPosition.xyz;
    var fCheckDP: f32 = abs(dot(normalize(normal.xyz), normalize(prevNormal.xyz)));
    let iPrevMesh: u32 = u32(ceil(prevMotionVectorAndMeshIDAndDepth.z - 0.5f)) - 1;
    var fCheckWorldPositionDistance: f32 = dot(worldPositionDiff, worldPositionDiff);

    return !(iMesh == iPrevMesh && fCheckDepth <= 0.01f && fCheckWorldPositionDistance <= 0.01f && fCheckDP >= 0.99f && prevWorldPosition.w > 0.0f && worldPosition.w > 0.0f);
}

/////
fn directionalShadow(
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    lightDirection: vec3<f32>,
    meshBBox0: vec3<f32>,
    meshBBox1: vec3<f32>,
    numBricks: vec3<i32>,
    screenCoord: vec2<u32>,
    randomResult: RandomResult
) -> DirectionalShadowInfo
{
    var randomResultCopy: RandomResult = randomResult;

    let textureSize: vec2<u32> = textureDimensions(blueNoiseTexture, 0);

    var ret: DirectionalShadowInfo;
    ret.mRadiance = vec3<f32>(0.0f, 0.0f, 0.0f);
    
    let fLightAngleSpread: f32 = PI / 80.0f;

    let iTileSize: u32 = 32u;
    let iNumTilesPerRow: u32 = textureSize.x / iTileSize;
    let iNumTotalTiles: u32 = iNumTilesPerRow * iNumTilesPerRow;

    let scaledWorldPosition: vec3<f32> = worldPosition * uniformData.mfPositionScale;

    let lightDirectionEndPosition: vec3<f32> = worldPosition + lightDirection;

    let iTotalTextureSize: u32 = textureSize.x * textureSize.y;

    // ray samples
    let iNumSamples: i32 = 1;
    let iCurrIndex: u32 = u32(iNumSamples * defaultUniformData.miFrame);
    var sphereTraceInfo: SphereTraceInfo;
    for(var iSample: i32 = 0; iSample < iNumSamples; iSample++)
    {
        let iTileX: u32 = (iCurrIndex + u32(iNumSamples)) % iNumTilesPerRow;
        let iTileY: u32 = ((iCurrIndex + u32(iNumSamples)) / iNumTilesPerRow) % (iNumTilesPerRow * iNumTilesPerRow);

        let iTileOffsetX: u32 = (iCurrIndex + u32(iNumSamples)) % iTileSize;
        let iTileOffsetY: u32 = ((iCurrIndex + u32(iNumSamples)) / iTileSize) % (iTileSize * iTileSize);

        let iOffsetX: u32 = iTileOffsetX + iTileX * iTileSize;
        let iOffsetY: u32 = iTileOffsetY + iTileY * iTileSize; 

        let sampleScreenCoord = vec2<u32>(
            (screenCoord.x + iOffsetX) % textureSize.x,
            (screenCoord.y + iOffsetY) % textureSize.y
        );
        let sampleUV: vec2<f32> = vec2<f32>(
            clamp(f32(sampleScreenCoord.x) / f32(textureSize.x), 0.0f, 1.0f),
            clamp(f32(sampleScreenCoord.y) / f32(textureSize.y), 0.0f, 1.0f) 
        );

        let blueNoise: vec3<f32> = textureSample(
            blueNoiseTexture,
            textureSampler,
            sampleUV
        ).xyz;

        // simulate light spread
        let randLightEndPosition: vec3<f32> = 
            lightDirectionEndPosition + 
            (blueNoise.xyz - vec3<f32>(0.5f, 0.5f, 0.5f)) *
            fLightAngleSpread; 
        let sampleLightDirection: vec3<f32> = lightDirection; // normalize(randLightEndPosition - worldPosition.xyz);

        sphereTraceInfo.miBrickIndex = UINT32_MAX;
        let tMinMax: vec2<f32> = rayBoxIntersect2(
            scaledWorldPosition,
            sampleLightDirection,
            meshBBox0,
            meshBBox1
        );
        let bIntersect: bool = (tMinMax.x <= tMinMax.y);
        let bInsideBBox: bool = (
            scaledWorldPosition.x >= meshBBox0.x && scaledWorldPosition.x <= meshBBox1.x &&
            scaledWorldPosition.y >= meshBBox0.y && scaledWorldPosition.y <= meshBBox1.y &&
            scaledWorldPosition.z >= meshBBox0.z && scaledWorldPosition.z <= meshBBox1.z);

        // sphere trace against sdf if either inside the bounding box or intersected with ray
        if(bIntersect || bInsideBBox)
        {
            // only apply ray direction position on pixels outside of the bounding box
            var startPosition: vec3<f32> = scaledWorldPosition; 
            if(bIntersect && !bInsideBBox)
            {
                startPosition += sampleLightDirection * tMinMax.x;
            }

            //startPosition += sampleLightDirection * uniformData.mfBrickDimension * 0.425f;
            startPosition += sampleLightDirection * uniformData.mfBrixelDimension * 2.0f;
            sphereTraceInfo = sphereTraceSDFDistance(
                startPosition,
                sampleLightDirection, 
                meshBBox0,
                numBricks,
                uniformData.mfBrickDimension,
                uniformData.mfBrixelDimension,
                8u,
                80.0f,
                0.0f,
                false
            );
        }

        // didn't hit anything, add radiance
        let fHitMesh: f32 = f32(sphereTraceInfo.miBrickIndex == UINT32_MAX);
        ret.mRadiance.x += fHitMesh; 

        if(sphereTraceInfo.miBrickIndex == UINT32_MAX && sphereTraceInfo.mDebug.x > 0.0f)
        {
            ret.mRadiance.x = clamp(sphereTraceInfo.mDebug.x, 0.0f, 1.0f);
        }

        ret.mRadiance.y += 1.0f;
    }

    let fRadiance: f32 = ret.mRadiance.x / ret.mRadiance.y;
    
    ret.mRadiance = vec3<f32>(fRadiance, fRadiance, fRadiance);
    ret.mRandomResult = randomResultCopy;

    return ret;
}

/////
fn cosineHemisphereSampling(
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    fRand0: f32,
    fRand1: f32) -> Ray
{
    let fPhi: f32 = 2.0f * PI * fRand0;
    let fTheta: f32 = acos(sqrt(fRand1));
    
    let h: vec3<f32> = vec3<f32>(
        sin(fTheta) * cos(fPhi),
        sin(fTheta) * sin(fPhi),
        cos(fTheta));

    var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    if(abs(normal.y) > 0.999f)
    {
        up = vec3<f32>(1.0f, 0.0f, 0.0f);
    }
    let tangent: vec3<f32> = normalize(cross(up, normal));
    let binormal: vec3<f32> = normalize(cross(normal, tangent));
    let rayDirection: vec3<f32> = normalize(tangent * h.x + binormal * h.y + normal * h.z);
    
    var ray: Ray;
    ray.mOrigin = vec4<f32>(worldPosition, 1.0f);
    ray.mDirection = vec4<f32>(rayDirection, 1.0f);
    ray.mfT = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

    return ray;
}

/////
fn rayBoxIntersect2(
    rayPosition: vec3<f32>,
    rayDirection: vec3<f32>,
    bboxMin: vec3<f32>,
    bboxMax: vec3<f32>) -> vec2<f32>
{
    let t1: vec3<f32> = (bboxMin - rayPosition) / rayDirection; 
    let t2: vec3<f32> = (bboxMax - rayPosition) / rayDirection; 

    var tmin: f32 = min(t1.x, t2.x);
    var tmax: f32 = max(t1.x, t2.x);

    tmin = max(tmin, min(t1.y, t2.y));
    tmax = min(tmax, max(t1.y, t2.y));

    tmin = max(tmin, min(t1.z, t2.z));
    tmax = min(tmax, max(t1.z, t2.z));

    return vec2<f32>(tmin, tmax);
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}