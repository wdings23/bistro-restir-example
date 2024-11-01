const FLT_MAX: f32 = 1000000.0f;
const UINT32_MAX: u32 = 0xffffffff;
const PI: f32 = 3.14159f;

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

struct BrickInfo
{
    mPosition: vec3<f32>,
    miBrixelIndex: u32,
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
    @location(0) indirectDiffuseRadiance : vec4<f32>,
};

@group(0) @binding(0)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var rayDirectionTexture: texture_2d<f32>;

@group(0) @binding(2)
var<storage, read> aBrixelDistances: array<f32>;

@group(0) @binding(3)
var<storage, read> aBrixelRadiance: array<vec4<f32>>;

@group(0) @binding(4)
var<storage, read> aBBox: array<i32>;

@group(0) @binding(5)
var<storage, read> aiBrickToBrixelMapping: array<i32>;

@group(0) @binding(6)
var hitBrickAndBrixelTexture: texture_2d<f32>;

@group(0) @binding(7)
var<storage, read> aBricks: array<BrickInfo>;

@group(0) @binding(8)
var textureSampler: sampler;

struct UniformData
{
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    mfPositionScale: f32,
};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var blueNoiseTexture: texture_2d<f32>;

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

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        in.texCoord
    );

    let rayDirection: vec3<f32> = textureSample(
        rayDirectionTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    let hitBrickAndBrixels: vec4<f32> = textureSample(
        hitBrickAndBrixelTexture,
        textureSampler,
        in.texCoord
    );

    out.indirectDiffuseRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);

    let iHitBrick: u32 = u32(ceil(hitBrickAndBrixels.x - 0.5f));
    let iHitBrixel: u32 = u32(ceil(hitBrickAndBrixels.y - 0.5f));
    if(iHitBrick > 0u)
    {
        let fReflectivity: f32 = 0.8f;

        let brickPosition: vec3<f32> = aBricks[iHitBrick].mPosition;
        let diff: vec3<f32> = brickPosition - worldPosition.xyz;
        let fAttenuation: f32 = max(dot(diff, diff), 1.0f) * fReflectivity; 

        let iBrickMappingIndex: i32 = aiBrickToBrixelMapping[iHitBrick];
        let iBrixelArrayIndex: u32 = u32(iBrickMappingIndex) * 512u + iHitBrixel;
        out.indirectDiffuseRadiance = vec4<f32>(
            aBrixelRadiance[iBrixelArrayIndex].x / fAttenuation,
            aBrixelRadiance[iBrixelArrayIndex].y / fAttenuation,
            aBrixelRadiance[iBrixelArrayIndex].z / fAttenuation,
            f32(iHitBrick));
    }

/*
    // sphere trace against sdf
    var sphereTraceInfo: SphereTraceInfo;
    sphereTraceInfo.miBrickIndex = UINT32_MAX;
    let rayBoxT: vec2<f32> = rayBoxIntersect2(
        worldPosition.xyz,
        rayDirection,
        meshBBox0,
        meshBBox1);
    let intersected: bool = (rayBoxT.x <= rayBoxT.y);
    
    out.indirectDiffuseRadiance = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0);
    //if(intersected)
    {
        sphereTraceInfo = sphereTraceSDFDistance(
            worldPosition.xyz * uniformData.mfPositionScale,
            rayDirection,
            meshBBox0,
            numBricks,
            uniformData.mfBrickDimension,
            uniformData.mfBrixelDimension,
            8u,
            100.0f,
            0.0f,
            true
        );

        if(sphereTraceInfo.miBrickIndex != UINT32_MAX)
        {
            let iBrickMappingIndex: i32 = aiBrickToBrixelMapping[sphereTraceInfo.miBrickIndex];
            let iBrixelArrayIndex: u32 = u32(iBrickMappingIndex) * 512u + sphereTraceInfo.miBrixelIndex;
            out.indirectDiffuseRadiance = vec4<f32>(
                aBrixelRadiance[iBrixelArrayIndex].x,
                aBrixelRadiance[iBrixelArrayIndex].y,
                aBrixelRadiance[iBrixelArrayIndex].z,
                1.0f);
        }
    }
*/

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
    var currPosition: vec3<f32> = startPosition + rayDirection * fBrixelDimension * 4.0f;
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

            currPosition = currPosition + rayDirection * tMinMax.x;
            continue;
        }

        var iBrixelDistanceArrayIndex: i32 = aiBrickToBrixelMapping[iGlobalBrickArrayIndex];
        if(iBrixelDistanceArrayIndex <= 0)
        {
            // didn't hit any valid brick, continue in ray direction

            let fDistance: f32 = fBrixelDimension; //(hitMeshBrick.mfShortestDistanceToOthers < fBrixelDimension) ? fBrixelDimension : hitMeshBrick.mfShortestDistanceToOthers;
            currPosition += rayDirection * fDistance;

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

        var fDistance: f32 = floor(fDistanceToSeed);
        if(fDistanceToSeed > 4.0f)
        {
            fDistance = 4.0f;
        }
        currPosition += rayDirection * fDistance * fBrixelDimension;

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

    return ret;
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