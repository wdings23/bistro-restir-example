const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const PROBE_IMAGE_SIZE: u32 = 8u;
const VALIDATION_STEP: u32 = 16u;

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

struct BrixelRayHit
{
    miBrickIndex: u32,
    miBrixelIndex: u32
};

struct ReservoirResult
{
    mReservoir: vec4<f32>,
    mbExchanged: bool
};

struct RandomResult 
{
    mfNum: f32,
    miSeed: u32,
};

struct IntersectBVHResult
{
    mHitPosition: vec3<f32>,
    mHitNormal: vec3<f32>,
    miHitTriangle: u32,
    mBarycentricCoordinate: vec3<f32>,
};

struct RayTriangleIntersectionResult
{
    mIntersectPosition: vec3<f32>,
    mIntersectNormal: vec3<f32>,
    mBarycentricCoordinate: vec3<f32>,
};

struct BVHNode2
{
    mMinBound: vec4<f32>,
    mMaxBound: vec4<f32>,
    mCentroid: vec4<f32>,
    
    miChildren0: u32,
    miChildren1: u32,
    miPrimitiveID: u32,
    miMeshID: u32,
};


@group(0) @binding(0)
var<storage, read_write> aBrixelRadiance: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read_write> aBrixelReservoir: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> aBrixelDistances: array<f32>;

@group(0) @binding(3)
var<storage, read> aBrixelBarycentricCoordinates: array<vec3<f32>>;

@group(0) @binding(4)
var<storage, read> aiBrickToBrixelMapping: array<i32>;

@group(0) @binding(5)
var<storage, read> aHitBrickQueue: array<BrixelRayHit>;

@group(0) @binding(6)
var<storage, read> aiQueueCounter: array<atomic<u32>>;

@group(0) @binding(7)
var<storage, read> aTrianglePositions: array<vec4<f32>>;

@group(0) @binding(8)
var<storage, read> aBBox: array<i32>;

@group(0) @binding(9)
var skyTexture: texture_2d<f32>;

@group(0) @binding(10)
var textureSampler: sampler;

struct UniformData
{
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    mfRand0: f32,
    mfRand1: f32,

    miScreenWidth: u32,
    miScreenHeight: u32,

    mfPositionScale: f32,

    miFrameIndex: u32,
};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> aiTriangleIndices: array<u32>;

@group(1) @binding(2)
var blueNoiseTexture: texture_2d<f32>;

@group(1) @binding(3)
var<storage, read> aBVHNodes: array<BVHNode2>;

const iNumThreads = 256u;

/////
@compute
@workgroup_size(iNumThreads)
fn cs_main(
    @builtin(num_workgroups) numWorkGroups: vec3<u32>,
    @builtin(local_invocation_index) iLocalThreadIndex: u32,
    @builtin(workgroup_id) workGroup: vec3<u32>)
{
    let iNumTotalThreads: u32 = numWorkGroups.x * iNumThreads;
    let iNumBrixels: u32 = aiQueueCounter[0];

    var iCurrThreadIndex: u32 = workGroup.x * iNumThreads + iLocalThreadIndex;
    
    var randomResult: RandomResult = initRand(
        u32(f32(iLocalThreadIndex) * 100.0f + f32(iCurrThreadIndex) * 200.0f + uniformData.mfRand0),
        u32(f32(workGroup.x) * 10.0f + f32(iLocalThreadIndex) * 20.0f + uniformData.mfRand0),
        10u);
    
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

    for(var iQueueBrixelIndex: u32 = iCurrThreadIndex; iQueueBrixelIndex < iNumBrixels; iQueueBrixelIndex += iNumTotalThreads)
    {
        let brixelInfo: BrixelRayHit = aHitBrickQueue[iQueueBrixelIndex];
        let iBrickMappingIndex: u32 = aiBrickToBrixelMapping[brixelInfo.miBrickIndex];
        let iBrixelArrayIndex: u32 = iBrickMappingIndex * 512u + brixelInfo.miBrixelIndex;

        var barycentricCoord: vec3<f32> = aBrixelBarycentricCoordinates[iBrixelArrayIndex];
        let iMesh: u32 = u32(barycentricCoord.x - fract(barycentricCoord.x)) - 1u;
        let iTriangle: u32 = u32(barycentricCoord.y - fract(barycentricCoord.y)) - 1u;
       
        // get starting ray position using barycentric coordinate
        let iV0: u32 = aiTriangleIndices[iTriangle * 3u];
        let iV1: u32 = aiTriangleIndices[iTriangle * 3u + 1u];
        let iV2: u32 = aiTriangleIndices[iTriangle * 3u + 2u];
        let v0: vec4<f32> = aTrianglePositions[iV0];
        let v1: vec4<f32> = aTrianglePositions[iV1];
        let v2: vec4<f32> = aTrianglePositions[iV2];
        let meshPosition: vec3<f32> = 
            v0.xyz * fract(barycentricCoord.x) + 
            v1.xyz * fract(barycentricCoord.y) + 
            v2.xyz * fract(barycentricCoord.z);
        
        // compute the normal for ray direction
        let diff0: vec3<f32> = normalize(v1.xyz - v0.xyz);
        let diff1: vec3<f32> = normalize(v2.xyz - v0.xyz);
        let normal: vec3<f32> = normalize(cross(diff0, diff1));

        randomResult = nextRand(randomResult.miSeed);
        let fRand0: f32 = randomResult.mfNum;
        randomResult = nextRand(randomResult.miSeed);
        let fRand1: f32 = randomResult.mfNum;
        randomResult = nextRand(randomResult.miSeed);
        let fRand2: f32 = randomResult.mfNum;

        // ray direction sample
        var ray: Ray = uniformSampling(
            meshPosition * uniformData.mfPositionScale, 
            normal, 
            fRand0, 
            fRand1);

        let intersectionInfo: IntersectBVHResult = intersectBVH4(ray, 0u);

        // radiance and luminance from sky light if not hit anything along the ray
        var radiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        if(intersectionInfo.miHitTriangle == UINT32_MAX)
        {
            let skyUV: vec2<f32> = octahedronMap2(ray.mDirection.xyz);
            radiance = textureSampleLevel(
                skyTexture,
                textureSampler,
                skyUV,
                0.0f).xyz;
        }
        let fLuminance: f32 = computeLuminance(radiance);

        // update reservoir, using luminance as PHat
        let fMaxReservoirSize: f32 = 12.0f;
        var updateResult: ReservoirResult = updateReservoir(
            aBrixelReservoir[iBrixelArrayIndex],
            fLuminance,
            1.0f,
            fRand2
        );
        if(updateResult.mReservoir.z > fMaxReservoirSize)
        {
            let fPct: f32 = fMaxReservoirSize / updateResult.mReservoir.z;
            updateResult.mReservoir.x *= fPct;
            updateResult.mReservoir.z = fMaxReservoirSize;
        }

        aBrixelReservoir[iBrixelArrayIndex] = updateResult.mReservoir;
        aBrixelReservoir[iBrixelArrayIndex].w = clamp(updateResult.mReservoir.x / max(updateResult.mReservoir.z * updateResult.mReservoir.y, 0.001f), 0.1f, 1.0f);
    
        if(updateResult.mbExchanged)
        {
            aBrixelRadiance[iBrixelArrayIndex] = vec4<f32>(
                radiance.x,
                radiance.y,
                radiance.z,
                1.0f);
        }
    }
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
            // compute the smallest increment along the ray direction to get to the next brick
            let startBrickIndexOrig: vec3<f32> = (currPosition - minGlobalBrickPosition) * fOneOverHitBrickSize;
            let startBrickIndex: vec3<f32> = floor(startBrickIndexOrig);
            var nextBrickIndexOnRayDirection: vec3<f32> = vec3<f32>(
                startBrickIndex.x + 1.0f,
                startBrickIndex.y + 1.0f,
                startBrickIndex.z + 1.0f);
            if(rayDirection.x < 0.0f)
            {
                nextBrickIndexOnRayDirection.x = startBrickIndex.x - 1.0f;
            }
            if(rayDirection.y < 0.0f)
            {
                nextBrickIndexOnRayDirection.x = startBrickIndex.y - 1.0f;
            }
            if(rayDirection.z < 0.0f)
            {
                nextBrickIndexOnRayDirection.x = startBrickIndex.z - 1.0f;
            }
            let nextPotentialBrickPosition: vec3<f32> = minGlobalBrickPosition + nextBrickIndexOnRayDirection * fBrickDimension;
            let positionDiff: vec3<f32> = nextPotentialBrickPosition - currPosition;
            let positionDiffOverRayDirection: vec3<f32> = positionDiff * oneOverRayDirection;
            let fSmallestInc: f32 = min(positionDiffOverRayDirection.x, min(positionDiffOverRayDirection.y, positionDiffOverRayDirection.z));

            let nextPosition: vec3<f32> = currPosition + rayDirection * fSmallestInc;
            currPosition = nextPosition;

            continue;
        }

        var iBrixelDistanceArrayIndex: u32 = aiBrickToBrixelMapping[iGlobalBrickArrayIndex];
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
        fDistanceToSeed = aBrixelDistances[iBrixelDistanceArrayIndex * 512u + iCheckBrixelIndex];

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

        var fDistance: f32 = fDistanceToSeed;
        if(fDistanceToSeed > fBrickDimension)
        {
            fDistance = 1.0f;
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
fn rayBoxIntersect2(
    rayPosition: vec3<f32>,
    rayDirection: vec3<f32>,
    bboxMin: vec3<f32>,
    bboxMax: vec3<f32>) -> bool
{
    let t1: vec3<f32> = (bboxMin - rayPosition) / rayDirection; 
    let t2: vec3<f32> = (bboxMax - rayPosition) / rayDirection; 

    var tmin: f32 = min(t1.x, t2.x);
    var tmax: f32 = max(t1.x, t2.x);

    tmin = max(tmin, min(t1.y, t2.y));
    tmax = min(tmax, max(t1.y, t2.y));

    tmin = max(tmin, min(t1.z, t2.z));
    tmax = min(tmax, max(t1.z, t2.z));

    return tmax >= tmin;
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
}

/////
fn updateReservoir(
    reservoir: vec4<f32>,
    fPHat: f32,
    fM: f32,
    fRand: f32) -> ReservoirResult
{
    var ret: ReservoirResult;
    ret.mReservoir = reservoir;
    ret.mbExchanged = false;

    ret.mReservoir.x += fPHat;
    ret.mReservoir.z += fM;
    var fWeightPct: f32 = fPHat / ret.mReservoir.x;

    if(fRand < fWeightPct || reservoir.z <= 0.0f)
    {
        ret.mReservoir.y = fPHat;
        ret.mbExchanged = true;
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
fn octahedronMap2(direction: vec3<f32>) -> vec2<f32>
{
    let fDP: f32 = dot(vec3<f32>(1.0f, 1.0f, 1.0f), abs(direction));
    let newDirection: vec3<f32> = vec3<f32>(direction.x, direction.z, direction.y) / fDP;

    var ret: vec2<f32> =
        vec2<f32>(
            (1.0f - abs(newDirection.z)) * sign(newDirection.x),
            (1.0f - abs(newDirection.x)) * sign(newDirection.z));
       
    if(newDirection.y < 0.0f)
    {
        ret = vec2<f32>(
            newDirection.x, 
            newDirection.z);
    }

    ret = ret * 0.5f + vec2<f32>(0.5f, 0.5f);
    ret.y = 1.0f - ret.y;

    return ret;
}

/////
fn intersectBVH4(
    ray: Ray,
    iRootNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 128>;
    aiStack[iStackTop] = iRootNodeIndex;

    ret.mHitPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    ret.miHitTriangle = UINT32_MAX;
    var fClosestDistance: f32 = FLT_MAX;

    for(var iStep: u32 = 0u; iStep < 10000u; iStep++)
    {
        if(iStackTop < 0)
        {
            break;
        }

        let iNodeIndex: u32 = aiStack[iStackTop];
        iStackTop -= 1;

        if(aBVHNodes[iNodeIndex].miPrimitiveID != UINT32_MAX)
        {
            let intersectionInfo: RayTriangleIntersectionResult = intersectTri4(
                ray,
                aBVHNodes[iNodeIndex].miPrimitiveID);

            if(abs(intersectionInfo.mIntersectPosition.x) < 10000.0f)
            {
                let fDistanceToEye: f32 = length(intersectionInfo.mIntersectPosition.xyz - ray.mOrigin.xyz);
                //if(fDistanceToEye < fClosestDistance)
                {
                    
                    //fClosestDistance = fDistanceToEye;
                    ret.mHitPosition = intersectionInfo.mIntersectPosition.xyz;
                    ret.mHitNormal = intersectionInfo.mIntersectNormal.xyz;
                    ret.miHitTriangle = aBVHNodes[iNodeIndex].miPrimitiveID;
                    ret.mBarycentricCoordinate = intersectionInfo.mBarycentricCoordinate;

                    break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aBVHNodes[iNodeIndex].mMinBound.xyz,
                aBVHNodes[iNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aBVHNodes[iNodeIndex].miChildren0;
                iStackTop += 1;
                aiStack[iStackTop] = aBVHNodes[iNodeIndex].miChildren1;
            }
        }
    }

    return ret;
}

/////
fn intersectTri4(
    ray: Ray,
    iTriangleIndex: u32) -> RayTriangleIntersectionResult
{
    let pos0: vec4<f32> = aTrianglePositions[aiTriangleIndices[iTriangleIndex * 3]];
    let pos1: vec4<f32> = aTrianglePositions[aiTriangleIndices[iTriangleIndex * 3 + 1]];
    let pos2: vec4<f32> = aTrianglePositions[aiTriangleIndices[iTriangleIndex * 3 + 2]];

    var iIntersected: u32 = 0;
    var fT: f32 = FLT_MAX;
    let intersectionInfo: RayTriangleIntersectionResult = rayTriangleIntersection(
        ray.mOrigin.xyz,
        ray.mOrigin.xyz + ray.mDirection.xyz * 1000.0f,
        pos0.xyz,
        pos1.xyz,
        pos2.xyz);
    
    return intersectionInfo;
}

/////
fn rayTriangleIntersection(
    rayPt0: vec3<f32>, 
    rayPt1: vec3<f32>, 
    triPt0: vec3<f32>, 
    triPt1: vec3<f32>, 
    triPt2: vec3<f32>) -> RayTriangleIntersectionResult
{
    var ret: RayTriangleIntersectionResult;

    let v0: vec3<f32> = normalize(triPt1 - triPt0);
    let v1: vec3<f32> = normalize(triPt2 - triPt0);
    let cp: vec3<f32> = cross(v0, v1);

    let triNormal: vec3<f32> = normalize(cp);
    let fPlaneDistance: f32 = -dot(triPt0, triNormal);

    let fT: f32 = rayPlaneIntersection(
        rayPt0, 
        rayPt1, 
        triNormal, 
        fPlaneDistance);
    if(fT < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }

    let collisionPt: vec3<f32> = rayPt0 + (rayPt1 - rayPt0) * fT;
    
    let edge0: vec3<f32> = normalize(triPt1 - triPt0);
    let edge1: vec3<f32> = normalize(triPt2 - triPt0);
    let edge2: vec3<f32> = normalize(triPt0 - triPt2);

    // edge 0
    var C: vec3<f32> = cross(edge0, normalize(collisionPt - triPt0));
    if(dot(triNormal, C) < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }

    // edge 1
    C = cross(edge1, normalize(collisionPt - triPt1));
    if(dot(triNormal, C) < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }

    // edge 2
    C = cross(edge2, normalize(collisionPt - triPt2));
    if(dot(triNormal, C) < 0.0f)
    {
        ret.mIntersectPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
        return ret;
    }
    
    ret.mBarycentricCoordinate = barycentric(collisionPt, triPt0, triPt1, triPt2);
    ret.mIntersectPosition = (triPt0 * ret.mBarycentricCoordinate.x + triPt1 * ret.mBarycentricCoordinate.y + triPt2 * ret.mBarycentricCoordinate.z);
    ret.mIntersectNormal = triNormal.xyz;

    return ret;
}

/////
fn rayPlaneIntersection(
    pt0: vec3<f32>,
    pt1: vec3<f32>,
    planeNormal: vec3<f32>,
    fPlaneDistance: f32) -> f32
{
    var fRet: f32 = FLT_MAX;
    let v: vec3<f32> = pt1 - pt0;

    let fDenom: f32 = dot(v, planeNormal);
    fRet = -(dot(pt0, planeNormal) + fPlaneDistance) / (fDenom + 1.0e-5f);

    return fRet;
}

/////
fn barycentric(
    p: vec3<f32>, 
    a: vec3<f32>, 
    b: vec3<f32>, 
    c: vec3<f32>) -> vec3<f32>
{
    let v0: vec3<f32> = b - a;
    let v1: vec3<f32> = c - a;
    let v2: vec3<f32> = p - a;
    let fD00: f32 = dot(v0, v0);
    let fD01: f32 = dot(v0, v1);
    let fD11: f32 = dot(v1, v1);
    let fD20: f32 = dot(v2, v0);
    let fD21: f32 = dot(v2, v1);
    let fOneOverDenom: f32 = 1.0f / (fD00 * fD11 - fD01 * fD01);
    let fV: f32 = (fD11 * fD20 - fD01 * fD21) * fOneOverDenom;
    let fW: f32 = (fD00 * fD21 - fD01 * fD20) * fOneOverDenom;
    let fU: f32 = 1.0f - fV - fW;

    return vec3<f32>(fU, fV, fW);
}

/////
fn rayBoxIntersect(
    rayPosition: vec3<f32>,
    rayDir: vec3<f32>,
    bboxMin: vec3<f32>,
    bboxMax: vec3<f32>) -> bool
{
    //let oneOverRay: vec3<f32> = 1.0f / rayDir.xyz;
    let tMin: vec3<f32> = (bboxMin - rayPosition) / rayDir.xyz;
    let tMax: vec3<f32> = (bboxMax - rayPosition) / rayDir.xyz;

    var fTMin: f32 = min(tMin.x, tMax.x);
    var fTMax: f32 = max(tMin.x, tMax.x);

    fTMin = max(fTMin, min(tMin.y, tMax.y));
    fTMax = min(fTMax, max(tMin.y, tMax.y));

    fTMin = max(fTMin, min(tMin.z, tMax.z));
    fTMax = min(fTMax, max(tMin.z, tMax.z));

    return fTMax >= fTMin;
}