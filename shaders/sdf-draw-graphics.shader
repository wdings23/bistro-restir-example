const FLT_MAX: f32 = 1000000.0f;
const UINT32_MAX: u32 = 0xffffffff;
const PI: f32 = 3.14159f;

const lightPosition0: vec3<f32> = vec3<f32>(10.0f, 0.0f, 0.0f);
const lightRadiance0: vec3<f32> = vec3<f32>(1.0f, 0.0f, 0.0f);

const lightPosition1: vec3<f32> = vec3<f32>(0.0f, 10.0f, 10.0f);
const lightRadiance1: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);

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
    @location(0) output : vec4<f32>,
    @location(1) radiance: vec4<f32>,
    @location(2) position: vec4<f32>,
    @location(3) volumetric: vec4<f32>,
    @location(4) debug: vec4<f32>,
};

struct AxisInfo
{
    mTangent: vec3<f32>,
    mBinormal: vec3<f32>, 
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
var<storage, read> aBrixelDistances: array<f32>;

@group(0) @binding(1)
var<storage, read> aBrixelRadiance: array<vec4<f32>>;

@group(0) @binding(2)
var<storage, read> aiBrickToBrixelMapping: array<i32>;

@group(0) @binding(3)
var<storage, read> aBBox: array<i32>;

@group(0) @binding(4)
var<storage, read> aBricks: array<BrickInfo>;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
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
    out.output = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);

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

    let iScreenX: u32 = u32(in.texCoord.x * f32(defaultUniformData.miScreenWidth));
    let iScreenY: u32 = u32(in.texCoord.y * f32(defaultUniformData.miScreenHeight));
    let screenCoord: vec2<u32> = vec2<u32>(iScreenX, iScreenY);

    var cameraLookDir: vec3<f32> = defaultUniformData.mCameraLookDir.xyz;
    let axisInfo: AxisInfo = getAxis(cameraLookDir);

    let uv: vec2<f32> = vec2<f32>(
        f32(iScreenX) / f32(defaultUniformData.miScreenWidth),
        f32(iScreenY) / f32(defaultUniformData.miScreenHeight));

    let fPctX: f32 = uv.x * 2.0f - 1.0f;
    let fPctY: f32 = (uv.y * 2.0f - 1.0f) * -1.0f;

    // re-orient ray direction to camera look at
    let rayDirection: vec3<f32> = normalize(vec3<f32>(fPctX, fPctY, 2.0f));
    let h: vec3<f32> = axisInfo.mTangent * rayDirection.x + axisInfo.mBinormal * rayDirection.y + cameraLookDir * rayDirection.z;

    let primaryHitPosition: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    let fHitDistanceFromSeed: f32 = FLT_MAX;
    let fClosestFromStart: f32 = FLT_MAX;
    let iHitMesh: u32 = UINT32_MAX;
    let iHitBrickIndex: u32 = UINT32_MAX;
    let iHitBrixelIndex: u32 = UINT32_MAX;

    let scaledCameraPosition: vec3<f32> = defaultUniformData.mCameraPosition.xyz * uniformData.mfPositionScale;
    let tMinMax: vec2<f32> = rayBoxIntersect2(
        scaledCameraPosition,
        h,
        meshBBox0,
        meshBBox1);
    let bIntersect: bool = (tMinMax.x <= tMinMax.y);
    let bInsideBBox: bool = (
        scaledCameraPosition.x >= meshBBox0.x && scaledCameraPosition.x <= meshBBox1.x &&
        scaledCameraPosition.y >= meshBBox0.y && scaledCameraPosition.y <= meshBBox1.y &&
        scaledCameraPosition.z >= meshBBox0.z && scaledCameraPosition.z <= meshBBox1.z);

    if(bIntersect || bInsideBBox)
    {
        let fStartBrixelIntersectionOffset: f32 = 2.0f;
        var startingPosition: vec3<f32> = scaledCameraPosition + h * uniformData.mfBrixelDimension * fStartBrixelIntersectionOffset;
        if(bIntersect && !bInsideBBox)
        {
            startingPosition = scaledCameraPosition + h * tMinMax.x + h * uniformData.mfBrixelDimension * fStartBrixelIntersectionOffset;
        }
        
        var sphereTraceInfo: SphereTraceInfo;
        sphereTraceInfo.miBrickIndex = UINT32_MAX;
        sphereTraceInfo = sphereTraceSDFDistance(
            startingPosition,
            h,
            meshBBox0,
            numBricks,
            uniformData.mfBrickDimension,
            uniformData.mfBrixelDimension,
            8u,
            1000.0f,
            1.0f,
            false
        );

        let volumetricColor: vec3<f32> = volumetricTraceSDFDistance(
            startingPosition,
            h,
            meshBBox0,
            numBricks,
            uniformData.mfBrickDimension,
            uniformData.mfBrixelDimension,
            8u,
            50.0f
        );

        out.volumetric = vec4<f32>(volumetricColor.xyz, 1.0f);

        if(sphereTraceInfo.miBrickIndex != UINT32_MAX)
        {
            let hitPositionDiff: vec3<f32> = sphereTraceInfo.mHitPosition - startingPosition;
            let fDistance: f32 = length(hitPositionDiff);
            let fMaxDistance = length(startingPosition + h * tMinMax.y); 

            let iBrixeDistanceArraylIndex: i32 = aiBrickToBrixelMapping[sphereTraceInfo.miBrickIndex];
            let fBrixelDistance: f32 = aBrixelDistances[u32(iBrixeDistanceArraylIndex) * 512u + sphereTraceInfo.miBrixelIndex];
            out.output = vec4<f32>(
                1.0f - (fDistance / fMaxDistance), 
                1.0f - (fDistance / fMaxDistance), 
                1.0f - (fDistance / fMaxDistance), 
                f32(sphereTraceInfo.miBrickIndex));

            let brixelRadiance: vec4<f32> = aBrixelRadiance[u32(iBrixeDistanceArraylIndex) * 512u + sphereTraceInfo.miBrixelIndex];
            out.radiance = vec4<f32>(
                brixelRadiance.x,
                brixelRadiance.y,
                brixelRadiance.z,
                brixelRadiance.w);

            out.position = vec4<f32>(
                aBricks[sphereTraceInfo.miBrickIndex].mPosition.x,
                aBricks[sphereTraceInfo.miBrickIndex].mPosition.y,
                aBricks[sphereTraceInfo.miBrickIndex].mPosition.z,
                1.0);
            
            out.debug = vec4<f32>(
                f32(sphereTraceInfo.miBrickIndex),
                f32(iBrixeDistanceArraylIndex),
                f32(iBrixeDistanceArraylIndex * 512),
                0.0f
            );
        }
    }

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
fn volumetricTraceSDFDistance(
    startPosition: vec3<f32>,
    rayDirection: vec3<f32>,
    minGlobalBrickPosition: vec3<f32>,
    numGlobalBricks: vec3<i32>,
    fBrickDimension: f32,
    fBrixelDimension: f32,
    iNumBrixelsPerBrickRow: u32,
    fNumSteps: f32) -> vec3<f32>
{
    var ret: SphereTraceInfo;
    ret.mHitPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    ret.miBrickIndex = UINT32_MAX;
    ret.miBrixelIndex = UINT32_MAX;

    let iNumBrixelsPerBrickRowSquared: u32 = iNumBrixelsPerBrickRow * iNumBrixelsPerBrickRow;
    let fOneOverHitBrickSize: f32 = 1.0f / fBrickDimension;
    
    // step through the mesh brixels along the ray
    var fPrevDistanceToSeed: f32 = FLT_MAX;
    var fDistanceToSeed: f32 = FLT_MAX;
    var currPosition: vec3<f32> = startPosition + rayDirection * fBrixelDimension * 4.0f;
    var lastPosition: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);

    let fNumBrixelsPerBrickRow: f32 = f32(iNumBrixelsPerBrickRow);
    let maxGlobalBrickPosition: vec3<f32> = minGlobalBrickPosition + vec3<f32>(f32(numGlobalBricks.x), f32(numGlobalBricks.y), f32(numGlobalBricks.z)) * fBrickDimension;
    
    let oneOverRayDirection: vec3<f32> = 1.0f / rayDirection;

    var fTotalDistance: f32 = 0.0f;
    var fOpacity: f32 = 1.0f;
    var volumetricColor: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    let volumeAlbedo: vec3<f32> = vec3<f32>(0.4f, 0.4f, 0.4f);
    let kfAbsorption: f32 = 0.1f;

    let lightPosition: vec3<f32> = vec3<f32>(0.0f, 10.0f, 10.0f);
    let lightRadiance: vec3<f32> = vec3<f32>(0.8f, 0.5f, 0.0f) * 100.0f;


    let iNumBrickXY: i32 = numGlobalBricks.x * numGlobalBricks.y;
    let iNumBrickX: i32 = numGlobalBricks.x;
    for(var fInc: f32 = 0.0f; fInc < fNumSteps; fInc += 1.0f)
    {
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
            currPosition += rayDirection * fBrixelDimension;
            continue;
        }

        // brixel index from the fractional part of the position difference
        let fracHitGlobalBrickIndexFloat: vec3<f32> = fract(positionDiff) * fNumBrixelsPerBrickRow;
        let hitBrixelIndex: vec3<i32> = vec3<i32>(
            i32(fracHitGlobalBrickIndexFloat.x),
            i32(fracHitGlobalBrickIndexFloat.y),
            i32(fracHitGlobalBrickIndexFloat.z));

        let fPrevOpacity: f32 = fOpacity;

        let iCheckBrixelIndex: u32 = u32(hitBrixelIndex.z) * iNumBrixelsPerBrickRowSquared + u32(hitBrixelIndex.y) * iNumBrixelsPerBrickRow + u32(hitBrixelIndex.x);
        fPrevDistanceToSeed = fDistanceToSeed;

        // fetch the distance to the seed
        let fMaxDistance: f32 = 3.0f;
        fDistanceToSeed = aBrixelDistances[u32(iBrixelDistanceArrayIndex) * 512u + iCheckBrixelIndex];
        let fDistance: f32 = clamp(fDistanceToSeed, 0.0f, fMaxDistance);
        //fOpacity *= beerLambert(kfAbsorption * (fNumBrixelsPerBrickRow - fDistance) / fNumBrixelsPerBrickRow, fTotalDistance);
        fOpacity *= beerLambert(kfAbsorption * (fMaxDistance - fDistance) / fMaxDistance, fTotalDistance);
        let fAbsorption: f32 = fPrevOpacity - fOpacity;

        let neighborBrixelIndex0: vec3<i32> = vec3<i32>( 
            clamp(i32(hitBrixelIndex.x) - 1, 0, 7),
            i32(hitBrixelIndex.y),
            i32(hitBrixelIndex.z)
        );
        let neighborBrixelIndex1: vec3<i32> = vec3<i32>(
            i32(hitBrixelIndex.x),  
            clamp(i32(hitBrixelIndex.y) - 1, 0, 7), 
            i32(hitBrixelIndex.y)
        );
        let neighborBrixelIndex2: vec3<i32> = vec3<i32>(
            i32(hitBrixelIndex.x),  
            i32(hitBrixelIndex.y), 
            clamp(i32(hitBrixelIndex.z) - 1, 0, 7)
        );

        let iNeighborIndex0: u32 = 
            u32(neighborBrixelIndex0.z) * iNumBrixelsPerBrickRowSquared + 
            u32(neighborBrixelIndex0.y) * iNumBrixelsPerBrickRow + 
            u32(neighborBrixelIndex0.x);
        let iNeighborIndex1: u32 = 
            u32(neighborBrixelIndex1.z) * iNumBrixelsPerBrickRowSquared + 
            u32(neighborBrixelIndex1.y) * iNumBrixelsPerBrickRow + 
            u32(neighborBrixelIndex1.x);
        let iNeighborIndex2: u32 = 
            u32(neighborBrixelIndex2.z) * iNumBrixelsPerBrickRowSquared + 
            u32(neighborBrixelIndex2.y) * iNumBrixelsPerBrickRow + 
            u32(neighborBrixelIndex2.x);

        var normal: vec3<f32> = vec3<f32>(
            aBrixelDistances[u32(iBrixelDistanceArrayIndex) * 512u + iNeighborIndex0],
            aBrixelDistances[u32(iBrixelDistanceArrayIndex) * 512u + iNeighborIndex1],
            aBrixelDistances[u32(iBrixelDistanceArrayIndex) * 512u + iNeighborIndex2]
        );
        normal = normalize(normal);

        var fLightVisibility: f32 = 1.0f;
        if(fDistanceToSeed <= 1.0f)
        {
            var toLightPosition: vec3<f32> = lightPosition0 - currPosition;
            var fLightDistance: f32 = length(toLightPosition);

            fLightVisibility *= lightTraceSDFDistance(
                currPosition,
                normalize(toLightPosition),
                minGlobalBrickPosition,
                numGlobalBricks,
                fBrickDimension,
                fBrixelDimension,
                iNumBrixelsPerBrickRow,
                50.0f,
                kfAbsorption);  

            toLightPosition = lightPosition1 - currPosition;
            fLightDistance = length(toLightPosition);
            fLightVisibility *= lightTraceSDFDistance(
                currPosition,
                normalize(toLightPosition),
                minGlobalBrickPosition,
                numGlobalBricks,
                fBrickDimension,
                fBrixelDimension,
                iNumBrixelsPerBrickRow,
                50.0f,
                kfAbsorption);
        }

        //var toLightPosition: vec3<f32> = normalize(lightPosition0 - currPosition);
        //fLightVisibility = max(dot(toLightPosition, normal), 0.0f);

        let fRadianceMult: f32 = 100.0f;

        var fDistanceToLight: f32 = length(lightPosition0 - currPosition);
        var toLightRadiance: vec3<f32> = ((lightRadiance0 * fRadianceMult) / (fDistanceToLight * fDistanceToLight)) * fLightVisibility;
        volumetricColor += volumeAlbedo * toLightRadiance * fAbsorption;

        fDistanceToLight = length(lightPosition1 - currPosition);
        toLightRadiance = ((lightRadiance1 * fRadianceMult) / (fDistanceToLight * fDistanceToLight)) * fLightVisibility;
        volumetricColor += volumeAlbedo * toLightRadiance * fAbsorption;

        currPosition += rayDirection * fBrixelDimension;
        fTotalDistance += fBrixelDimension;

    }   // for inc = 0 to num steps

    return volumetricColor;
}

/////
fn lightTraceSDFDistance(
    startPosition: vec3<f32>,
    rayDirection: vec3<f32>,
    minGlobalBrickPosition: vec3<f32>,
    numGlobalBricks: vec3<i32>,
    fBrickDimension: f32,
    fBrixelDimension: f32,
    iNumBrixelsPerBrickRow: u32,
    fNumSteps: f32,
    fAbsorptionFactor: f32) -> f32
{
    var ret: SphereTraceInfo;
    ret.mHitPosition = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    ret.miBrickIndex = UINT32_MAX;
    ret.miBrixelIndex = UINT32_MAX;

    let iNumBrixelsPerBrickRowSquared: u32 = iNumBrixelsPerBrickRow * iNumBrixelsPerBrickRow;
    let fOneOverHitBrickSize: f32 = 1.0f / fBrickDimension;
    
    // step through the mesh brixels along the ray
    var fPrevDistanceToSeed: f32 = FLT_MAX;
    var fDistanceToSeed: f32 = FLT_MAX;
    var currPosition: vec3<f32> = startPosition + rayDirection * fBrixelDimension * 4.0f;
    var lastPosition: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);

    let fNumBrixelsPerBrickRow: f32 = f32(iNumBrixelsPerBrickRow);
    let maxGlobalBrickPosition: vec3<f32> = minGlobalBrickPosition + vec3<f32>(f32(numGlobalBricks.x), f32(numGlobalBricks.y), f32(numGlobalBricks.z)) * fBrickDimension;
    
    let oneOverRayDirection: vec3<f32> = 1.0f / rayDirection;

    var fTotalDistance: f32 = 0.0f;
    var fLightVisibility: f32 = 1.0f;
    
    let iNumBrickXY: i32 = numGlobalBricks.x * numGlobalBricks.y;
    let iNumBrickX: i32 = numGlobalBricks.x;
    for(var fInc: f32 = 0.0f; fInc < fNumSteps; fInc += 1.0f)
    {
        var positionDiff: vec3<f32> = currPosition - minGlobalBrickPosition;
        var hitGlobalBrickIndexFloat: vec3<f32> = floor(positionDiff / fBrickDimension);
        var iGlobalBrickArrayIndex: u32 = u32(
            hitGlobalBrickIndexFloat.z * f32(numGlobalBricks.x * numGlobalBricks.y) +
            hitGlobalBrickIndexFloat.y * f32(numGlobalBricks.x) +
            hitGlobalBrickIndexFloat.x
        );
        
        lastPosition = currPosition;
        
        var iBrixelDistanceArrayIndex: i32 = aiBrickToBrixelMapping[iGlobalBrickArrayIndex];
        if(iBrixelDistanceArrayIndex <= 0)
        {
            // didn't hit any valid brick, continue in ray direction
            currPosition += rayDirection * fBrixelDimension;
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
        if(fDistanceToSeed <= 1.0f)
        {
            fLightVisibility *= beerLambert(fAbsorptionFactor, fBrixelDimension);
        }

        currPosition += rayDirection * fBrixelDimension;
        fTotalDistance += fBrixelDimension;

    }   // for inc = 0 to num steps

    return fLightVisibility;
}


/////
fn beerLambert(
    fAbsorptionCoefficient: f32, 
    fDistanceTraveled: f32) -> f32
{
    return exp(-fAbsorptionCoefficient * fDistanceTraveled);
}

/////
fn getAxis(
    normal: vec3<f32>) -> AxisInfo
{
    var ret: AxisInfo;

    var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    if(abs(normal.y) > abs(normal.x) && abs(normal.y) > abs(normal.z)) 
    { 
        up = vec3<f32>(1.0f, 0.0f, 0.0f);
    }

    ret.mTangent = normalize(cross(normal, up));//normalize(cross(up, normal));
    ret.mBinormal = normalize(cross(ret.mTangent, normal)); //normalize(cross(normal, ret.mTangent));

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