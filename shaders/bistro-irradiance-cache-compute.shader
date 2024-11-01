const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const PROBE_IMAGE_SIZE: u32 = 8u;

var<workgroup> giNumQueueEntries: atomic<i32>;

struct RandomResult {
    mfNum: f32,
    miSeed: u32,
};

struct IntersectBVHResult
{
    mHitPosition: vec3<f32>,
    mHitNormal: vec3<f32>,
    mBarycentricCoordinate: vec3<f32>,
    miHitTriangle: u32,
    miMeshID: u32,
    mHitUV: vec2<f32>,
};


struct RayTriangleIntersectionResult
{
    mIntersectPosition: vec3<f32>,
    mIntersectNormal: vec3<f32>,
    mBarycentricCoordinate: vec3<f32>,
    mIntersectionUV: vec2<f32>,
};

struct Tri
{
    miV0 : u32,
    miV1 : u32,
    miV2 : u32,
    mPadding : u32,

    mCentroid : vec4<f32>
};

struct Ray
{
    mOrigin:                    vec4<f32>,
    mDirection:                 vec4<f32>,
    mfT:                        vec4<f32>,
};

struct IrradianceCacheQueueEntry
{
    mPosition:                  vec4<f32>,
    mNormal:                    vec4<f32>
};

struct IrradianceCacheEntry
{
    mPosition:              vec4<f32>,
    mImageProbe:            array<vec4<f32>, 64>,            // 8x8 image probe
};

struct MeshInfo
{
    miPositionOffsetStart: u32,
    miNumPositions: u32,

    miVertexIndexOffsetStart: u32,
    miNumTriangles: u32
};

struct MeshInfoList
{
    miNumMeshes: u32,
    maMeshInfo: array<MeshInfo, 4>
};

struct UniformData
{
    miFrame: u32,
    miRand0: u32,
    miRand1: u32,
    miRand2: u32,

    mViewProjectionMatrix: mat4x4<f32>,
    
    miScreenWidth: u32,
    miScreenHeight: u32,
    miFrame: u32,
    miNumMeshes: u32,
};

struct UpdateIrradianceCacheResult
{
    mRandomResult: RandomResult,
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

struct BVHNodeIndex
{
    miStartNodeIndex: u32,
    miEndNodeIndex: u32,
    miBufferIndex: u32,
};

struct Vertex
{
    mPosition: vec4<f32>,
    mTexCoord: vec4<f32>,
    mNormal: vec4<f32>,
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
var<storage, read_write> irradianceCache: array<IrradianceCacheEntry>;

@group(0) @binding(1)
var<storage, read_write> irradianceCacheQueue: array<IrradianceCacheQueueEntry>;

@group(0) @binding(2)
var<storage, read_write> counters: array<i32>;

@group(0) @binding(3)
var skyTexture: texture_2d<f32>;

@group(0) @binding(4)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(5)
var hitPositionTexture: texture_2d<f32>;

@group(0) @binding(6)
var hitNormalTexture: texture_2d<f32>;

@group(0) @binding(7)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> aTLASBVHNodes: array<BVHNode2>;

@group(1) @binding(2)
var<storage, read> aBLASBVHNodes0: array<BVHNode2>;

@group(1) @binding(3)
var<storage, read> aBLASBVHNodes1: array<BVHNode2>;

@group(1) @binding(4)
var<storage, read> aMeshBLASNodeIndices: array<BVHNodeIndex>;

@group(1) @binding(5)
var<storage, read> aVertexBuffer: array<Vertex>;

@group(1) @binding(6)
var<storage, read> aiIndexBuffer: array<u32>;

@group(1) @binding(7)
var<storage, read> aMaterials: array<Material>;

@group(1) @binding(8)
var<storage, read> aMeshMaterialID: array<u32>;

@group(1) @binding(9)
var<uniform> defaultUniformData: DefaultUniformData;

const iNumThreadX: u32 = 14u;
const iNumThreadY: u32 = 14u;

/////
@compute
@workgroup_size(iNumThreadX, iNumThreadY)
//fn cs_main(@builtin(global_invocation_id) index: vec3<u32>) 
fn cs_main(
    @builtin(workgroup_id) workGroupIndex: vec3<u32>,
    @builtin(local_invocation_id) localIndex: vec3<u32>,
    @builtin(num_workgroups) workGroupSize: vec3<u32>) 
{
    let iMaxNumCachePerFrame: i32 = 200;

    var randomResult: RandomResult = initRand(
        u32(f32(localIndex.x + workGroupIndex.y) * 100.0f + f32(localIndex.y + workGroupIndex.x) * 200.0f) + u32(defaultUniformData.mfRand0 * 100.0f),
        u32(f32(localIndex.x + workGroupIndex.x) * 10.0f + f32(localIndex.z + workGroupIndex.y) * 20.0f) + u32(defaultUniformData.mfRand1 * 100.0f),
        10u);

    let iThreadID: u32 = u32(localIndex.y) + iNumThreadX + u32(localIndex.x);
    if(iThreadID == 0)
    {
        giNumQueueEntries = 0;
    }
    workgroupBarrier();

    // total threads with workgroup and local workgroup
    let iNumTotalThreadX: u32 = iNumThreadX * workGroupSize.x;
    let iNumTotalThreadY: u32 = iNumThreadY * workGroupSize.y; 

    //let iTileWidth: u32 = defaultUniformData.miScreenWidth / iNumThreadX;
    //let iTileHeight: u32 = defaultUniformData.miScreenHeight / iNumThreadY;

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
            
            let hitPosition: vec4<f32> = textureSampleLevel(
                hitPositionTexture,
                textureSampler,
                uv,
                0.0f
            );

            let fRayLength: f32 = length(hitPosition.xyz - worldPosition.xyz);

            // skip on pixels hitting the sky
            if(fRayLength >= 100.0f)
            {
                continue;
            }

            // check if on screen
            var clipSpacePosition: vec4<f32> = vec4<f32>(hitPosition.xyz, 1.0) * defaultUniformData.mViewProjectionMatrix;
            clipSpacePosition.x /= clipSpacePosition.w;
            clipSpacePosition.y /= clipSpacePosition.w;
            clipSpacePosition.z /= clipSpacePosition.w;
            clipSpacePosition.x = clipSpacePosition.x * 0.5f + 0.5f;
            clipSpacePosition.y = 1.0f - (clipSpacePosition.y * 0.5f + 0.5f);
            clipSpacePosition.z = clipSpacePosition.z * 0.5f + 0.5f;
            
            let hitWorldPosition: vec4<f32> = textureSampleLevel(
                worldPositionTexture,
                textureSampler,
                clipSpacePosition.xy,
                0.0f);
            var hitPositionClipSpace: vec4<f32> = vec4<f32>(hitWorldPosition.xyz, 1.0f) * defaultUniformData.mViewProjectionMatrix;
            hitPositionClipSpace.x /= hitPositionClipSpace.w;
            hitPositionClipSpace.y /= hitPositionClipSpace.w;
            hitPositionClipSpace.z /= hitPositionClipSpace.w;
            hitPositionClipSpace.x = hitPositionClipSpace.x * 0.5f + 0.5f;
            hitPositionClipSpace.y = 1.0f - (hitPositionClipSpace.y * 0.5f + 0.5f);
            hitPositionClipSpace.z = hitPositionClipSpace.z * 0.5f + 0.5f;

            // depth difference between the hit position and current on screen world position
            // larger hit position's depth means it's in the back of the current on screen world position
            // so the difference greater than threshold means the hit position is obscured by the current world position
            let fDepthDiff: f32 = clipSpacePosition.z - hitPositionClipSpace.z; 

            if(giNumQueueEntries < 100 &&
               (clipSpacePosition.x < 0.0f || clipSpacePosition.x > 1.0f ||
               clipSpacePosition.y < 0.0f || clipSpacePosition.y > 1.0f || 
               fDepthDiff >= 0.01f))
            {
                // not on screen, register irradiance cache for hit position
                
                //if(abs(hitPosition.x) <= 100.0f && abs(hitPosition.y) <= 100.0f && abs(hitPosition.z) <= 100.0f)
                {
                    let hitNormal: vec4<f32> = textureSampleLevel(
                        hitNormalTexture,
                        textureSampler,
                        uv,
                        0.0f
                    );

                    updateIrradianceCache(
                        hitPosition.xyz, 
                        hitNormal.xyz,
                        randomResult,
                        localIndex.x,
                        localIndex.y);

                    atomicAdd(&giNumQueueEntries, 1);
                }
            }

            if(giNumQueueEntries >= iMaxNumCachePerFrame)
            {
                break;
            }
        }

        if(giNumQueueEntries >= iMaxNumCachePerFrame)
        {
            break;
        }
    }

    //counters[0] = giNumQueueEntries;
}

/////
fn fetchIrradianceCacheIndex(
    position: vec3<f32>
) -> u32
{
    var scaledPosition: vec3<f32> = position * 10.0f;
    let fSignX: f32 = sign(position.x);
    let fSignY: f32 = sign(position.y);
    let fSignZ: f32 = sign(position.z);
    scaledPosition.x = f32(floor(abs(position.x) + 0.5f)) * 0.1f * fSignX; 
    scaledPosition.y = f32(floor(abs(position.y) + 0.5f)) * 0.1f * fSignY; 
    scaledPosition.z = f32(floor(abs(position.z) + 0.5f)) * 0.1f * fSignZ; 

    let iHashKey: u32 = hash13(
        scaledPosition,
        5000u
    );

    return iHashKey;
}

/////
fn createIrradianceCacheEntry(
    position: vec3<f32>
) -> u32
{
    let iIrradianceCacheIndex: u32 = fetchIrradianceCacheIndex(position);

    var scaledPosition: vec3<f32> = position * 10.0f;
    let fSignX: f32 = sign(position.x);
    let fSignY: f32 = sign(position.y);
    let fSignZ: f32 = sign(position.z);
    scaledPosition.x = f32(floor(abs(position.x) + 0.5f)) * 0.1f * fSignX; 
    scaledPosition.y = f32(floor(abs(position.y) + 0.5f)) * 0.1f * fSignY; 
    scaledPosition.z = f32(floor(abs(position.z) + 0.5f)) * 0.1f * fSignZ; 
    
    irradianceCache[iIrradianceCacheIndex].mPosition = vec4<f32>(scaledPosition, 1.0f);

    for(var i: u32 = 0; i < 64u; i++)
    {
        irradianceCache[iIrradianceCacheIndex].mImageProbe[i] = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    }

    return iIrradianceCacheIndex;
}

/////
fn getRadianceFromIrradianceCacheProbe(
    rayDirection: vec3<f32>,
    iIrradianceCacheIndex: u32
) -> vec3<f32>
{
    let probeImageUV: vec2<f32> = octahedronMap2(rayDirection);
    var iImageY: u32 = clamp(u32(probeImageUV.y * f32(PROBE_IMAGE_SIZE)), 0u, PROBE_IMAGE_SIZE - 1u);
    var iImageX: u32 = clamp(u32(probeImageUV.x * f32(PROBE_IMAGE_SIZE)), 0u, PROBE_IMAGE_SIZE - 1u);
    var iImageIndex: u32 = iImageY * PROBE_IMAGE_SIZE + iImageX;

    return irradianceCache[iIrradianceCacheIndex].mImageProbe[iImageIndex].xyz;
}

/////
fn updateIrradianceCache(
    hitPosition: vec3<f32>,
    hitNormal: vec3<f32>,
    randomResult: RandomResult,
    iThreadX: u32,
    iThreadY: u32
) -> UpdateIrradianceCacheResult
{
    var ret: UpdateIrradianceCacheResult;
    var randomResultCopy: RandomResult;

    // create new irradiance cache entry if non-existent
    var iCurrIrradianceCacheEntry: u32 = fetchIrradianceCacheIndex(hitPosition);
    if(irradianceCache[iCurrIrradianceCacheEntry].mPosition.w <= 0.0f)
    {
        iCurrIrradianceCacheEntry = createIrradianceCacheEntry(hitPosition);
    }

    // sample ray for the center sample
    randomResultCopy = nextRand(randomResultCopy.miSeed);
    let fRand0: f32 = randomResultCopy.mfNum;
    randomResultCopy = nextRand(randomResultCopy.miSeed);
    let fRand1: f32 = randomResultCopy.mfNum;
    let ray: Ray = uniformSampling(
        hitPosition,
        hitNormal,
        fRand0,
        fRand1);

    // hit position and hit normal
    var intersectionInfo: IntersectBVHResult;
    intersectionInfo.miHitTriangle = UINT32_MAX;
    intersectionInfo = intersectTotalBVH4(ray, 0u);
    var candidateHitPosition: vec4<f32> = vec4<f32>(intersectionInfo.mHitPosition.xyz, 1.0f);
    var candidateHitNormal: vec4<f32> = vec4<f32>(intersectionInfo.mHitNormal.xyz, 1.0f);

    let fRayLength: f32 = length(intersectionInfo.mHitPosition.xyz - hitPosition);

    // process intersection
    var cacheRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    if(intersectionInfo.miHitTriangle == UINT32_MAX || fRayLength >= 100.0f)
    {
        // didn't hit anything, use skylight
        let fRadianceDP: f32 = max(dot(hitNormal.xyz, ray.mDirection.xyz), 0.0f);
        let skyUV: vec2<f32> = octahedronMap2(ray.mDirection.xyz);
        
        let skyRadiance: vec3<f32> = textureSampleLevel(
            skyTexture,
            textureSampler,
            skyUV,
            0.0f).xyz;
        cacheRadiance = skyRadiance * fRadianceDP;

        // update irradiance cache probe with the radiance and ray direction 
        updateIrradianceCacheProbe(
            cacheRadiance, 
            ray.mDirection.xyz * -1.0f,
            iCurrIrradianceCacheEntry);
    }
    else 
    {
        var iHitTriangleIndex: i32 = i32(intersectionInfo.miHitTriangle);

        // get hit mesh
        let iMaterialID: u32 = aMeshMaterialID[intersectionInfo.miMeshID];
        let material: Material = aMaterials[iMaterialID-1];
        cacheRadiance += material.mEmissive.xyz;

        // hit another mesh triangle, get the irradiance cache in the vicinity
        let iHitIrradianceCacheIndex: u32 = fetchIrradianceCacheIndex(candidateHitPosition.xyz);
        if(irradianceCache[iHitIrradianceCacheIndex].mPosition.w <= 0.0f)
        {
            // no valid irradiance cache here, create a new one
            createIrradianceCacheEntry(candidateHitPosition.xyz);
        }

        // get the radiance from probe image
        let fRadianceDP: f32 = max(dot(hitNormal.xyz, ray.mDirection.xyz), 0.0f);
        cacheRadiance += getRadianceFromIrradianceCacheProbe(ray.mDirection.xyz * -1.0f, iHitIrradianceCacheIndex) * fRadianceDP;
    
        // update irradiance cache probe with the radiance and ray direction 
        updateIrradianceCacheProbe(
            cacheRadiance, 
            ray.mDirection.xyz * -1.0f,
            iCurrIrradianceCacheEntry);
    }
    
    

// test test test
//irradianceCache[iCurrIrradianceCacheEntry].mPosition.w = f32(iThreadX) + f32(iThreadY) * 0.1f;
irradianceCache[iCurrIrradianceCacheEntry].mPosition.w = f32(defaultUniformData.miFrame);

    ret.mRandomResult = randomResultCopy;
    return ret;
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
fn decodeOctahedronMap(uv: vec2<f32>) -> vec3<f32>
{
    let newUV: vec2<f32> = uv * 2.0f - vec2<f32>(1.0f, 1.0f);

    let absUV: vec2<f32> = vec2<f32>(abs(newUV.x), abs(newUV.y));
    var v: vec3<f32> = vec3<f32>(newUV.x, newUV.y, 1.0f - (absUV.x + absUV.y));

    if(absUV.x + absUV.y > 1.0f) 
    {
        v.x = (abs(newUV.y) - 1.0f) * -sign(newUV.x);
        v.y = (abs(newUV.x) - 1.0f) * -sign(newUV.y);
    }

    v.y *= -1.0f;

    return v;
}

struct SphericalHarmonicsEncodeResult
{
    maCoefficients: array<vec3<f32>, 4>, 
};

struct SphericalHarmonicsDecodeResult
{
    mProbeImage: array<vec4<f32>, 64>,
};

/////
fn encodeProbeImage(
    aFilteredProbeImage: array<vec4<f32>, 64>
) -> SphericalHarmonicsEncodeResult
{
    var ret: SphericalHarmonicsEncodeResult;

    var probeImage: array<vec4<f32>, 64> = aFilteredProbeImage;

    var aTotalCoefficients: array<vec3<f32>, 4>;
    aTotalCoefficients[0] = vec3<f32>(0.0f, 0.0f, 0.0f);
    aTotalCoefficients[1] = vec3<f32>(0.0f, 0.0f, 0.0f);
    aTotalCoefficients[2] = vec3<f32>(0.0f, 0.0f, 0.0f);
    aTotalCoefficients[3] = vec3<f32>(0.0f, 0.0f, 0.0f);
    
    // encode to spherical harmonics coefficients
    var fNumSamples: f32 = 0.0f;
    for(var iLocalTileImageY: u32 = 0; iLocalTileImageY < PROBE_IMAGE_SIZE; iLocalTileImageY++)
    {
        for(var iLocalTileImageX: u32 = 0; iLocalTileImageX < PROBE_IMAGE_SIZE; iLocalTileImageX++)
        {
            let tileUV: vec2<f32> = vec2<f32>(
                f32(iLocalTileImageX) / f32(PROBE_IMAGE_SIZE),
                f32(iLocalTileImageY) / f32(PROBE_IMAGE_SIZE));

            // radiance and direction
            let iLocalImageIndex: u32 = iLocalTileImageY * PROBE_IMAGE_SIZE + iLocalTileImageX;
            let radiance: vec4<f32> = vec4<f32>(probeImage[iLocalImageIndex].xyz, 1.0f);
            let direction: vec3<f32> = decodeOctahedronMap(tileUV);
            
            // encode coefficients with direction
            var afC: array<f32, 4>;
            afC[0] = 0.282095f;
            afC[1] = 0.488603f;
            afC[2] = 0.488603f;
            afC[3] = 0.488603f;

            var afCoefficients: array<f32, 4>;
            afCoefficients[0] = afC[0];
            afCoefficients[1] = afC[1] * direction.y;
            afCoefficients[2] = afC[2] * direction.z;
            afCoefficients[3] = afC[3] * direction.x;
            
            // encode radiance with direction coefficients
            var aResults: array<vec3<f32>, 4>;
            aResults[0] = radiance.xyz * afCoefficients[0];
            aResults[1] = radiance.xyz * afCoefficients[1];
            aResults[2] = radiance.xyz * afCoefficients[2];
            aResults[3] = radiance.xyz * afCoefficients[3];
            
            // apply to total coefficients
            aTotalCoefficients[0] += aResults[0];
            aTotalCoefficients[1] += aResults[1];
            aTotalCoefficients[2] += aResults[2];
            aTotalCoefficients[3] += aResults[3];

            fNumSamples += 1.0f;
            
        }   // for x = 0 to tile image size

    }   // for y = 0 to tile image size

    ret.maCoefficients = aTotalCoefficients;

    return ret;
}

/////
fn decodeProbeImage(
    encoding: SphericalHarmonicsEncodeResult,
    maxRadiance: vec3<f32>) -> SphericalHarmonicsDecodeResult
{
    var ret: SphericalHarmonicsDecodeResult;

    var aTotalCoefficients: array<vec3<f32>, 4>;
    let fTotalSamples: f32 = f32(PROBE_IMAGE_SIZE * PROBE_IMAGE_SIZE);
    let fFactor: f32 = (4.0f * 3.14159f) / fTotalSamples;
    aTotalCoefficients[0] = encoding.maCoefficients[0] * fFactor;
    aTotalCoefficients[1] = encoding.maCoefficients[1] * fFactor;
    aTotalCoefficients[2] = encoding.maCoefficients[2] * fFactor;
    aTotalCoefficients[3] = encoding.maCoefficients[3] * fFactor;

    let fC1: f32 = 0.42904276540489171563379376569857f;
    let fC2: f32 = 0.51166335397324424423977581244463f;
    let fC3: f32 = 0.24770795610037568833406429782001f;
    let fC4: f32 = 0.88622692545275801364908374167057f;

    // apply coefficients for decoding
    for(var iLocalTileImageY: u32 = 0; iLocalTileImageY < PROBE_IMAGE_SIZE; iLocalTileImageY++)
    {
        for(var iLocalTileImageX: u32 = 0; iLocalTileImageX < PROBE_IMAGE_SIZE; iLocalTileImageX++)
        {
            let tileUV = vec2<f32>(
                f32(iLocalTileImageX) / f32(PROBE_IMAGE_SIZE),
                f32(iLocalTileImageY) / f32(PROBE_IMAGE_SIZE));
            
            let direction: vec3<f32> = decodeOctahedronMap(tileUV);
            var decoded: vec3<f32> =
                aTotalCoefficients[0] * fC4 +
                (aTotalCoefficients[3] * direction.x + aTotalCoefficients[1] * direction.y + aTotalCoefficients[2] * direction.z) * 
                fC2 * 2.0f;
            decoded = clamp(decoded, vec3<f32>(0.0f, 0.0f, 0.0f), maxRadiance);
            let iImageIndex: u32 = iLocalTileImageY * PROBE_IMAGE_SIZE + iLocalTileImageX;
            ret.mProbeImage[iImageIndex] = vec4<f32>(decoded, 1.0f);
        }
    }

    return ret;

}

/////
fn updateIrradianceCacheProbe(
    radiance: vec3<f32>,
    direction: vec3<f32>,
    iCacheEntryIndex: u32)
{
    // Steps: 
    //      1) fetch image probe from the given irradiance cache index
    //      2) add radiance to image probe
    //      3) encode to spherical harmonics coefficients
    //      4) decode back to probe with the coefficients above, the result is a filtered probe with the spherical harmonics info built in, ie. image probe in spherical directions
    //      5) save probe to the irradiance cache

    // set probe image pixel
    let probeImageUV: vec2<f32> = octahedronMap2(direction);
    let probeImageCoord: vec2<u32> = vec2<u32>(
        clamp(u32(probeImageUV.x * 8.0f), 0u, 7u),
        clamp(u32(probeImageUV.y * 8.0f), 0u, 7u)
    );
    let iImageIndex: u32 = probeImageCoord.x * 8 + probeImageCoord.x;

    irradianceCache[iCacheEntryIndex].mImageProbe[iImageIndex] = vec4<f32>(radiance.xyz, 1.0f);

    // max radiance for clamping during decoding phase
    var maxRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    for(var i: u32 = 0; i < PROBE_IMAGE_SIZE * PROBE_IMAGE_SIZE; i++)
    {
        maxRadiance.x = max(irradianceCache[iCacheEntryIndex].mImageProbe[i].x, maxRadiance.x);
        maxRadiance.y = max(irradianceCache[iCacheEntryIndex].mImageProbe[i].y, maxRadiance.y);
        maxRadiance.z = max(irradianceCache[iCacheEntryIndex].mImageProbe[i].z, maxRadiance.z); 
    }

    // get the updated spherical harmonics encoded probe image
    let encodedResult: SphericalHarmonicsEncodeResult = encodeProbeImage(irradianceCache[iCacheEntryIndex].mImageProbe);
    var decodedResult: SphericalHarmonicsDecodeResult = decodeProbeImage(
        encodedResult, 
        maxRadiance);

    // update the spherical harmonic probe image
    irradianceCache[iCacheEntryIndex].mImageProbe = decodedResult.mProbeImage;
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
    ray.mOrigin = vec4<f32>(worldPosition + normal * 0.05f, 1.0f);
    ray.mDirection = vec4<f32>(rayDirection, 1.0f);
    ray.mfT = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

    return ray;
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
fn intersectTri4(
    ray: Ray,
    iTriangleIndex: u32) -> RayTriangleIntersectionResult
{
    let iIndex: u32 = iTriangleIndex * 3u;
    let pos0: vec4<f32> = aVertexBuffer[aiIndexBuffer[iIndex]].mPosition;
    let pos1: vec4<f32> = aVertexBuffer[aiIndexBuffer[iIndex + 1u]].mPosition;
    let pos2: vec4<f32> = aVertexBuffer[aiIndexBuffer[iIndex + 2u]].mPosition;

    let uv0: vec2<f32> = aVertexBuffer[aiIndexBuffer[iIndex]].mTexCoord.xy;
    let uv1: vec2<f32> = aVertexBuffer[aiIndexBuffer[iIndex + 1u]].mTexCoord.xy;
    let uv2: vec2<f32> = aVertexBuffer[aiIndexBuffer[iIndex + 2u]].mTexCoord.xy;

    var iIntersected: u32 = 0u;
    var intersectionInfo: RayTriangleIntersectionResult = rayTriangleIntersection(
        ray.mOrigin.xyz,
        ray.mOrigin.xyz + ray.mDirection.xyz * 1000.0f,
        pos0.xyz,
        pos1.xyz,
        pos2.xyz);
    
    intersectionInfo.mIntersectionUV = (
        uv0 * intersectionInfo.mBarycentricCoordinate.x + 
        uv1 * intersectionInfo.mBarycentricCoordinate.y + 
        uv2 * intersectionInfo.mBarycentricCoordinate.z);

    return intersectionInfo;
}

/////
fn intersectBLASBVH4Part0(
    ray: Ray,
    iStartNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 16>;
    aiStack[iStackTop] = iStartNodeIndex;

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

        if(aBLASBVHNodes0[iNodeIndex].miPrimitiveID != UINT32_MAX)
        {
            let intersectionInfo: RayTriangleIntersectionResult = intersectTri4(
                ray,
                aBLASBVHNodes0[iNodeIndex].miPrimitiveID);

            if(abs(intersectionInfo.mIntersectPosition.x) < 10000.0f)
            {
                //let fDistanceToEye: f32 = length(intersectionInfo.mIntersectPosition.xyz - ray.mOrigin.xyz);
                {
                    ret.mHitPosition = intersectionInfo.mIntersectPosition.xyz;
                    ret.mHitNormal = intersectionInfo.mIntersectNormal.xyz;
                    ret.miHitTriangle = aBLASBVHNodes0[iNodeIndex].miPrimitiveID;
                    ret.mBarycentricCoordinate = intersectionInfo.mBarycentricCoordinate;
                    ret.miMeshID = aBLASBVHNodes0[iNodeIndex].miMeshID;
                    ret.mHitUV = intersectionInfo.mIntersectionUV;

                    break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aBLASBVHNodes0[iNodeIndex].mMinBound.xyz,
                aBLASBVHNodes0[iNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aBLASBVHNodes0[iNodeIndex].miChildren0 + iStartNodeIndex;
                iStackTop += 1;
                aiStack[iStackTop] = aBLASBVHNodes0[iNodeIndex].miChildren1 + iStartNodeIndex;
            }
        }
    }

    return ret;
}


/////
fn intersectBLASBVH4Part1(
    ray: Ray,
    iStartNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 16>;
    aiStack[iStackTop] = iStartNodeIndex;

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

        if(aBLASBVHNodes1[iNodeIndex].miPrimitiveID != UINT32_MAX)
        {
            let intersectionInfo: RayTriangleIntersectionResult = intersectTri4(
                ray,
                aBLASBVHNodes1[iNodeIndex].miPrimitiveID);

            if(abs(intersectionInfo.mIntersectPosition.x) < 10000.0f)
            {
                //let fDistanceToEye: f32 = length(intersectionInfo.mIntersectPosition.xyz - ray.mOrigin.xyz);
                {
                    ret.mHitPosition = intersectionInfo.mIntersectPosition.xyz;
                    ret.mHitNormal = intersectionInfo.mIntersectNormal.xyz;
                    ret.miHitTriangle = aBLASBVHNodes1[iNodeIndex].miPrimitiveID;
                    ret.mBarycentricCoordinate = intersectionInfo.mBarycentricCoordinate;
                    ret.miMeshID = aBLASBVHNodes1[iNodeIndex].miMeshID;
                    ret.mHitUV = intersectionInfo.mIntersectionUV;

                    break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aBLASBVHNodes1[iNodeIndex].mMinBound.xyz,
                aBLASBVHNodes1[iNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aBLASBVHNodes1[iNodeIndex].miChildren0 + iStartNodeIndex;
                iStackTop += 1;
                aiStack[iStackTop] = aBLASBVHNodes1[iNodeIndex].miChildren1 + iStartNodeIndex;
            }
        }
    }

    return ret;
}

/////
fn intersectTotalBVH4(
    ray: Ray,
    iRootNodeIndex: u32) -> IntersectBVHResult
{
    var ret: IntersectBVHResult;

    var iStackTop: i32 = 0;
    var aiStack: array<u32, 16>;
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

        let iTLASNodeIndex: u32 = aiStack[iStackTop];
        iStackTop -= 1;

        if(aTLASBVHNodes[iTLASNodeIndex].miMeshID != UINT32_MAX)
        {
            // min and max bounds of the top mesh bvh node to check ray intersection against
            let iMeshID: u32 = aTLASBVHNodes[iTLASNodeIndex].miMeshID;
            let iBufferIndex: u32 = aMeshBLASNodeIndices[iMeshID].miBufferIndex;
            let iBLASNodeIndex: u32 = aMeshBLASNodeIndices[iMeshID].miStartNodeIndex;
            let diff: vec3<f32> = aTLASBVHNodes[iTLASNodeIndex].mCentroid.xyz - ray.mOrigin.xyz;
            let fDistance: f32 = dot(diff, diff);

            if(fDistance <= 100.0f)
            {
                var intersectBLASResult: IntersectBVHResult;
                if(iBufferIndex == 0u)
                {
                    intersectBLASResult = intersectBLASBVH4Part0(ray, iBLASNodeIndex);
                }
                else 
                {
                    intersectBLASResult = intersectBLASBVH4Part1(ray, iBLASNodeIndex);
                }

                if(intersectBLASResult.miHitTriangle != UINT32_MAX)
                {
                    ret.miMeshID = iMeshID;
                    ret.miHitTriangle = intersectBLASResult.miHitTriangle;
                    ret.mHitPosition = intersectBLASResult.mHitPosition;
                    ret.mHitUV = intersectBLASResult.mHitUV;
                    ret.mHitNormal = intersectBLASResult.mHitNormal;
                    break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                aTLASBVHNodes[iTLASNodeIndex].mMinBound.xyz,
                aTLASBVHNodes[iTLASNodeIndex].mMaxBound.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = aTLASBVHNodes[iTLASNodeIndex].miChildren0;
                iStackTop += 1;
                aiStack[iStackTop] = aTLASBVHNodes[iTLASNodeIndex].miChildren1;
            }
        }
    }

    return ret;
}
