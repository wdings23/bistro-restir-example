const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const PROBE_IMAGE_SIZE: u32 = 8u;

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @location(2) normal : vec4<f32>
};
struct VertexOutput {
    @location(0) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) normal: vec4<f32>
};
struct FragmentOutput {
    @location(0) radianceOutput : vec4<f32>,
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

struct Vertex
{
    mPosition: vec4<f32>,
    mTexCoord: vec4<f32>,
    mNormal: vec4<f32>,
};


@group(0) @binding(0)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var texCoordTexture: texture_2d<f32>;

@group(0) @binding(3)
var skyTexture: texture_2d<f32>;

@group(0) @binding(4)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(5)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(6)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(7)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(8)
var<storage, read> sphericalHarmonicCoefficient0: array<vec4<f32>>;

@group(0) @binding(9)
var<storage, read> sphericalHarmonicCoefficient1: array<vec4<f32>>;

@group(0) @binding(10)
var<storage, read> sphericalHarmonicCoefficient2: array<vec4<f32>>;

@group(0) @binding(11)
var textureSampler: sampler;

@group(1) @binding(0)
var blueNoiseTexture: texture_2d<f32>;

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
    var fReservoirSize: f32 = 8.0f;
    let blueNoiseDimension: vec2<u32> = textureDimensions(blueNoiseTexture, 0);
    let textureSize: vec2<u32> = textureDimensions(worldPositionTexture, 0);

    var out: FragmentOutput;

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
        out.radianceOutput = vec4<f32>(0.0f, 0.0f, 1.0f, 0.0f);
        return out;
    }

    let iOffsetX: u32 = u32(defaultUniformData.miFrame) % blueNoiseDimension.x;
    let iOffsetY: u32 = u32((u32(defaultUniformData.miFrame) / blueNoiseDimension.x)) % (blueNoiseDimension.x * blueNoiseDimension.y);

    var screenCoord: vec2<u32> = vec2<u32>(
        u32(in.texCoord.x * f32(defaultUniformData.miScreenWidth)),
        u32(in.texCoord.y * f32(defaultUniformData.miScreenHeight))
    );
    screenCoord.x = (screenCoord.x + iOffsetX) % textureSize.x;
    screenCoord.y = (screenCoord.y + iOffsetY) % textureSize.y;
    var sampleUV: vec2<f32> = vec2<f32>(
        f32(screenCoord.x) / f32(textureSize.x),
        f32(screenCoord.y) / f32(textureSize.y) 
    );

    var blueNoise: vec3<f32> = textureSample(
        blueNoiseTexture,
        textureSampler,
        sampleUV
    ).xyz;

    let iNumCenterSamples: i32 = 64;
    let iCurrIndex: u32 = u32(defaultUniformData.miFrame) * u32(iNumCenterSamples);

    let iTileSize: u32 = 32u;
    let iNumTilesPerRow: u32 = textureSize.x / iTileSize;

    let prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);

    var radiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    for(var i: i32 = 0; i < iNumCenterSamples; i++)
    {
        let iTileX: u32 = (iCurrIndex + u32(iNumCenterSamples)) % iNumTilesPerRow;
        let iTileY: u32 = ((iCurrIndex + u32(iNumCenterSamples)) / iNumTilesPerRow) % (iNumTilesPerRow * iNumTilesPerRow);

        let iTileOffsetX: u32 = (iCurrIndex + u32(iNumCenterSamples)) % iTileSize;
        let iTileOffsetY: u32 = ((iCurrIndex + u32(iNumCenterSamples)) / iTileSize) % (iTileSize * iTileSize);

        let iOffsetX: u32 = iTileOffsetX + iTileX * iTileSize;
        let iOffsetY: u32 = iTileOffsetY + iTileY * iTileSize; 

        screenCoord.x = (screenCoord.x + iOffsetX) % textureSize.x;
        screenCoord.y = (screenCoord.y + iOffsetY) % textureSize.y;
        var sampleUV: vec2<f32> = vec2<f32>(
            f32(screenCoord.x) / f32(textureSize.x),
            f32(screenCoord.y) / f32(textureSize.y) 
        );

        var blueNoise: vec3<f32> = textureSample(
            blueNoiseTexture,
            textureSampler,
            sampleUV
        ).xyz;

        let direction: vec3<f32> = uniformSampling(
            worldPosition.xyz,
            normal.xyz,
            blueNoise.x,
            blueNoise.y);

        let fDP: f32 = max(dot(direction, normal.xyz), 0.0f);
        radiance += decodeFromSphericalHarmonicCoefficients(
            direction,
            vec3<f32>(5.0f, 5.0f, 5.0f),
            in.texCoord
        ) * fDP;
    }

    out.radianceOutput = vec4<f32>(radiance.xyz / f32(iNumCenterSamples), 1.0f);

    return out;
}

/////
fn uniformSampling(
    worldPosition: vec3<f32>,
    normal: vec3<f32>,
    fRand0: f32,
    fRand1: f32) -> vec3<f32>
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

    return rayDirection;
}

/////
fn decodeFromSphericalHarmonicCoefficients(
    direction: vec3<f32>,
    maxRadiance: vec3<f32>,
    texCoord: vec2<f32>
) -> vec3<f32>
{
    let fNumSamples: f32 = textureSample(
        ambientOcclusionTexture,
        textureSampler,
        texCoord
    ).y;

    let iOutputX: i32 = i32(texCoord.x * f32(defaultUniformData.miScreenWidth));
    let iOutputY: i32 = i32(texCoord.y * f32(defaultUniformData.miScreenHeight));
    let iImageIndex: i32 = iOutputY * defaultUniformData.miScreenWidth + iOutputX;
    let fFactor: f32 = (4.0f * 3.14159f) / (fNumSamples * 1.0f);

    let SHCoefficent0: vec4<f32> = sphericalHarmonicCoefficient0[iImageIndex];
    let SHCoefficent1: vec4<f32> = sphericalHarmonicCoefficient1[iImageIndex];
    let SHCoefficent2: vec4<f32> = sphericalHarmonicCoefficient2[iImageIndex];

    var aTotalCoefficients: array<vec3<f32>, 4>;
    aTotalCoefficients[0] = vec3<f32>(SHCoefficent0.x, SHCoefficent0.y, SHCoefficent0.z) * fFactor;
    aTotalCoefficients[1] = vec3<f32>(SHCoefficent0.w, SHCoefficent1.x, SHCoefficent1.y) * fFactor;
    aTotalCoefficients[2] = vec3<f32>(SHCoefficent1.z, SHCoefficent1.w, SHCoefficent2.x) * fFactor;
    aTotalCoefficients[3] = vec3<f32>(SHCoefficent2.y, SHCoefficent2.z, SHCoefficent2.w) * fFactor;

    let fC1: f32 = 0.42904276540489171563379376569857f;
    let fC2: f32 = 0.51166335397324424423977581244463f;
    let fC3: f32 = 0.24770795610037568833406429782001f;
    let fC4: f32 = 0.88622692545275801364908374167057f;

    var decoded: vec3<f32> =
        aTotalCoefficients[0] * fC4 +
        (aTotalCoefficients[3] * direction.x + aTotalCoefficients[1] * direction.y + aTotalCoefficients[2] * direction.z) * 
        fC2 * 2.0f;
    decoded = clamp(decoded, vec3<f32>(0.0f, 0.0f, 0.0f), maxRadiance);

    return decoded;
}

/////
fn getPreviousScreenUV(
    screenUV: vec2<f32>) -> vec2<f32>
{
    var motionVector: vec2<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        screenUV).xy;
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
            var sampleUV: vec2<f32> = prevScreenUV + vec2<f32>(
                f32(iX) * fOneOverScreenWidth,
                f32(iY) * fOneOverScreenHeight 
            );

            sampleUV.x = clamp(sampleUV.x, 0.0f, 1.0f);
            sampleUV.y = clamp(sampleUV.y, 0.0f, 1.0f);

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