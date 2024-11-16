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
var rayHitPositionTexture: texture_2d<f32>;

@group(0) @binding(12)
var roughMetalTexture: texture_2d<f32>;

@group(0) @binding(13)
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

    let viewDir: vec3<f32> = normalize(defaultUniformData.mCameraPosition.xyz - worldPosition.xyz);
    let fRoughness: f32 = 0.3f;     // temp for now

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

        //let direction: vec3<f32> = sampleGGXVNDF(
        //    viewDir,
        //    fRoughness,
        //    blueNoise.x,
        //    blueNoise.y);

        let direction: vec3<f32> = uniformSampling(
            normal.xyz,
            blueNoise.x,
            blueNoise.y);

        //let fDP: f32 = max(dot(direction, normal.xyz), 0.0f);
        radiance += decodeFromSphericalHarmonicCoefficients(
            direction,
            vec3<f32>(5.0f, 5.0f, 5.0f),
            in.texCoord
        );
    }

    radiance /= f32(iNumCenterSamples);

    let rayHitPosition: vec4<f32> = textureSample(
        rayHitPositionTexture,
        textureSampler,
        in.texCoord);

    let roughMetal: vec4<f32> = textureSample(
        roughMetalTexture,
        textureSampler,
        in.texCoord);

    let lightDir: vec3<f32> = extractDominantLightDirectionFromSphericalHarmonicCoefficients(in.texCoord);

    // PBR
    var specularLight: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var kS: f32 = 0.0f;
    {
        let fSpecular: f32 = max(roughMetal.y - 0.2f, 0.0f);

        let viewDir: vec3<f32> = normalize(defaultUniformData.mCameraPosition.xyz - worldPosition.xyz);

        var F0: vec3<f32> = vec3<f32>(0.04f, 0.04f, 0.04f);
        F0 = mix(F0, vec3<f32>(1.0f, 1.0f, 1.0f), fSpecular);

        // calculate per-light radiance
        //let lightDir: vec3<f32> = normalize(rayHitPosition.xyz - worldPosition.xyz);
        let halfV: vec3<f32> = normalize(viewDir + lightDir);
        
        // cook-torrance brdf
        let NDF: f32 = distributionGGX(normal.xyz, halfV, fRoughness);
        let geometry: f32 = geometrySmith(normal.xyz, viewDir, lightDir, fRoughness);
        let fresnel: vec3<f32> = fresnelShlickRoughness(max(dot(halfV, viewDir), 0.0f), F0, fRoughness);

        let numerator: vec3<f32> = fresnel * (NDF * geometry);
        let denominator: f32 = 4.0f * max(dot(normal.xyz, viewDir), 0.0f) * max(dot(normal.xyz, lightDir), 0.0f) + 0.0001f;
        let specular: vec3<f32> = clamp(
            numerator / denominator, 
            vec3<f32>(0.0f, 0.0f, 0.0f), 
            vec3<f32>(1.0f, 1.0f, 1.0f));

        kS = 1.0f - fresnel.x;
        out.radianceOutput = vec4<f32>(
            specular * radiance * 5.0f * max(dot(normal.xyz, lightDir.xyz), 0.0f),
            kS
        );
    }

    return out;
}

/////
fn uniformSampling(
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
fn cosineHemisphereSampling(
    normal: vec3<f32>,
    fRand0: f32,
    fRand1: f32) -> vec3<f32>
{
    let fPhi: f32 = 2.0f * PI * fRand1;
    let fTheta: f32 = acos(sqrt(fRand0));
    let h: vec3<f32> = vec3<f32>(
        sin(fTheta) * cos(fPhi),
        sin(fTheta) * sin(fPhi),
        cos(fTheta)
    );

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

/////
fn distributionGGX(
    N: vec3<f32>,
    H: vec3<f32>,
    roughness: f32) -> f32
{
    let a: f32 = roughness * roughness;
    let a2: f32 = a * a;
    let NdotH: f32 = max(dot(N, H), 0.0f);
    let NdotH2: f32 = NdotH * NdotH;

    let num: f32 = a2;
    var denom: f32 = (NdotH2 * (a2 - 1.0f) + 1.0f);
    denom = PI * denom * denom;

    return num / denom;
}

/////
fn geometrySchlickGGX(
    NdotV: f32,
    roughness: f32) -> f32
{
    let r: f32 = (roughness + 1.0f);
    let k: f32 = (r * r) / 8.0f;

    let num: f32 = NdotV;
    let denom: f32 = NdotV * (1.0f - k) + k;

    return num / denom;
}


/////
fn geometrySmith(
    N: vec3<f32>,
    V: vec3<f32>,
    L: vec3<f32>,
    roughness: f32) -> f32
{
    let NdotV: f32 = max(dot(N, V), 0.0f);
    let NdotL: f32 = max(dot(N, L), 0.0f);
    let ggx2: f32 = geometrySchlickGGX(NdotV, roughness);
    let ggx1: f32 = geometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}

/////
fn fresnelShlickRoughness(
    fCosTheta: f32,
    f0: vec3<f32>,
    fRoughness: f32) -> vec3<f32>
{
    let fOneMinusRoughness: f32 = 1.0f - fRoughness;
    let fClamp: f32 = clamp(1.0f - fCosTheta, 0.0f, 1.0f);
    return f0 + (max(vec3<f32>(fOneMinusRoughness, fOneMinusRoughness, fOneMinusRoughness), f0) - f0) * pow(fClamp, 5.0f);
}

/////
fn sampleGGXVNDF(
    viewDirection: vec3<f32>,
    fRoughness: f32,
    fRand0: f32,
    fRand1: f32) -> vec3<f32>
{
    // Section 3.2: transforming the view direction to the hemisphere configuration
    let Vh: vec3<f32> = normalize(vec3<f32>(fRoughness * viewDirection.x, fRoughness * viewDirection.y, viewDirection.z));
    // Section 4.1: orthonormal basis (with special case if cross product is zero)
    let lensq: f32 = Vh.x * Vh.x + Vh.y * Vh.y;
    var T1: vec3<f32> = vec3<f32>(1.0f, 0.0f, 0.0f);
    if(lensq > 0)
    {
        T1 = vec3<f32>(-Vh.y, Vh.x, 0) * (1.0f / sqrt(lensq));
    }

    var T2: vec3<f32> = cross(Vh, T1);
    // Section 4.2: parameterization of the projected area
    let r: f32 = sqrt(fRand0);
    let phi: f32 = 2.0f * PI * fRand1;
    let t1: f32 = r * cos(phi);
    var t2: f32 = r * sin(phi);
    let s: f32 = 0.5f * (1.0f + Vh.z);
    t2 = (1.0f - s)*sqrt(1.0f - t1*t1) + s*t2;
    // Section 4.3: reprojection onto hemisphere
    let Nh: vec3<f32> = t1*T1 + t2*T2 + sqrt(max(0.0f, 1.0f - t1*t1 - t2*t2))*Vh;
    // Section 3.4: transforming the normal back to the ellipsoid configuration
    let Ne: vec3<f32> = normalize(vec3<f32>(fRoughness * Nh.x, fRoughness * Nh.y, max(0.0, Nh.z)));
    return Ne;
}

/////
fn extractDominantLightDirectionFromSphericalHarmonicCoefficients(
    texCoord: vec2<f32>
) -> vec3<f32>
{
    let iOutputX: i32 = i32(texCoord.x * f32(defaultUniformData.miScreenWidth));
    let iOutputY: i32 = i32(texCoord.y * f32(defaultUniformData.miScreenHeight));
    let iImageIndex: i32 = iOutputY * defaultUniformData.miScreenWidth + iOutputX;

    let SHCoefficent0: vec4<f32> = sphericalHarmonicCoefficient0[iImageIndex];
    let SHCoefficent1: vec4<f32> = sphericalHarmonicCoefficient1[iImageIndex];
    let SHCoefficent2: vec4<f32> = sphericalHarmonicCoefficient2[iImageIndex];

    //let L1: vec3<f32> = vec3<f32>(SHCoefficent0.w, SHCoefficent1.x, SHCoefficent1.y);
    //let L2: vec3<f32> = vec3<f32>(SHCoefficent1.z, SHCoefficent1.w, SHCoefficent2.x);
    //let L3: vec3<f32> = vec3<f32>(SHCoefficent2.y, SHCoefficent2.z, SHCoefficent2.w);

    let L1: vec3<f32> = vec3<f32>(SHCoefficent0.w, SHCoefficent1.z, SHCoefficent2.y);
    let L2: vec3<f32> = vec3<f32>(SHCoefficent1.x, SHCoefficent1.w, SHCoefficent2.z);
    let L3: vec3<f32> = vec3<f32>(SHCoefficent2.y, SHCoefficent2.x, SHCoefficent2.w);

    var ret: vec3<f32> = vec3<f32>(
        -L3.x * 0.3f + -L3.y * 0.59f + -L3.z * 0.11f,
        -L1.x * 0.3f + -L1.y * 0.59f + -L1.z * 0.11f,
        L2.x * 0.3f + L2.y * 0.59f + L2.z * 0.11f
    );

    return normalize(ret.yzx);
}