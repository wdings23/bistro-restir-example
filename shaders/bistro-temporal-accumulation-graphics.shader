const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const kfMaxBlendFrames = 60.0f;
const kfOneOverMaxBlendFrames: f32 = 1.0f / kfMaxBlendFrames;

const INDIRECT_DIFFUSE: i32 = 0;

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
    @location(0) indirectDiffuseOutput: vec4<f32>,
    @location(1) indirectDiffuseMomentOutput: vec4<f32>,
    @location(2) directSunOutput: vec4<f32>,
    @location(3) specularOutput: vec4<f32>,
    @location(4) debugOutput: vec4<f32>,
};

struct RandomResult 
{
    mfNum: f32,
    miSeed: u32,
};

struct UniformData
{
    miFrame: u32,
    miScreenWidth: u32,
    miScreenHeight: u32,
    mfRand: f32,

    maSamples: array<vec4<f32>, 64>,

    mfRand0: f32,
    mfRand1: f32,
    mfRand2: f32,
    mfRand3: f32,

};

struct DisocclusionResult
{
    mBackProjectScreenCoord: vec2<i32>,
    mBackProjectScreenUV: vec2<f32>,
    mbDisoccluded: bool,
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

struct MIPTextureResult
{
    mIndirectDiffuseRadiance: vec3<f32>,
    mIndirectDiffuseRadianceHistory: vec4<f32>,
    mfAO: f32,
};

@group(0) @binding(0)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var indirectDiffuseRadianceTexture: texture_2d<f32>;

@group(0) @binding(3)
var directSunRadianceTexture: texture_2d<f32>;

@group(0) @binding(4)
var specularRadianceTexture: texture_2d<f32>;

@group(0) @binding(5)
var indirectDiffuseRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(6)
var indirectDiffuseMomentHistoryTexture: texture_2d<f32>;

@group(0) @binding(7)
var directSunRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(8)
var specularRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(9)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(10)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(11)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(12)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(13)
var rayCountTexture: texture_2d<f32>;



@group(0) @binding(14)
var indirectDiffuseMIPTexture0: texture_2d<f32>;

@group(0) @binding(15)
var indirectDiffuseMIPTexture1: texture_2d<f32>;

@group(0) @binding(16)
var indirectDiffuseMIPTexture2: texture_2d<f32>;

@group(0) @binding(17)
var indirectDiffuseMIPTexture3: texture_2d<f32>;



@group(0) @binding(18)
var indirectDiffuseHistoryMIPTexture0: texture_2d<f32>;

@group(0) @binding(19)
var indirectDiffuseHistoryMIPTexture1: texture_2d<f32>;

@group(0) @binding(20)
var indirectDiffuseHistoryMIPTexture2: texture_2d<f32>;

@group(0) @binding(21)
var indirectDiffuseHistoryMIPTexture3: texture_2d<f32>;



@group(0) @binding(22)
var ambientOcclusionMIPTexture0: texture_2d<f32>;

@group(0) @binding(23)
var ambientOcclusionMIPTexture1: texture_2d<f32>;

@group(0) @binding(24)
var ambientOcclusionMIPTexture2: texture_2d<f32>;

@group(0) @binding(25)
var ambientOcclusionMIPTexture3: texture_2d<f32>;




@group(0) @binding(26)
var textureSampler: sampler;

@group(0) @binding(23)
var linearTextureSampler: sampler;

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

    let fOneOverScreenWidth: f32 = 1.0f / f32(defaultUniformData.miScreenWidth);
    let fOneOverScreenHeight: f32 = 1.0f / f32(defaultUniformData.miScreenHeight);

    let worldPosition: vec4<f32> = textureSample(
        worldPositionTexture, 
        textureSampler, 
        in.texCoord);

    if(worldPosition.w <= 0.0f)
    {
        out.indirectDiffuseOutput = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        out.indirectDiffuseMomentOutput = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
        return out;
    }

    var indirectDiffuseRadiance: vec3<f32> = textureSample(
        indirectDiffuseRadianceTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    var directSunRadiance: vec3<f32> = textureSample(
        directSunRadianceTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    var specularRadiance: vec3<f32> = textureSample(
        specularRadianceTexture,
        textureSampler,
        in.texCoord
    ).xyz;

    // previous radiance
    let prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);
    
    let afWeights: array<f32, 4> = array<f32, 4>(1.0f, 1.0f, 1.0f, 1.0f);
    var indirectDiffuseRadianceHistory: vec4<f32> = textureSample(
        indirectDiffuseRadianceHistoryTexture,
        textureSampler,
        prevScreenUV
    );
    let directSunRadianceHistory: vec4<f32> = textureSample(
        directSunRadianceHistoryTexture,
        textureSampler,
        prevScreenUV                    
    );
    let specularRadianceHistory: vec4<f32> = textureSample(
        specularRadianceHistoryTexture,
        textureSampler,
        prevScreenUV
    );

    let fLuminance: f32 = computeLuminance(indirectDiffuseRadiance.xyz);
    let indirectDiffuseMoment: vec4<f32> = vec4<f32>(
        fLuminance,
        fLuminance * fLuminance,
        0.0f, 
        0.0f
    );

    // ray counts for ambient occlusion
    let rayCount: vec4<f32> = textureSample(
        rayCountTexture,
        textureSampler,
        prevScreenUV
    );

    // ambient occlusion
    var fAO: f32 = smoothstep(0.0f, 1.0f, rayCount.z);
    
    let indirectDiffuseMomentHistory: vec4<f32> = textureSample(
        indirectDiffuseMomentHistoryTexture,
        textureSampler,
        prevScreenUV
    );

    // accumulation weight
    var fAccumulationBlendWeight: f32 = 1.0f / 10.0f;

    // range for box clamp
    var minIndirectRadiance: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    var maxIndirectRadiance: vec3<f32> = vec3<f32>(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            let sampleUV: vec2<f32> = vec2<f32>(
                in.texCoord.x + f32(iX) * fOneOverScreenWidth,
                in.texCoord.y + f32(iY) * fOneOverScreenHeight
            );

            let radiance: vec3<f32> = textureSample(
                indirectDiffuseRadianceTexture,
                textureSampler,
                sampleUV
            ).xyz;

            minIndirectRadiance = min(minIndirectRadiance, radiance);
            maxIndirectRadiance = max(maxIndirectRadiance, radiance);
        }
    }

    // small history size, use correct higher MIP for more samples 
    if(indirectDiffuseRadianceHistory.w <= 10.0f)
    {
        let iMIPLevel: i32 = i32(
            (1.0f - clamp(indirectDiffuseRadianceHistory.w / 10.0f, 0.0f, 1.0f)) * 3.0f
        );

        // let ret: MIPTextureResult = sampleMIPTexture(
        //     in.texCoord, 
        //     prevScreenUV,
        //     iMIPLevel, 
        //     fOneOverScreenWidth,
        //     fOneOverScreenHeight);
        // indirectDiffuseRadiance = clamp(ret.mIndirectDiffuseRadiance, minIndirectRadiance, maxIndirectRadiance);
        // indirectDiffuseRadianceHistory = ret.mIndirectDiffuseRadianceHistory;
        
        indirectDiffuseRadiance = textureSample(
            indirectDiffuseRadianceTexture,
            textureSampler,
            in.texCoord
        ).xyz;
        
        indirectDiffuseRadianceHistory = textureSample(
            indirectDiffuseRadianceHistoryTexture,
            textureSampler,
            in.texCoord 
        );

        indirectDiffuseRadiance = clamp(indirectDiffuseRadiance, minIndirectRadiance, maxIndirectRadiance);
        
        //fAO = ret.mfAO;
    }

    // disoccluded
    let bDisoccluded: bool = isDisoccluded2(in.texCoord, prevScreenUV);
    if(bDisoccluded)
    {
        //indirectDiffuseRadiance = textureSample(
        //    indirectDiffuseRadianceTexture,
        //    textureSampler,
        //    in.texCoord
        //).xyz;

        out.indirectDiffuseOutput = vec4<f32>(indirectDiffuseRadiance, 0.0f);
        out.indirectDiffuseMomentOutput = indirectDiffuseMoment;
        out.directSunOutput = vec4<f32>(directSunRadiance, 0.0f);
        out.specularOutput = vec4<f32>(specularRadiance, 0.0f);
        out.debugOutput = vec4<f32>(fAO, fAO, fAO, 1.0f);

        return out;
    }

    var minDirectSunRadiance: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    var maxDirectSunRadiance: vec3<f32> = vec3<f32>(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            let sampleUV: vec2<f32> = vec2<f32>(
                in.texCoord.x + f32(iX) * fOneOverScreenWidth,
                in.texCoord.y + f32(iY) * fOneOverScreenHeight
            );

            let radiance: vec3<f32> = textureSample(
                directSunRadianceTexture,
                textureSampler,
                sampleUV
            ).xyz;

            minDirectSunRadiance = min(minDirectSunRadiance, radiance);
            maxDirectSunRadiance = max(maxDirectSunRadiance, radiance);
        }
    }

    var minSpecularRadiance: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    var maxSpecularRadiance: vec3<f32> = vec3<f32>(-FLT_MAX, -FLT_MAX, -FLT_MAX);
    for(var iY: i32 = -1; iY <= 1; iY++)
    {
        for(var iX: i32 = -1; iX <= 1; iX++)
        {
            let sampleUV: vec2<f32> = vec2<f32>(
                in.texCoord.x + f32(iX) * fOneOverScreenWidth,
                in.texCoord.y + f32(iY) * fOneOverScreenHeight
            );

            let radiance: vec3<f32> = textureSample(
                specularRadianceTexture,
                textureSampler,
                sampleUV
            ).xyz;

            minSpecularRadiance = min(minSpecularRadiance, radiance);
            maxSpecularRadiance = max(maxSpecularRadiance, radiance);
        }
    }

    // mix previous to current radiance
    let mixedIndirectDiffuseRadiance: vec3<f32> = mix(
        clamp(indirectDiffuseRadianceHistory.xyz, minIndirectRadiance, maxIndirectRadiance),
        indirectDiffuseRadiance,
        fAccumulationBlendWeight
    );

    let mixedDirectSunRadiance: vec3<f32> = mix(
        clamp(directSunRadianceHistory.xyz, minDirectSunRadiance, maxDirectSunRadiance),
        directSunRadiance.xyz,
        fAccumulationBlendWeight
    );

    let mixedSpecularRadiance: vec3<f32> = mix(
        clamp(specularRadianceHistory.xyz, minSpecularRadiance, maxSpecularRadiance),
        specularRadiance.xyz,
        fAccumulationBlendWeight
    );

    let mixedIndirectDiffuseMoment: vec3<f32> = mix(
        indirectDiffuseMomentHistory.xyz,
        indirectDiffuseMoment.xyz,
        fAccumulationBlendWeight
    );

    out.indirectDiffuseOutput.x = mixedIndirectDiffuseRadiance.x;
    out.indirectDiffuseOutput.y = mixedIndirectDiffuseRadiance.y;
    out.indirectDiffuseOutput.z = mixedIndirectDiffuseRadiance.z;
    out.indirectDiffuseOutput.w = indirectDiffuseRadianceHistory.w + 1.0f;

    out.directSunOutput.x = mixedDirectSunRadiance.x;
    out.directSunOutput.y = mixedDirectSunRadiance.y;
    out.directSunOutput.z = mixedDirectSunRadiance.z;
    out.directSunOutput.w = directSunRadianceHistory.w + 1.0f;

    out.specularOutput.x = mixedSpecularRadiance.x;
    out.specularOutput.y = mixedSpecularRadiance.y;
    out.specularOutput.z = mixedSpecularRadiance.z;
    out.specularOutput.w = specularRadianceHistory.w + 1.0f;

    // moment with variance in z component
    out.indirectDiffuseMomentOutput = vec4<f32>(
        mixedIndirectDiffuseMoment.x,
        mixedIndirectDiffuseMoment.y,
        abs(mixedIndirectDiffuseMoment.y - mixedIndirectDiffuseMoment.x * mixedIndirectDiffuseMoment.x),
        indirectDiffuseMomentHistory.w + 1.0f
    );

    out.debugOutput = vec4<f32>(fAO, fAO, fAO, 1.0f);

    return out;
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
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
fn isDisoccluded2(
    screenUV: vec2<f32>,
    prevScreenUV: vec2<f32>
) -> bool
{
    var worldPosition: vec3<f32> = textureSample(
        worldPositionTexture,
        textureSampler,
        screenUV).xyz;

    var prevWorldPosition: vec3<f32> = textureSample(
        prevWorldPositionTexture,
        textureSampler,
        prevScreenUV).xyz;

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

    return !(iMesh == iPrevMesh && fCheckDepth <= 0.01f && fCheckWorldPositionDistance <= 0.01f && fCheckDP >= 0.99f);
}

///// 
fn isPrevUVOutOfBounds(inputTexCoord: vec2<f32>) -> bool
{
    var motionVector: vec4<f32> = textureSample(
        motionVectorTexture,
        textureSampler,
        inputTexCoord);
    let backProjectedScreenUV: vec2<f32> = inputTexCoord - motionVector.xy;

    return (backProjectedScreenUV.x < 0.0f || backProjectedScreenUV.x > 1.0 || backProjectedScreenUV.y < 0.0f || backProjectedScreenUV.y > 1.0f);
}

/////
fn sampleMIPTexture(
    uv: vec2<f32>, 
    prevScreenUV: vec2<f32>,
    iMIP: i32, 
    fOneOverScreenWidth: f32,
    fOneOverScreenHeight: f32) -> MIPTextureResult
{
    var indirectDiffuseRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var indirectDiffuseRadianceHistory: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    var fAO = 0.0f;
    var fHistoryCount = 0.0f;

    let sampleUV0: vec2<f32> = uv;
    let prevSampleUV0: vec2<f32> = prevScreenUV;
    let sampleUV1: vec2<f32> = uv + vec2<f32>(0.0f, -fOneOverScreenHeight);
    let prevSampleUV1: vec2<f32> = prevScreenUV + vec2<f32>(0.0f, -fOneOverScreenHeight);
    let sampleUV2: vec2<f32> = uv + vec2<f32>(-fOneOverScreenWidth, 0.0f);
    let prevSampleUV2: vec2<f32> = prevScreenUV + vec2<f32>(-fOneOverScreenWidth, 0.0f);
    let sampleUV3: vec2<f32> = uv + vec2<f32>(fOneOverScreenWidth, 0.0f);
    let prevSampleUV3: vec2<f32> = prevScreenUV + vec2<f32>(fOneOverScreenWidth, 0.0f);
    let sampleUV4: vec2<f32> = uv + vec2<f32>(0.0f, fOneOverScreenHeight);
    let prevSampleUV4: vec2<f32> = prevScreenUV + vec2<f32>(0.0f, fOneOverScreenHeight);

    if(iMIP == 0)
    {
        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture0,
            textureSampler,
            sampleUV0
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture0,
            textureSampler,
            prevSampleUV0
        );
        fHistoryCount = indirectDiffuseRadianceHistory.w;
        fAO += textureSample(
            ambientOcclusionMIPTexture0,
            textureSampler,
            sampleUV0
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture0,
            textureSampler,
            sampleUV1
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture0,
            textureSampler,
            prevSampleUV1
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture0,
            textureSampler,
            sampleUV1
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture0,
            textureSampler,
            sampleUV2
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture0,
            textureSampler,
            prevSampleUV2
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture0,
            textureSampler,
            sampleUV2
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture0,
            textureSampler,
            sampleUV3
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture0,
            textureSampler,
            prevSampleUV3
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture0,
            textureSampler,
            sampleUV3
        ).z;
    }
    else if(iMIP == 1)
    {
        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture1,
            textureSampler,
            sampleUV0
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture1,
            textureSampler,
            prevSampleUV0
        );
        fHistoryCount = indirectDiffuseRadianceHistory.w;
        fAO += textureSample(
            ambientOcclusionMIPTexture1,
            textureSampler,
            sampleUV0
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture1,
            textureSampler,
            sampleUV1
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture1,
            textureSampler,
            prevSampleUV1
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture1,
            textureSampler,
            sampleUV1
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture1,
            textureSampler,
            sampleUV2
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture1,
            textureSampler,
            prevSampleUV2
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture1,
            textureSampler,
            sampleUV2
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture1,
            textureSampler,
            sampleUV3
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture1,
            textureSampler,
            prevSampleUV3
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture1,
            textureSampler,
            sampleUV3
        ).z;
    }
    else if(iMIP == 2)
    {
        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture2,
            textureSampler,
            sampleUV0
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture2,
            textureSampler,
            prevSampleUV0
        );
        fHistoryCount = indirectDiffuseRadianceHistory.w;
        fAO += textureSample(
            ambientOcclusionMIPTexture2,
            textureSampler,
            sampleUV0
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture2,
            textureSampler,
            sampleUV1
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture2,
            textureSampler,
            prevSampleUV1
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture2,
            textureSampler,
            sampleUV1
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture2,
            textureSampler,
            sampleUV2
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture2,
            textureSampler,
            prevSampleUV2
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture2,
            textureSampler,
            sampleUV2
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture2,
            textureSampler,
            sampleUV3
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture2,
            textureSampler,
            prevSampleUV3
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture2,
            textureSampler,
            sampleUV3
        ).z;
    }
    else if(iMIP == 3)
    {
        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture3,
            textureSampler,
            sampleUV0
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture3,
            textureSampler,
            prevSampleUV0
        );
        fHistoryCount = indirectDiffuseRadianceHistory.w;
        fAO += textureSample(
            ambientOcclusionMIPTexture3,
            textureSampler,
            sampleUV0
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture3,
            textureSampler,
            sampleUV1
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture3,
            textureSampler,
            prevSampleUV1
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture3,
            textureSampler,
            sampleUV1
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture3,
            textureSampler,
            sampleUV2
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture3,
            textureSampler,
            prevSampleUV2
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture3,
            textureSampler,
            sampleUV2
        ).z;

        indirectDiffuseRadiance += textureSample(
            indirectDiffuseMIPTexture3,
            textureSampler,
            sampleUV3
        ).xyz;
        indirectDiffuseRadianceHistory += textureSample(
            indirectDiffuseHistoryMIPTexture3,
            textureSampler,
            prevSampleUV3
        );
        fAO += textureSample(
            ambientOcclusionMIPTexture3,
            textureSampler,
            sampleUV3
        ).z;
    }

    indirectDiffuseRadiance *= 0.25f;
    indirectDiffuseRadianceHistory *= 0.25f;
    indirectDiffuseRadianceHistory.w = fHistoryCount;
    fAO *= 0.25f;

    var ret: MIPTextureResult;
    ret.mIndirectDiffuseRadiance = indirectDiffuseRadiance;
    ret.mIndirectDiffuseRadianceHistory = indirectDiffuseRadianceHistory;
    ret.mfAO = fAO;

    return ret;
}
