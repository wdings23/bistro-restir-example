const UINT32_MAX: u32 = 0xffffffffu;
const FLT_MAX: f32 = 1.0e+10;
const PI: f32 = 3.14159f;
const kfMaxBlendFrames = 60.0f;
const kfOneOverMaxBlendFrames: f32 = 1.0f / kfMaxBlendFrames;

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
    @location(3) directSunMomentOutput: vec4<f32>,

    @location(4) emissiveOutput: vec4<f32>,
    @location(5) emissiveMomentOutput: vec4<f32>,

    @location(6) specularOutput: vec4<f32>,
    @location(7) specularMomentOutput: vec4<f32>,
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

@group(0) @binding(0)
var worldPositionTexture: texture_2d<f32>;

@group(0) @binding(1)
var normalTexture: texture_2d<f32>;

@group(0) @binding(2)
var indirectDiffuseRadianceTexture: texture_2d<f32>;

@group(0) @binding(3)
var indirectDiffuseRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(4)
var indirectDiffuseMomentHistoryTexture: texture_2d<f32>;

@group(0) @binding(5)
var ambientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(6)
var prevWorldPositionTexture: texture_2d<f32>;

@group(0) @binding(7)
var prevNormalTexture: texture_2d<f32>;

@group(0) @binding(8)
var motionVectorTexture: texture_2d<f32>;

@group(0) @binding(9)
var prevMotionVectorTexture: texture_2d<f32>;

@group(0) @binding(10)
var directSunLightTexture: texture_2d<f32>;

@group(0) @binding(11)
var emissiveLightTexture: texture_2d<f32>;

@group(0) @binding(12)
var specularRadianceTexture: texture_2d<f32>;

@group(0) @binding(13)
var directSunLightHistoryTexture: texture_2d<f32>;

@group(0) @binding(14)
var emissiveLightHistoryTexture: texture_2d<f32>;

@group(0) @binding(15)
var emissiveLightMomentHistoryTexture: texture_2d<f32>;

@group(0) @binding(16)
var specularRadianceHistoryTexture: texture_2d<f32>;

@group(0) @binding(17)
var specularMomentHistoryTexture: texture_2d<f32>;

@group(0) @binding(18)
var prevFilteredAmbientOcclusionTexture: texture_2d<f32>;

@group(0) @binding(19)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> materialData: array<vec4<f32>, 32>;

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

    let normal: vec4<f32> = textureSample(
        normalTexture,
        textureSampler,
        in.texCoord);

    var randomResult: RandomResult = initRand(
        u32(in.texCoord.x * 100.0f + in.texCoord.y * 200.0f) + u32(uniformData.mfRand0 * 100.0f),
        u32(in.pos.x * 10.0f + in.pos.z * 20.0f) + u32(uniformData.mfRand1 * 100.0f),
        10u);

    let fMaxPlaneDistance: f32 = 2.0f;

    let fCenterDepth: f32 = fract(worldPosition.w);
    let fPlaneD: f32 = -(dot(normal.xyz, worldPosition.xyz));

    let oneOverScreenDimensions: vec2<f32> = vec2<f32>(
        1.0f / f32(uniformData.miScreenWidth),
        1.0f / f32(uniformData.miScreenHeight)
    );

    var prevScreenUV: vec2<f32> = getPreviousScreenUV(in.texCoord);

    // sample radiance from surrounding neighbor box
    var fTotalWeight: f32 = 0.0f;
    var iValidSampleCount: u32 = 0u;
    
    var totalIndirectDiffuseRadianceHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var totalDirectSunLightHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var totalEmissiveLightHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var totalSpecularRadianceHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);

    var centerIndirectDiffuseRadianceHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var centerIndirectDiffuseRadiance: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var fCenterIndirectDiffuseHistory: f32 = 0.0f;

    var fTotalIndirectDiffuseHistoryLuminance: f32 = 0.0f;
    var fTotalIndirectDiffuseHistoryLuminanceSquared: f32 = 0.0f;

    var fTotalIndirectDiffuseLuminance: f32 = 0.0f;
    var fTotalIndirectDiffuseLuminanceSquared: f32 = 0.0f;

    var fTotalIndirectDiffuseMomentWeights: f32 = 0.0f;

    var fTotalEmissiveHistoryLuminance: f32 = 0.0f;
    var fTotalEmissiveHistoryLuminanceSquared: f32 = 0.0f;
    var fEmissiveHistoryCount: f32 = 0.0f;

    var fTotalEmissiveLuminance: f32 = 0.0f;
    var fTotalEmissiveLuminanceSquared: f32 = 0.0f;

    var fTotalDirectSunHistoryLuminance: f32 = 0.0f;
    var fTotalDirectSunHistoryLuminanceSquared: f32 = 0.0f;

    var fTotalDirectSunLuminance: f32 = 0.0f;
    var fTotalDirectSunLuminanceSquared: f32 = 0.0f;

    var aMixedIndirectDiffuseRadianceHistory: array<vec3<f32>, 25>;
    var aMixedDirectSunLightHistory: array<vec3<f32>, 25>;
    var aMixedEmissiveLightHistory: array<vec3<f32>, 25>;
    var aMixedSpecularRadianceHistory: array<vec3<f32>, 25>; 

    var fAOHistory: f32 = 0.0f;

    for(var iOffsetY: i32 = -2; iOffsetY <= 2; iOffsetY++)
    {
        let fOffsetY: f32 = f32(iOffsetY) / f32(uniformData.miScreenHeight);
        for(var iOffsetX: i32 = -2; iOffsetX <= 2; iOffsetX++)
        {
            let fOffsetX: f32 = f32(iOffsetX) / f32(uniformData.miScreenWidth);
            var sampleOffset: vec2<f32> = vec2<f32>(fOffsetX, fOffsetY);
            let prevFrameSampleCoord: vec2<f32> = clamp(
                prevScreenUV + sampleOffset, 
                vec2<f32>(0.0f, 0.0f), 
                vec2<f32>(1.0f, 1.0f));

            let indirectDiffuseRadianceHistory: vec4<f32> = textureSample(
                indirectDiffuseRadianceHistoryTexture,
                textureSampler,
                prevFrameSampleCoord);

            let directSunLightHistory: vec4<f32> = textureSample(
                directSunLightHistoryTexture,
                textureSampler,
                prevFrameSampleCoord);

            let emissiveLightHistory: vec4<f32> = textureSample(
                emissiveLightHistoryTexture,
                textureSampler,
                prevFrameSampleCoord);

            let specularRadianceHistory: vec4<f32> = textureSample(
                specularRadianceHistoryTexture,
                textureSampler,
                prevFrameSampleCoord);

            let sampleWorldPosition: vec4<f32> = textureSample(
                prevWorldPositionTexture, 
                textureSampler, 
                prevFrameSampleCoord);

            let sampleNormal: vec4<f32> = textureSample(
                prevNormalTexture,
                textureSampler,
                prevFrameSampleCoord);

            if(sampleWorldPosition.w <= 0.0f)
            {
                continue;
            }

            let fSampleDepth: f32 = fract(sampleWorldPosition.w);
            let fPlaneDistanceWeight: f32 = clamp(fMaxPlaneDistance - (dot(sampleWorldPosition.xyz, normal.xyz) + fPlaneD), 0.0f, 1.0f); 
            let fNormalWeight: f32 = pow(max(dot(sampleNormal.xyz, normal.xyz), 0.0f), 10.0f);
            let fDepthWeight: f32 = exp(-abs(fCenterDepth - fSampleDepth) / (0.5f + 0.0001f));
            
            let fSampleWeight = fNormalWeight * fDepthWeight * fPlaneDistanceWeight;

            let totalIndirectDiffuseRadiance: vec3<f32> = indirectDiffuseRadianceHistory.xyz * fSampleWeight;
            let totalDirectSunRadiance: vec3<f32> = directSunLightHistory.xyz * fSampleWeight;
            let totalEmissiveRadiance: vec3<f32> = emissiveLightHistory.xyz * fSampleWeight;
            let totalSpecularRadiance: vec3<f32> = specularRadianceHistory.xyz * fSampleWeight;

            totalIndirectDiffuseRadianceHistory.x += totalIndirectDiffuseRadiance.x;
            totalIndirectDiffuseRadianceHistory.y += totalIndirectDiffuseRadiance.y;
            totalIndirectDiffuseRadianceHistory.z += totalIndirectDiffuseRadiance.z;

            totalDirectSunLightHistory.x += totalDirectSunRadiance.x;
            totalDirectSunLightHistory.y += totalDirectSunRadiance.y;
            totalDirectSunLightHistory.z += totalDirectSunRadiance.z;

            totalEmissiveLightHistory.x += totalEmissiveRadiance.x;
            totalEmissiveLightHistory.y += totalEmissiveRadiance.y;
            totalEmissiveLightHistory.z += totalEmissiveRadiance.z;

            totalSpecularRadianceHistory.x += totalSpecularRadiance.x;
            totalSpecularRadianceHistory.y += totalSpecularRadiance.y;
            totalSpecularRadianceHistory.z += totalSpecularRadiance.z;

            aMixedIndirectDiffuseRadianceHistory[iValidSampleCount] = totalIndirectDiffuseRadiance;
            aMixedDirectSunLightHistory[iValidSampleCount] = totalDirectSunRadiance;
            aMixedEmissiveLightHistory[iValidSampleCount] = totalEmissiveRadiance;
            aMixedSpecularRadianceHistory[iValidSampleCount] = totalSpecularRadiance;

            fTotalWeight += fSampleWeight;

            let currFrameSampleCoord: vec2<f32> = clamp(
                in.texCoord + sampleOffset,
                vec2<f32>(0.0f, 0.0f),
                vec2<f32>(1.0f, 1.0f)
            ); 
            if(iOffsetY == 0 && iOffsetX == 0)
            {
                centerIndirectDiffuseRadiance = textureSample(
                    indirectDiffuseRadianceTexture,
                    textureSampler,
                    currFrameSampleCoord).xyz;

                fAOHistory = textureSample(
                    directSunLightHistoryTexture,
                    textureSampler,
                    currFrameSampleCoord
                ).w;

                centerIndirectDiffuseRadiance = indirectDiffuseRadianceHistory.xyz;                
                fCenterIndirectDiffuseHistory = indirectDiffuseRadianceHistory.w;
            }

            var fMomentWeight: f32 = exp(-3.0f * f32(iOffsetX * iOffsetX + iOffsetY * iOffsetY) / f32(3 * 3));

            var fIndirectDiffuseLuminanceHistory: f32 = computeLuminance(indirectDiffuseRadianceHistory.xyz) ;
            fTotalIndirectDiffuseHistoryLuminance += fIndirectDiffuseLuminanceHistory * fMomentWeight;
            fTotalIndirectDiffuseHistoryLuminanceSquared += (fIndirectDiffuseLuminanceHistory * fIndirectDiffuseLuminanceHistory) * fMomentWeight;

            var indirectDiffuseRadiance: vec3<f32> = textureSample(
                indirectDiffuseRadianceTexture,
                textureSampler,
                currFrameSampleCoord
            ).xyz;

            var fIndirectDiffuseLuminance: f32 = computeLuminance(indirectDiffuseRadiance.xyz);
            fTotalIndirectDiffuseLuminance += fIndirectDiffuseLuminance * fMomentWeight;
            fTotalIndirectDiffuseLuminanceSquared += fIndirectDiffuseLuminance * fIndirectDiffuseLuminance * fMomentWeight;
            
            // emissive
            {
                fEmissiveHistoryCount = emissiveLightHistory.w;

                var fEmissiveHistoryLuminance: f32 = computeLuminance(emissiveLightHistory.xyz);
                fTotalEmissiveHistoryLuminance += fEmissiveHistoryLuminance * fMomentWeight;
                fTotalEmissiveHistoryLuminanceSquared += fEmissiveHistoryLuminance * fEmissiveHistoryLuminance * fMomentWeight;

                var emissiveRadiance: vec3<f32> = textureSample(
                    emissiveLightTexture,
                    textureSampler,
                    currFrameSampleCoord
                ).xyz;

                var fEmissiveLuminance: f32 = computeLuminance(emissiveRadiance.xyz);
                fTotalEmissiveLuminance += fEmissiveLuminance * fMomentWeight;
                fTotalEmissiveLuminanceSquared += fEmissiveLuminance * fEmissiveLuminance * fMomentWeight;
            }

            // direct sun
            {
                var fTotalDirectSunHistoryLuminance: f32 = 0.0f;
                var fTotalDirectSunHistoryLuminanceSquared: f32 = 0.0f;

                var fTotalDirectSunLuminance: f32 = 0.0f;
                var fTotalDirectSunLuminanceSquared: f32 = 0.0f;

                fEmissiveHistoryCount = emissiveLightHistory.w;

                var fDirectSunHistoryLuminance: f32 = computeLuminance(directSunLightHistory.xyz);
                fTotalDirectSunHistoryLuminance += fDirectSunHistoryLuminance * fMomentWeight;
                fTotalDirectSunHistoryLuminanceSquared += fDirectSunHistoryLuminance * fDirectSunHistoryLuminance * fMomentWeight;

                var directSunLightRadiance: vec3<f32> = textureSample(
                    directSunLightTexture,
                    textureSampler,
                    currFrameSampleCoord
                ).xyz;

                var fDirectSunLuminance: f32 = computeLuminance(directSunLightRadiance.xyz);
                fTotalDirectSunLuminance += fDirectSunLuminance * fMomentWeight;
                fTotalDirectSunLuminanceSquared += fDirectSunLuminance * fDirectSunLuminance * fMomentWeight;
            }

            fTotalIndirectDiffuseMomentWeights += fMomentWeight;
            iValidSampleCount += 1u;
        }
    }

    let fOneOverValidSampleCount: f32 = 1.0f / f32(iValidSampleCount);

    // indirect diffuse
    fTotalIndirectDiffuseHistoryLuminance /= fTotalIndirectDiffuseMomentWeights;
    fTotalIndirectDiffuseHistoryLuminanceSquared /= fTotalIndirectDiffuseMomentWeights;
    fTotalIndirectDiffuseLuminance /= fTotalIndirectDiffuseMomentWeights;
    fTotalIndirectDiffuseLuminanceSquared /= fTotalIndirectDiffuseMomentWeights;

    // emissive
    fTotalEmissiveHistoryLuminance /= fTotalIndirectDiffuseMomentWeights;
    fTotalEmissiveHistoryLuminanceSquared /= fTotalIndirectDiffuseMomentWeights;
    fTotalEmissiveLuminance /= fTotalIndirectDiffuseMomentWeights;
    fTotalEmissiveLuminanceSquared /= fTotalIndirectDiffuseMomentWeights;

    // direct sun
    fTotalDirectSunHistoryLuminance /= fTotalIndirectDiffuseMomentWeights;
    fTotalDirectSunHistoryLuminanceSquared /= fTotalIndirectDiffuseMomentWeights;
    fTotalDirectSunLuminance /= fTotalIndirectDiffuseMomentWeights;
    fTotalDirectSunLuminanceSquared /= fTotalIndirectDiffuseMomentWeights;

    var fTotalDiffuseHistoryLuminance: f32 = (
        fTotalIndirectDiffuseHistoryLuminance + 
        fTotalEmissiveHistoryLuminance + 
        fTotalDirectSunHistoryLuminance) / fTotalIndirectDiffuseMomentWeights;
    var fTotalDiffuseHistoryLuminanceSquared: f32 = (
        fTotalIndirectDiffuseHistoryLuminanceSquared + 
        fTotalEmissiveHistoryLuminanceSquared + 
        fTotalDirectSunHistoryLuminanceSquared) / fTotalIndirectDiffuseMomentWeights;
    var fTotalDiffuseLuminance: f32 = (
        fTotalIndirectDiffuseLuminance +
        fTotalEmissiveLuminance + 
        fTotalDirectSunLuminance) / fTotalIndirectDiffuseMomentWeights;
    var fTotalDiffuseLuminanceSquared: f32 = (
        fTotalIndirectDiffuseLuminanceSquared + 
        fTotalEmissiveLuminanceSquared + 
        fTotalDirectSunLuminanceSquared) / fTotalIndirectDiffuseMomentWeights;

    // mean
    let totalIndirectDiffuseHistoryMean: vec3<f32> = totalIndirectDiffuseRadianceHistory / fTotalWeight;
    let totalDirectSunLightHistoryMean: vec3<f32> = totalDirectSunLightHistory / fTotalWeight;
    let totalEmissiveLightHistoryMean: vec3<f32> = totalEmissiveLightHistory / fTotalWeight;
    let totalSpecularRadianceHistoryMean: vec3<f32> = totalSpecularRadianceHistory / fTotalWeight;

    // total radiance - mean squared
    var totalIndirectDiffuseHistoryDiffSquared: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var totalDirectSunLightHistoryDiffSquared: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var totalEmissiveLightHistoryDiffSquared: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var totalSpecularRadianceHistorySquared: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    for(var i: u32 = 0; i < iValidSampleCount; i++)
    {
        let indirectDiffuseDiff: vec3<f32> = aMixedIndirectDiffuseRadianceHistory[i] - totalIndirectDiffuseHistoryMean;
        let directSunLightDiff: vec3<f32> = aMixedDirectSunLightHistory[i] - totalDirectSunLightHistoryMean;
        let emissiveLightDiff: vec3<f32> = aMixedEmissiveLightHistory[i] - totalEmissiveLightHistoryMean;
        let specularDiff: vec3<f32> = aMixedSpecularRadianceHistory[i] - totalSpecularRadianceHistoryMean;
        
        totalIndirectDiffuseHistoryDiffSquared += indirectDiffuseDiff * indirectDiffuseDiff;
        totalDirectSunLightHistoryDiffSquared += directSunLightDiff * directSunLightDiff;
        totalEmissiveLightHistoryDiffSquared += emissiveLightDiff * emissiveLightDiff;
        totalSpecularRadianceHistorySquared += specularDiff * specularDiff;
    }
    // standard deviation 
    let stdDeviationIndirectDiffuseHistory: vec3<f32> = vec3<f32>(
        sqrt(totalIndirectDiffuseHistoryDiffSquared.x) * fOneOverValidSampleCount,
        sqrt(totalIndirectDiffuseHistoryDiffSquared.y) * fOneOverValidSampleCount,
        sqrt(totalIndirectDiffuseHistoryDiffSquared.z) * fOneOverValidSampleCount
    );
    let stdDeviationDirectSunHistory: vec3<f32> = vec3<f32>(
        sqrt(totalDirectSunLightHistoryDiffSquared.x) * fOneOverValidSampleCount,
        sqrt(totalDirectSunLightHistoryDiffSquared.y) * fOneOverValidSampleCount,
        sqrt(totalDirectSunLightHistoryDiffSquared.z) * fOneOverValidSampleCount
    );
    let stdDeviationEmissiveHistory: vec3<f32> = vec3<f32>(
        sqrt(totalEmissiveLightHistoryDiffSquared.x) * fOneOverValidSampleCount,
        sqrt(totalEmissiveLightHistoryDiffSquared.y) * fOneOverValidSampleCount,
        sqrt(totalEmissiveLightHistoryDiffSquared.z) * fOneOverValidSampleCount
    );
    let stdDeviationSpecularHistory: vec3<f32> = vec3<f32>(
        sqrt(totalSpecularRadianceHistorySquared.x) * fOneOverValidSampleCount,
        sqrt(totalSpecularRadianceHistorySquared.y) * fOneOverValidSampleCount,
        sqrt(totalSpecularRadianceHistorySquared.z) * fOneOverValidSampleCount
    );

    // min and max for clamping
    let minIndirectDiffuseRadianceHistory: vec3<f32> = max(
        totalIndirectDiffuseHistoryMean - stdDeviationIndirectDiffuseHistory,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );
    let maxIndirectDiffuseRadianceHistory: vec3<f32> = max(
        totalIndirectDiffuseHistoryMean + stdDeviationIndirectDiffuseHistory,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );

    let minDirectSunRadianceHistory: vec3<f32> = max(
        totalDirectSunLightHistoryMean - stdDeviationDirectSunHistory,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );
    let maxDirectSunRadianceHistory: vec3<f32> = max(
        totalDirectSunLightHistoryMean + stdDeviationDirectSunHistory,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );

    let minEmissiveRadianceHistory: vec3<f32> = max(
        totalEmissiveLightHistoryMean - stdDeviationEmissiveHistory,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );
    let maxEmissiveRadianceHistory: vec3<f32> = max(
        totalEmissiveLightHistoryMean + stdDeviationEmissiveHistory,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );

    let minSpecularRadianceHistory: vec3<f32> = max(
        totalSpecularRadianceHistoryMean - stdDeviationSpecularHistory,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );
    let maxSpecularRadianceHistory: vec3<f32> = max(
        totalSpecularRadianceHistoryMean + stdDeviationSpecularHistory,
        vec3<f32>(0.0f, 0.0f, 0.0f)
    );

    var bDisoccluded: bool = isDisoccluded2(in.texCoord, prevScreenUV);

    // current history pixel with ambient occlusion
    var indirectDiffuseRadianceHistory: vec4<f32> = textureSample(
        indirectDiffuseRadianceHistoryTexture,
        textureSampler,
        prevScreenUV                    
    );

    var directSunLightHistory: vec4<f32> = textureSample(
        directSunLightHistoryTexture,
        textureSampler,
        prevScreenUV
    );

    var emissiveLightHistory: vec4<f32> = textureSample(
        emissiveLightHistoryTexture,
        textureSampler,
        prevScreenUV
    );

    var specularRadianceHistory: vec4<f32> = textureSample(
        specularRadianceHistoryTexture,
        textureSampler,
        prevScreenUV
    );

    var ambientOcclusion: vec4<f32> = textureSample(
        ambientOcclusionTexture, 
        textureSampler,
        in.texCoord);               // use the current uv since we've already updated the ao in earlier stage
    var fAO: f32 = 1.0f - (ambientOcclusion.x / ambientOcclusion.y) * ambientOcclusion.w; 
    
    indirectDiffuseRadianceHistory.x = indirectDiffuseRadianceHistory.x;
    indirectDiffuseRadianceHistory.y = indirectDiffuseRadianceHistory.y;
    indirectDiffuseRadianceHistory.z = indirectDiffuseRadianceHistory.z;
 
    directSunLightHistory.x = directSunLightHistory.x;
    directSunLightHistory.y = directSunLightHistory.y;
    directSunLightHistory.z = directSunLightHistory.z;

    emissiveLightHistory.x = emissiveLightHistory.x;
    emissiveLightHistory.y = emissiveLightHistory.y;
    emissiveLightHistory.z = emissiveLightHistory.z;

    specularRadianceHistory.x = specularRadianceHistory.x;
    specularRadianceHistory.y = specularRadianceHistory.y;
    specularRadianceHistory.z = specularRadianceHistory.z;

    // box clamp for radiance history
    var clampedIndirectDiffuseHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    clampedIndirectDiffuseHistory.x = clamp(
        indirectDiffuseRadianceHistory.x,
        minIndirectDiffuseRadianceHistory.x,
        maxIndirectDiffuseRadianceHistory.x
    );
    clampedIndirectDiffuseHistory.y = clamp(
        indirectDiffuseRadianceHistory.y,
        minIndirectDiffuseRadianceHistory.y,
        maxIndirectDiffuseRadianceHistory.y
    );
    clampedIndirectDiffuseHistory.z = clamp(
        indirectDiffuseRadianceHistory.z,
        minIndirectDiffuseRadianceHistory.z,
        maxIndirectDiffuseRadianceHistory.z
    );

    var clampedDirectSunHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    clampedDirectSunHistory.x = clamp(
        directSunLightHistory.x,
        minDirectSunRadianceHistory.x,
        maxDirectSunRadianceHistory.x
    );
    clampedDirectSunHistory.y = clamp(
        directSunLightHistory.y,
        minDirectSunRadianceHistory.y,
        maxDirectSunRadianceHistory.y
    );
    clampedDirectSunHistory.z = clamp(
        directSunLightHistory.z,
        minDirectSunRadianceHistory.z,
        maxDirectSunRadianceHistory.z
    );

    var clampedEmissiveHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    clampedEmissiveHistory.x = clamp(
        emissiveLightHistory.x,
        minEmissiveRadianceHistory.x,
        maxEmissiveRadianceHistory.x
    );
    clampedEmissiveHistory.y = clamp(
        emissiveLightHistory.y,
        minEmissiveRadianceHistory.y,
        maxEmissiveRadianceHistory.y
    );
    clampedEmissiveHistory.z = clamp(
        emissiveLightHistory.z,
        minEmissiveRadianceHistory.z,
        maxEmissiveRadianceHistory.z
    );

    var clampedSpecularHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    clampedSpecularHistory.x = clamp(
        specularRadianceHistory.x,
        minSpecularRadianceHistory.x,
        maxSpecularRadianceHistory.x
    );
    clampedSpecularHistory.y = clamp(
        specularRadianceHistory.y,
        minSpecularRadianceHistory.y,
        maxSpecularRadianceHistory.y
    );
    clampedSpecularHistory.z = clamp(
        specularRadianceHistory.z,
        minSpecularRadianceHistory.z,
        maxSpecularRadianceHistory.z
    );

    var totalDiffuseHistory: vec3<f32> = 
        clampedIndirectDiffuseHistory.xyz +
        clampedEmissiveHistory.xyz + 
        clampedDirectSunHistory.xyz;
    var totalMinDiffuseRadianceHistory: vec3<f32> = 
        minIndirectDiffuseRadianceHistory.xyz +
        minEmissiveRadianceHistory.xyz +
        minDirectSunRadianceHistory.xyz;
    var totalMaxDiffuseRadianceHistory: vec3<f32> = 
        maxIndirectDiffuseRadianceHistory.xyz +
        maxEmissiveRadianceHistory.xyz +
        maxDirectSunRadianceHistory.xyz;

    var clampedTotalDiffuseHistory: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    clampedTotalDiffuseHistory.x = clamp(
        indirectDiffuseRadianceHistory.x + directSunLightHistory.x + emissiveLightHistory.x,
        totalMinDiffuseRadianceHistory.x,
        totalMaxDiffuseRadianceHistory.x
    );
    clampedTotalDiffuseHistory.y = clamp(
        indirectDiffuseRadianceHistory.y + directSunLightHistory.y + emissiveLightHistory.y,
        totalMinDiffuseRadianceHistory.y,
        totalMaxDiffuseRadianceHistory.y
    );
    clampedTotalDiffuseHistory.z = clamp(
        indirectDiffuseRadianceHistory.z + directSunLightHistory.z + emissiveLightHistory.z,
        totalMinDiffuseRadianceHistory.z,
        totalMaxDiffuseRadianceHistory.z
    );

    var indirectDiffuseRadiance: vec4<f32> = textureSample(
        indirectDiffuseRadianceTexture,
        textureSampler,
        in.texCoord);
    //ambientOcclusion = textureSample(
    //    ambientOcclusionTexture, 
    //    textureSampler,
    //    in.texCoord);
    //fAO = 1.0f - (ambientOcclusion.x / ambientOcclusion.y) * ambientOcclusion.w; 
    
    ambientOcclusion = textureSample(
        prevFilteredAmbientOcclusionTexture,
        textureSampler,
        prevScreenUV
    );
    fAO = ambientOcclusion.x;

    var directSunLightRadiance: vec4<f32> = textureSample(
        directSunLightTexture,
        textureSampler,
        in.texCoord);

    var emissiveLightRadiance: vec4<f32> = textureSample(
        emissiveLightTexture,
        textureSampler,
        in.texCoord);

    var specularRadiance: vec4<f32> = textureSample(
        specularRadianceTexture,
        textureSampler,
        in.texCoord);

    var totalDiffuseRadiance: vec3<f32> = 
        indirectDiffuseRadiance.xyz +
        directSunLightRadiance.xyz + 
        emissiveLightRadiance.xyz;

    //let emissiveRadiance: vec3<f32> = materialData[iMeshIndex * 2 + 1].xyz;
    //totalRadiance += emissiveRadiance;

    // this two value will reset the valid frame count when needed
    var fValidPixel: f32 = 1.0f - f32(bDisoccluded);
    var fInsideViewPixel: f32 = 1.0f - f32(isPrevUVOutOfBounds(prevScreenUV));
    
    var fAccumulationBlendWeight: f32 = 1.0f / 30.0f;
    if(fCenterIndirectDiffuseHistory <= 20.0f)
    {
        fAccumulationBlendWeight = 1.0f;
    }
    else
    {
        fAccumulationBlendWeight = 1.0f / 30.0f;
    }

    var momentHistory: vec2<f32> = textureSample(
        indirectDiffuseMomentHistoryTexture,
        textureSampler,
        prevScreenUV).xy;

    // luminance diff
    var fVarianceFactor: f32 = 2.0f;
    var fIndirectDiffuseLuminanceDiff: f32 = abs(fTotalIndirectDiffuseLuminance - fTotalIndirectDiffuseHistoryLuminance);
    var fEmissiveLuminanceDiff: f32 = abs(fTotalEmissiveLuminance - fTotalEmissiveHistoryLuminance);
    var fDirectSunLuminanceDiff: f32 = abs(fTotalDirectSunLuminance - fTotalDirectSunHistoryLuminance);
    var fTotalDiffuseLuminanceDiff: f32 = abs(fTotalDiffuseLuminance - fTotalDiffuseHistoryLuminance);

    var fTotalIndirectDiffuseVariance: f32 = abs(fTotalIndirectDiffuseHistoryLuminance * fTotalIndirectDiffuseHistoryLuminance - fTotalIndirectDiffuseHistoryLuminanceSquared);

    // variance driven accumulation weight, see svgf luminance difference weight calculation
    var fIndirectDiffuseVariance: f32  = abs(fTotalIndirectDiffuseHistoryLuminance * fTotalIndirectDiffuseHistoryLuminance - fTotalIndirectDiffuseHistoryLuminanceSquared);
    var fIndirectDiffuseBlendPct: f32 = clamp(
        1.0f - exp(-fIndirectDiffuseLuminanceDiff / (sqrt(fTotalIndirectDiffuseVariance) * fVarianceFactor + 1.0e-5f)) * fValidPixel * fInsideViewPixel,
        kfOneOverMaxBlendFrames,
        1.0f
    );

    var fEmissiveVariance: f32 = abs(fTotalEmissiveHistoryLuminance * fTotalEmissiveHistoryLuminance - fTotalDiffuseHistoryLuminanceSquared);
    var fEmissiveBlendPct: f32 = clamp(
        1.0f - exp(-fEmissiveLuminanceDiff / (sqrt(fEmissiveVariance) * fVarianceFactor + 1.0e-5f)) * fValidPixel * fInsideViewPixel,
        kfOneOverMaxBlendFrames,
        1.0f
    );

    var fTotalDiffuseVariance: f32 = abs(fTotalDiffuseHistoryLuminance * fTotalDiffuseHistoryLuminance - fTotalDiffuseHistoryLuminanceSquared);
    var fTotalDiffuseBlendPct: f32 = clamp(
        1.0f - exp(-fTotalDiffuseLuminanceDiff / (sqrt(fTotalDiffuseVariance) * fVarianceFactor + 1.0e-5f)) * fValidPixel * fInsideViewPixel,
        kfOneOverMaxBlendFrames,
        1.0f
    );


//fAccumulationBlendWeight = clamp(smoothstep(0.05f, 1.0f, fIndirectDiffuseVariance), 0.05f, 1.0f); // max(fIndirectDiffuseBlendPct, fEmissiveBlendPct);
fAccumulationBlendWeight = min(fTotalDiffuseBlendPct, fEmissiveBlendPct); //clamp(smoothstep(0.05f, 1.0f, fTotalDiffuseBlendPct), 0.05f, 1.0f);

//fAccumulationBlendWeight = 1.0f / 60.0f;

    var mixedIndirectDiffuseRadiance: vec3<f32> = mix(
        clampedIndirectDiffuseHistory.xyz,
        indirectDiffuseRadiance.xyz,
        fAccumulationBlendWeight
    );

    var mixedDirectSunRadiance: vec3<f32> = mix(
        clampedDirectSunHistory.xyz,
        directSunLightRadiance.xyz,
        fAccumulationBlendWeight
    );

    var mixedEmissiveRadiance: vec3<f32> = mix(
        totalEmissiveLightHistoryMean.xyz,
        emissiveLightRadiance.xyz,
        fAccumulationBlendWeight
    );
    
    var mixedSpecularRadiance: vec3<f32> = mix(
        clampedSpecularHistory.xyz,
        specularRadiance.xyz,
        fAccumulationBlendWeight
    );

    var mixedTotalDiffuseRadiance: vec3<f32> = mix(
        clampedTotalDiffuseHistory.xyz,
        totalDiffuseRadiance.xyz,
        fAccumulationBlendWeight
    );

    out.indirectDiffuseOutput.x = mixedIndirectDiffuseRadiance.x;
    out.indirectDiffuseOutput.y = mixedIndirectDiffuseRadiance.y;
    out.indirectDiffuseOutput.z = mixedIndirectDiffuseRadiance.z;
    out.indirectDiffuseOutput.w = clamp(indirectDiffuseRadianceHistory.w + 1.0f, 0.0f, 30.0f) * fValidPixel * fInsideViewPixel;

    out.directSunOutput.x = mixedDirectSunRadiance.x;
    out.directSunOutput.y = mixedDirectSunRadiance.y;
    out.directSunOutput.z = mixedDirectSunRadiance.z;
    out.directSunOutput.w = (directSunLightHistory.w + 1.0f) * fValidPixel * fInsideViewPixel;

    out.emissiveOutput.x = mixedEmissiveRadiance.x;
    out.emissiveOutput.y = mixedEmissiveRadiance.y;
    out.emissiveOutput.z = mixedEmissiveRadiance.z;
    out.emissiveOutput.w = (emissiveLightHistory.w + 1.0f) * fValidPixel * fInsideViewPixel;

    out.specularOutput.x = mixedSpecularRadiance.x;
    out.specularOutput.y = mixedSpecularRadiance.y;
    out.specularOutput.z = mixedSpecularRadiance.z;
    out.specularOutput.w = (specularRadianceHistory.w + 1.0f) * fValidPixel * fInsideViewPixel;

//let iMeshIndex: u32 = u32(floor((worldPosition.w - fract(worldPosition.w)) + 0.5f)) - 1;
//let fSpecular: f32 = materialData[iMeshIndex * 2].w;
//var totalRadiance: vec3<f32> = (
//    indirectDiffuseRadiance.xyz * (1.0f - fSpecular) + 
//    directSunLightRadiance.xyz + 
//    emissiveLightRadiance.xyz + 
//    specularRadiance.xyz * fSpecular
//) * fAO;
//var totalClampedRadiance: vec3<f32> = (
//    clampedIndirectDiffuseHistory.xyz * (1.0f - fSpecular) + 
//    clampedDirectSunHistory.xyz + 
//    clampedEmissiveHistory.xyz + 
//    clampedSpecularHistory.xyz * fSpecular
//) * fAO;
//
//var mixedTotalRadiance: vec3<f32> = mix(
//    totalClampedRadiance,
//    totalRadiance,
//    fAccumulationBlendWeight
//);
//out.indirectDiffuseOutput.x = mixedTotalRadiance.x;
//out.indirectDiffuseOutput.y = mixedTotalRadiance.y;
//out.indirectDiffuseOutput.z = mixedTotalRadiance.z;

    // moment and variance
    let indirectDiffuseMomentHistory: vec4<f32> = textureSample(
        indirectDiffuseMomentHistoryTexture,
        textureSampler,
        in.texCoord
    );
    var fLuminance: f32 = computeLuminance(out.indirectDiffuseOutput.xyz);
    let indirectDiffuseMoment: vec3<f32> = vec3<f32>(
        fLuminance,
        fLuminance * fLuminance,
        0.0f
    );
    out.indirectDiffuseMomentOutput.x = mix(
        indirectDiffuseMoment.x,
        indirectDiffuseMomentHistory.x,
        0.2f * fValidPixel * fInsideViewPixel
    );
    out.indirectDiffuseMomentOutput.y = mix(
        indirectDiffuseMoment.y,
        indirectDiffuseMomentHistory.y,
        0.2f * fValidPixel * fInsideViewPixel
    );
    
    //out.indirectDiffuseMomentOutput.x = mix(
    //    fTotalIndirectDiffuseHistoryLuminance,
    //    fTotalIndirectDiffuseLuminance,
    //    fAccumulationBlendWeight
    //);
    //out.indirectDiffuseMomentOutput.y = mix(
    //    fTotalIndirectDiffuseHistoryLuminanceSquared,
    //    fTotalIndirectDiffuseLuminanceSquared,
    //    fAccumulationBlendWeight
    //);

    // variance
    out.indirectDiffuseMomentOutput.z = max(
        out.indirectDiffuseMomentOutput.y - out.indirectDiffuseMomentOutput.x * out.indirectDiffuseMomentOutput.x,
        0.0f);

    // moment and variance history count
    out.indirectDiffuseMomentOutput.w = fIndirectDiffuseBlendPct; // (indirectDiffuseMomentHistory.w + 1.0f) * fValidPixel * fInsideViewPixel;

/*
    let directSunMomentHistory: vec4<f32> = textureSample(
        directSunMomentHistoryTexture,
        textureSampler,
        in.texCoord
    );
    fLuminance = computeLuminance(out.directSunOutput.xyz);
    let directSunMoment: vec3<f32> = vec3<f32>(
        fLuminance,
        fLuminance * fLuminance,
        0.0f
    );
    out.directSunMomentOutput.x = mix(
        directSunMoment.x,
        directSunMomentHistory.x,
        fAccumulationBlendWeight
    );
    out.directSunMomentOutput.y = mix(
        directSunMoment.y,
        directSunMomentHistory.y,
        fAccumulationBlendWeight
    );
    
    // variance
    out.directSunMomentOutput.z = max(
        out.directSunMomentOutput.y - out.directSunMomentOutput.x * out.directSunMomentOutput.x,
        0.0f);

    // moment and variance history count
    out.directSunMomentOutput.w = directSunMomentHistory.w + 1.0f;
*/

    //let emissiveMomentHistory: vec4<f32> = textureSample(
    //    emissiveLightMomentHistoryTexture,
    //    textureSampler,
    //    in.texCoord
    //);
    //fLuminance = computeLuminance(out.emissiveOutput.xyz);
    //let emissiveMoment: vec3<f32> = vec3<f32>(
    //    fLuminance,
    //    fLuminance * fLuminance,
    //    0.0f
    //);
    out.emissiveMomentOutput.x = mix(
        fTotalEmissiveHistoryLuminance,
        fTotalEmissiveLuminance,
        fAccumulationBlendWeight
    );
    out.emissiveMomentOutput.y = mix(
        fTotalEmissiveHistoryLuminanceSquared,
        fTotalEmissiveLuminanceSquared,
        fAccumulationBlendWeight
    );
    
    // variance
    out.emissiveMomentOutput.z = max(
        out.emissiveMomentOutput.y - out.emissiveMomentOutput.x * out.emissiveMomentOutput.x,
        0.0f);

    // moment and variance history count
    out.emissiveMomentOutput.w = fAccumulationBlendWeight; // (fEmissiveHistoryCount + 1.0f) * fValidPixel * fInsideViewPixel;

    let specularMomentHistory: vec4<f32> = textureSample(
        specularMomentHistoryTexture,
        textureSampler,
        in.texCoord
    );
    fLuminance = computeLuminance(out.specularOutput.xyz);
    let specularMoment: vec3<f32> = vec3<f32>(
        fLuminance,
        fLuminance * fLuminance,
        0.0f
    );
    out.specularMomentOutput.x = mix(
        specularMoment.x,
        specularMomentHistory.x,
        fAccumulationBlendWeight
    );
    out.specularMomentOutput.y = mix(
        specularMoment.y,
        specularMomentHistory.y,
        fAccumulationBlendWeight
    );
    
    // variance
    out.specularMomentOutput.z = max(
        out.specularMomentOutput.y - out.specularMomentOutput.x * out.specularMomentOutput.x,
        0.0f);

    // moment and variance history count
    out.specularMomentOutput.w = (specularMomentHistory.w + 1.0f) * fValidPixel * fInsideViewPixel;


//out.indirectDiffuseOutput = out.specularOutput;
//out.indirectDiffuseMomentOutput = out.specularMomentOutput;

    return out;
}

/////
fn computeLuminance(
    radiance: vec3<f32>) -> f32
{
    return dot(radiance, vec3<f32>(0.2126f, 0.7152f, 0.0722f));
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

    var fOneOverScreenWidth: f32 = 1.0f / f32(uniformData.miScreenWidth);
    var fOneOverScreenHeight: f32 = 1.0f / f32(uniformData.miScreenHeight);

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
    motionVector.x = motionVector.x * 2.0f - 1.0f;
    motionVector.y = motionVector.y * 2.0f - 1.0f;
    let backProjectedScreenUV: vec2<f32> = inputTexCoord - motionVector.xy;

    return (backProjectedScreenUV.x < 0.0f || backProjectedScreenUV.x > 1.0 || backProjectedScreenUV.y < 0.0f || backProjectedScreenUV.y > 1.0f);
}


