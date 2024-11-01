const PI: f32 = 3.14159f;

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

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texcoord: vec2<f32>,
    @location(2) color : vec4<f32>
};
struct VertexOutput {
    @location(0) texcoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) color: vec4<f32>
};
struct FragmentOutput {
    @location(0) color0 : vec4<f32>,
    @location(1) color1 : vec4<f32>,
    @location(2) color2 : vec4<f32>,
    @location(3) color3 : vec4<f32>,
    @location(4) brdfLUT: vec4<f32>,
};

@group(0) @binding(0)
var skyTexture: texture_2d<f32>;

@group(0) @binding(1)
var textureSampler: sampler;

@group(1) @binding(0)
var<uniform> defaultUniformData: DefaultUniformData;

/////
@vertex
fn vs_main(in: VertexInput) -> VertexOutput 
{
    var out: VertexOutput;
    out.pos = in.pos;
    out.texcoord = in.texcoord;
    out.color = in.color;
    return out;
}

//////
@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput
{
    var output: FragmentOutput;
    let kiNumLoops: u32 = 128u;

    let color0: vec3<f32> = convolve(
        in.texcoord, 
        128u,
        -1.0f,
        1.0f);
    let color1: vec3<f32> = convolve(
        in.texcoord, 
        128u,
        -0.7f,
        0.7f);
    let color2: vec3<f32> = convolve(
        in.texcoord, 
        128u,
        -0.4f,
        0.4f);
    let color3: vec3<f32> = convolve(
        in.texcoord, 
        128u,
        -0.1f,
        0.1f);

    output.color0 = vec4<f32>(color0, 1.0f);
    output.color1 = vec4<f32>(color1, 1.0f);
    output.color2 = vec4<f32>(color2, 1.0f);
    output.color3 = vec4<f32>(color3, 1.0f);

    let integratedBRDF: vec2<f32> = IntegrateBRDF(in.texcoord.x, 1.0f - in.texcoord.y);
    output.brdfLUT = vec4<f32>(integratedBRDF.x, integratedBRDF.y, 0.0f, 1.0f);

    return output;
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

/////
fn uniformSampling(
    normal: vec3<f32>,
    fVal0: f32,
    fVal1: f32) -> vec3<f32>
{
    let fPhi: f32 = 2.0f * PI * fVal0;
    let fCosTheta: f32 = 1.0f - fVal1;
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

//////
fn convolve(
    texCoord: vec2<f32>,
    iNumLoops: u32,
    fPhiStart: f32,
    fPhiEnd: f32) -> vec3<f32>
{
    var fCount: f32 = 0.0f;
    var color: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    let normal: vec3<f32> = normalize(decodeOctahedronMap(texCoord));
    for(var i: i32 = 0; i < i32(iNumLoops); i++)
    {
        //var fPhi: f32 = f32(i) / f32(iNumLoops);
        var fPct0: f32 = f32(i) / f32(iNumLoops);
        var fPhi: f32 = (fPhiStart + (fPhiEnd - fPhiStart) * fPct0) * PI;
        let fCosPhi: f32 = cos(fPhi);
        let fSinPhi: f32 = sin(fPhi);
        for(var j: u32 = 0u; j < iNumLoops; j++)
        {
            var fPct1: f32 = f32(j) / f32(iNumLoops);
            var fTheta: f32 = fPct1 * 2.0f * PI;
            let localNormal: vec3<f32> = vec3<f32>(
                fSinPhi * cos(fTheta),
                fSinPhi * sin(fTheta),
                fCosPhi
            );
            var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
            if(abs(normal.y) > 0.98f)
            {
                up = vec3<f32>(1.0f, 0.0f, 0.0f);
            }
            let tangent: vec3<f32> = normalize(cross(up, normal));
            let binormal: vec3<f32> = normalize(cross(normal, tangent));
            let direction: vec3<f32> = normalize(tangent * localNormal.x + binormal * localNormal.y + normal * localNormal.z);

            //var fTheta: f32 = f32(j) / f32(iNumLoops);
            //let direction: vec3<f32> = uniformSampling(
            //    normal,
            //    fPhi,
            //    fTheta
            //);

            var fDP: f32 = max(dot(normalize(direction), normal), 0.0f);

            let uv: vec2<f32> = octahedronMap2(direction);
            color += textureSample(
                skyTexture,
                textureSampler,
                uv
            ).xyz * fDP;

            fCount += 1.0f;
        }
    }

    return color / fCount;
}




// ----------------------------------------------------------------------------
// http://holger.dammertz.org/stuff/notes_HammersleyOnHemisphere.html
// efficient VanDerCorpus calculation.
fn RadicalInverse_VdC(bits: u32) -> f32
{
    var bitsCopy: u32 = bits;

    bitsCopy = (bitsCopy << 16u) | (bitsCopy >> 16u);
    bitsCopy = ((bitsCopy & 0x55555555u) << 1u) | ((bitsCopy & 0xAAAAAAAAu) >> 1u);
    bitsCopy = ((bitsCopy & 0x33333333u) << 2u) | ((bitsCopy & 0xCCCCCCCCu) >> 2u);
    bitsCopy = ((bitsCopy & 0x0F0F0F0Fu) << 4u) | ((bitsCopy & 0xF0F0F0F0u) >> 4u);
    bitsCopy = ((bitsCopy & 0x00FF00FFu) << 8u) | ((bitsCopy & 0xFF00FF00u) >> 8u);
    return f32(bitsCopy) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
fn Hammersley(
    i: u32, 
    N: u32) -> vec2<f32>
{
	return vec2<f32>(f32(i)/f32(N), RadicalInverse_VdC(i));
}
// ----------------------------------------------------------------------------
fn ImportanceSampleGGX(
    Xi: vec2<f32>, 
    N: vec3<f32>, 
    roughness: f32) -> vec3<f32>
{
	let a: f32 = roughness*roughness;
	
	let phi: f32 = 2.0f * PI * Xi.x;
	let cosTheta: f32 = sqrt((1.0f - Xi.y) / (1.0f + (a*a - 1.0f) * Xi.y));
	let sinTheta: f32 = sqrt(1.0f - cosTheta*cosTheta);
	
	// from spherical coordinates to cartesian coordinates - halfway vector
	var H: vec3<f32> = vec3<f32>(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );
	
	// from tangent-space H vector to world-space sample vector
	var up: vec3<f32> = vec3<f32>(1.0, 0.0, 0.0);         
    if(abs(N.z) < 0.999)
    { 
        up = vec3<f32>(0.0f, 0.0f, 1.0f); 
    } 
	let tangent: vec3<f32>   = normalize(cross(up, N));
	let bitangent: vec3<f32> = cross(N, tangent);
	
	var sampleVec: vec3<f32> = tangent * H.x + bitangent * H.y + N * H.z;
	return normalize(sampleVec);
}
// ----------------------------------------------------------------------------
fn GeometrySchlickGGX(
    NdotV: f32, 
    roughness: f32) -> f32
{
    // note that we use a different k for IBL
    let a: f32 = roughness;
    let k: f32 = (a * a) / 2.0;

    let nom: f32   = NdotV;
    let denom: f32 = NdotV * (1.0 - k) + k;

    return nom / denom;
}
// ----------------------------------------------------------------------------
fn GeometrySmith(
    N: vec3<f32>, 
    V: vec3<f32>, 
    L: vec3<f32>, 
    roughness: f32) -> f32
{
    let NdotV: f32 = max(dot(N, V), 0.0);
    let NdotL: f32 = max(dot(N, L), 0.0);
    let ggx2: f32 = GeometrySchlickGGX(NdotV, roughness);
    let ggx1: f32 = GeometrySchlickGGX(NdotL, roughness);

    return ggx1 * ggx2;
}
// ----------------------------------------------------------------------------
fn IntegrateBRDF(
    NdotV: f32, 
    roughness: f32) -> vec2<f32>
{
    var V: vec3<f32> = vec3<f32>(
        sqrt(1.0f - NdotV*NdotV),
        0.0f,
        NdotV
    );

    var A: f32 = 0.0f;
    var B: f32 = 0.0f; 

    var N: vec3<f32> = vec3(0.0, 0.0, 1.0);
    
    let SAMPLE_COUNT: u32 = 1024u;
    for(var i: u32 = 0u; i < SAMPLE_COUNT; i += 1u)
    {
        // generates a sample vector that's biased towards the
        // preferred alignment direction (importance sampling).
        let Xi: vec2<f32> = Hammersley(i, SAMPLE_COUNT);
        let H: vec3<f32> = ImportanceSampleGGX(Xi, N, roughness);
        let L: vec3<f32> = normalize(2.0 * dot(V, H) * H - V);

        let NdotL: f32 = max(L.z, 0.0);
        let NdotH: f32 = max(H.z, 0.0);
        let VdotH: f32 = max(dot(V, H), 0.0);

        if(NdotL > 0.0)
        {
            let G: f32 = GeometrySmith(N, V, L, roughness);
            let G_Vis: f32 = (G * VdotH) / (NdotH * NdotV);
            let Fc: f32 = pow(1.0 - VdotH, 5.0);

            A += (1.0 - Fc) * G_Vis;
            B += Fc * G_Vis;
        }
    }
    A /= f32(SAMPLE_COUNT);
    B /= f32(SAMPLE_COUNT);
    return vec2<f32>(A, B);
}
