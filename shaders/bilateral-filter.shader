const SIGMA: f32 = 1.0f;
const BSIGMA: f32 = 0.1f;
const MSIZE: i32 = 32;

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
    @location(0) radiance: vec4<f32>,
};

@group(0) @binding(0)
var radianceTexture: texture_2d<f32>;

@group(0) @binding(1)
var textureSampler: sampler;

struct UniformData
{
    miFrame: i32,
    miScreenWidth: i32,
    miScreenHeight: i32,
    mfRand: f32,

    mfAmbientOcclusionRadius: f32,
    mfAmbientOcclusionQuality: f32,
    mfAmbientOcclusionNumDirections: f32,
    mfPadding: f32,

    maSamples: array<vec4<f32>, 64>,

    mfRand0: f32,
    mfRand1: f32,
    mfRand2: f32,
    mfRand3: f32,

};

@group(1) @binding(0)
var<uniform> uniformData: UniformData;


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

    let fOneOverScreenWidth: f32 = 1.0f / f32(uniformData.miScreenWidth);
    let fOneOverScreenHeight: f32 = 1.0f / f32(uniformData.miScreenHeight);

	var color: vec3<f32> = textureSample(
		radianceTexture, 
		textureSampler,
		in.texCoord.xy).rgb;
	
	//declare stuff
	let kSize: i32 = (MSIZE-1)/2;
	var kernel: array<f32, MSIZE>;
	var final_colour: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
	
	//create the 1-D kernel
	var Z: f32 = 0.0f;
	for (var j: i32 = 0; j <= kSize; j += 1)
	{
		kernel[kSize+j] = normpdf(f32(j), SIGMA);
        kernel[kSize-j] = normpdf(f32(j), SIGMA);
    }
	
	var sampleColor: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
	var factor: f32 = 1.0f;
	var bZ: f32 = 1.0f / normpdf(0.0f, BSIGMA);
	
	//read out the texels
    var sampleUV: vec2<f32> = in.texCoord.xy;
	for(var i: i32 =-kSize; i <= kSize; i += 1)
	{
        sampleUV.y = clamp(in.texCoord.y + f32(i) * fOneOverScreenHeight, 0.0f, 1.0f);
		for(var j: i32 = -kSize; j <= kSize; j += 1)
		{
            sampleUV.x += clamp(in.texCoord.x + f32(j) * fOneOverScreenWidth, 0.0f, 1.0f);
            
			sampleColor = textureSample(
				radianceTexture, 
				textureSampler,
				sampleUV).rgb;
			factor = normpdf3(sampleColor - color, BSIGMA) * bZ * kernel[kSize+j] * kernel[kSize+i];
			Z += factor;
			final_colour += factor * sampleColor;

		}
	}
	
	out.radiance = vec4<f32>(final_colour / Z, 1.0f);

	return out;
}

//////
fn normpdf(
	fX: f32, 
	fSigma: f32) -> f32
{
	return 0.39894f * exp(-0.5f * fX * fX/(fSigma * fSigma)) / fSigma;
}

//////
fn normpdf3(
	v: vec3<f32>, 
	fSigma: f32) -> f32
{
	return 0.39894f * exp(-0.5f * dot(v, v )/(fSigma * fSigma)) / fSigma;
}