struct Locals {
    transform: mat4x4<f32>,
};
@group(0) @binding(0)
var<uniform> r_locals: Locals;

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texcoord: vec4<f32>,
    @location(2) color : vec4<f32>
};
struct VertexOutput {
    @location(0) texcoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(1) color: vec4<f32>
};
struct FragmentOutput {
    @location(0) color : vec4<f32>,
};


@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    let ndc: vec4<f32> = r_locals.transform * vec4<f32>(in.pos.xyz, 1.0);
    var out: VertexOutput;
    out.pos = r_locals.transform * vec4<f32>(in.pos.xyz, 1.0);
    out.texcoord = in.texcoord;
    out.color = in.color;
    return out;
}

@group(0) @binding(1)
var r_tex: texture_2d<f32>;

@group(0) @binding(2)
var r_sampler: sampler;

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput {
    let value = textureSample(r_tex, r_sampler, in.texcoord).r;
    let physical_color = vec3<f32>(pow(value, 2.2));  // gamma correct
    var out: FragmentOutput;
    //out.color = vec4<f32>(physical_color.rgb, 1.0);
    //var depth = in.pos.z * 0.5 + 0.5;
    //out.color = vec4<f32>(depth, depth, depth, 1.0);
    //out.color = vec4<f32>(1.0, 0.0, 0.0, 1.0);
    //out.color = vec4<f32>(in.color.xyz, 1.0);
    var dp: f32 = dot(in.color.xyz, normalize(vec3<f32>(1.0, 1.0, 1.0)));
    if(dp < 0.0)
    {
        dp = 0.0;
    }

    var ambientShade: f32 = 0.2;
    out.color = vec4<f32>(dp + ambientShade, dp + ambientShade, dp + ambientShade, 1.0);

    return out;
}