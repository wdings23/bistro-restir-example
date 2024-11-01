

struct UniformData {
    mViewProjectionMatrix: mat4x4<f32>,
    mLightViewProjectionMatrix: mat4x4<f32>,
    mInverseViewProjectionMatrix: mat4x4<f32>,

    miScreenWidth: u32,
    miScreenHeight: u32,

    miFrustumIndex: u32,
    miPadding: u32,

    mStartPosition: vec4<f32>,
    mMidPoint: vec4<f32>,
    mTopLeftLightPosition: vec4<f32>,
    mTopRightLightPosition: vec4<f32>,
    mBottomLeftLightPosition: vec4<f32>,
    mBottomRightLightPosition: vec4<f32>,

};
@group(0) @binding(0)
var<uniform> uniformData: UniformData;

struct VertexInput {
    @location(0) pos : vec4<f32>,
    @location(1) texCoord: vec4<f32>,
    @location(2) normal : vec4<f32>,
};
struct VertexOutput {
    @location(0) worldPosition: vec4<f32>,
    @location(1) texCoord: vec2<f32>,
    @builtin(position) pos: vec4<f32>,
    @location(2) normal: vec4<f32>,
};
struct FragmentOutput {
    @location(0) output : vec4<f32>,
    @location(1) worldPosition : vec4<f32>,
};


@vertex
fn vs_main(in: VertexInput, @builtin(vertex_index) iVertexIndex: u32) -> VertexOutput 
{
    var out: VertexOutput;
    out.pos = vec4<f32>(in.pos.x, in.pos.y, in.pos.z, 1.0f) * uniformData.mLightViewProjectionMatrix;
    out.worldPosition = in.pos;
    out.texCoord = in.texCoord.xy;
    out.normal = in.normal;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput 
{
    var out: FragmentOutput;

    var viewClipSpace: vec4<f32> = 
        vec4<f32>(in.worldPosition.x, in.worldPosition.y, in.worldPosition.z, 1.0f) *
        uniformData.mViewProjectionMatrix;

    viewClipSpace.x /= viewClipSpace.w;
    viewClipSpace.y /= viewClipSpace.w;
    viewClipSpace.z /= viewClipSpace.w;

    var viewScreenSpace: vec3<f32> = vec3<f32>(
        viewClipSpace.x * 0.5f + 0.5f,
        1.0f - (viewClipSpace.y * 0.5f + 0.5f),
        viewClipSpace.z * 0.5f + 0.5f
    );

    // see z-to-depth-chart for the values
    if(viewScreenSpace.x >= 0.0f && viewScreenSpace.x <= 1.0f && 
       viewScreenSpace.y >= 0.0f && viewScreenSpace.y <= 1.0f)
    {
        if(viewClipSpace.z <= 0.5285f)
        {
            if(viewScreenSpace.x <= 0.5f)
            {
                if(viewClipSpace.z <= 0.38125f)
                {
                    out.output = vec4<f32>(1.0f, 0.0f, 1.0f, viewClipSpace.z);
                }
                else 
                {
                    out.output = vec4<f32>(1.0f, 0.0f, 0.0f, viewClipSpace.z);
                }
            }
            else
            {
                if(viewClipSpace.z <= 0.38125f)
                {
                    out.output = vec4<f32>(1.0f, 0.5f, 0.0f, viewClipSpace.z);
                }
                else 
                {
                    out.output = vec4<f32>(1.0f, 1.0f, 0.0f, viewClipSpace.z);
                }
            }
        }
        else if(viewClipSpace.z <= 0.75f)
        {
            out.output = vec4<f32>(0.0f, 1.0f, 0.0f, viewClipSpace.z);
        }
        else
        {
            out.output = vec4<f32>(0.0f, 0.0f, 1.0f, viewClipSpace.z);
        }
    }
    else 
    {
        out.output = vec4<f32>(1.0f, 1.0f, 1.0f, viewClipSpace.z);
    }

    var diff: vec3<f32> = in.worldPosition.xyz - uniformData.mStartPosition.xyz;
    if(length(diff) <= 0.05f)
    {
        out.output = vec4<f32>(0.0f, 0.0f, 0.0f, viewClipSpace.z);
    }

    diff = in.worldPosition.xyz - uniformData.mMidPoint.xyz;
    if(length(diff) <= 0.05f)
    {
        out.output = vec4<f32>(0.0f, 0.0f, 0.0f, viewClipSpace.z);
    }

    diff = in.worldPosition.xyz - uniformData.mTopLeftLightPosition.xyz;
    if(length(diff) <= 0.05f)
    {
        out.output = vec4<f32>(0.0f, 0.0f, 0.0f, viewClipSpace.z);
    }

    diff = in.worldPosition.xyz - uniformData.mTopRightLightPosition.xyz;
    if(length(diff) <= 0.05f)
    {
        out.output = vec4<f32>(0.0f, 0.0f, 0.0f, viewClipSpace.z);
    }

    diff = in.worldPosition.xyz - uniformData.mBottomLeftLightPosition.xyz;
    if(length(diff) <= 0.05f)
    {
        out.output = vec4<f32>(0.0f, 0.0f, 0.0f, viewClipSpace.z);
    }

    diff = in.worldPosition.xyz - uniformData.mBottomRightLightPosition.xyz;
    if(length(diff) <= 0.05f)
    {
        out.output = vec4<f32>(0.0f, 0.0f, 0.0f, viewClipSpace.z);
    }

    out.output.x *= in.pos.z;
    out.output.y *= in.pos.z;
    out.output.z *= in.pos.z;

    out.worldPosition = in.worldPosition;

    return out;
}

