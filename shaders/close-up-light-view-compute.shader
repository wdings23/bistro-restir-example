const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;

@group(0) @binding(0)
var<storage, read_write> aCameraMatrices: array<mat4x4<f32>>;

@group(0) @binding(1)
var<storage, read> aBBox: array<i32>;

const iNumThreads: u32 = 1u;

@compute
@workgroup_size(iNumThreads)
fn cs_main(@builtin(global_invocation_id) index: vec3<u32>) 
{
    let iThreadID: u32 = index.x;

    var maxPosition: vec3<f32> = vec3<f32>(
        f32(aBBox[0]) * 0.001f,
        f32(aBBox[1]) * 0.001f,
        f32(aBBox[2]) * 0.001f
    );

    var minPosition: vec3<f32> = vec3<f32>(
        f32(aBBox[3]) * 0.001f,
        f32(aBBox[4]) * 0.001f,
        f32(aBBox[5]) * 0.001f
    );

    let kfRadiusMult: f32 = 1.1f;

    let midPt: vec3<f32> = (maxPosition + minPosition) * 0.5f;
    let diff: vec3<f32> = maxPosition - minPosition * 0.5f;
    var fRadius: f32 = max(max(abs(diff.x), abs(diff.y)), abs(diff.z)) * 0.5f;

    let lightDirection: vec3<f32> = normalize(vec3<f32>(1.0f, -0.7f, 0.0f));

    let eyePosition: vec3<f32> = midPt + lightDirection * fRadius * kfRadiusMult;
    let lookAtPosition: vec3<f32> = midPt;

    var up: vec3<f32> = vec3<f32>(0.0f, 1.0f, 0.0f);
    if(abs(lightDirection.y) >= 0.99f)
    {
        up = vec3<f32>(1.0f, 0.0f, 0.0f);
    }

    let viewMatrix: mat4x4<f32> = view_matrix(
        eyePosition,
        lookAtPosition,
        up
    );

    let orthoGraphicProjectionMatrix = orthographic_projection_matrix(
        fRadius * kfRadiusMult * -1.0f,
        fRadius * kfRadiusMult,
        fRadius * kfRadiusMult,
        fRadius * kfRadiusMult * -1.0f,
        fRadius * kfRadiusMult * 2.0f,
        fRadius * kfRadiusMult * -2.0f
    );

    aCameraMatrices[0] = viewMatrix * orthoGraphicProjectionMatrix;
    aCameraMatrices[1] = viewMatrix;
    aCameraMatrices[2] = orthoGraphicProjectionMatrix;
}

/////
fn orthographic_projection_matrix(
    left: f32,
    right: f32,
    top: f32,
    bottom: f32,
    far: f32,
    near: f32) -> mat4x4<f32>
{
    let width: f32 = right - left;
    let height: f32 = top - bottom;
    let far_minus_near: f32 = far - near;

    let m00: f32 = 2.0f / width;
    let m03: f32 = -(right + left) / (right - left);
    let m11: f32 = 2.0f / height;
    let m13: f32 = -(top + bottom) / (top - bottom);
    let m22: f32 = 1.0f / far_minus_near;
    let m23: f32 = -near / far_minus_near;
    
    return mat4x4<f32>(
        m00, 0.0f, 0.0f, m03,
        0.0f, m11, 0.0f, m13,
        0.0f, 0.0f, m22, m23,
        0.0f, 0.0f, 0.0f, 1.0f
    );
}

/////
fn view_matrix(
    eye_position: vec3<f32>,
    look_at: vec3<f32>,
    up: vec3<f32>) -> mat4x4<f32>
{
    var dir: vec3<f32> = look_at - eye_position;
    dir = normalize(dir);
    
    let tangent: vec3<f32> = normalize(cross(up, dir));
    let binormal: vec3<f32> = normalize(cross(dir, tangent));
    
    let xform_matrix = mat4x4<f32>(
        tangent.x, tangent.y, tangent.z, 0.0f,
        binormal.x, binormal.y, binormal.z, 0.0f,
        -dir.x, -dir.y, -dir.z, 0.0f,
        0.0f, 0.0f, 0.0f, 1.0f
    );
    
    let translation_matrix: mat4x4<f32> = mat4x4<f32>(
        1.0f, 0.0f, 0.0f, -eye_position.x,
        0.0f, 1.0f, 0.0f, -eye_position.y,
        0.0f, 0.0f, 1.0f, -eye_position.z,
        0.0f, 0.0f, 0.0f, 1.0f
    );

    return translation_matrix * xform_matrix;
}