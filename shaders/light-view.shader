@group(0) @binding(0)
var<storage, read> aiVisibleTiles: array<u32>;

struct UniformData {
    mViewProjectionMatrix: mat4x4<f32>,
    mLightViewProjectionMatrix: mat4x4<f32>,
    mInverseViewProjectionMatrix: mat4x4<f32>,

    miScreenWidth: u32,
    miScreenHeight: u32,

    miFrustumIndex: u32,
};
@group(1) @binding(0)
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

    let iOutputTextureSize: u32 = 4096u;
    let iTileSize: u32 = 128u;

    let iStartTileX: u32 = (uniformData.miFrustumIndex % 2) * iOutputTextureSize; 
    let iStartTileY: u32 = u32(uniformData.miFrustumIndex / 2) * iOutputTextureSize;  

    let fOneOverNumTiles: f32 = f32(iTileSize) / f32(iOutputTextureSize);
    let iNumTilesPerRow: i32 = i32(iOutputTextureSize / iTileSize);

    let iTileX: i32 = i32(f32(u32(in.pos.x) + iStartTileX) / f32(iTileSize));
    let iTileY: i32 = i32(f32(u32(in.pos.y) + iStartTileY) / f32(iTileSize));
    let iBit = getBit(
        iTileX, 
        iTileY, 
        iNumTilesPerRow);

    //if(iBit != 0)
    {
        out.output = vec4<f32>(in.worldPosition.x, in.worldPosition.y, in.worldPosition.z, in.worldPosition.w + in.pos.z);
    }
    //else 
    //{
    //    out.output = vec4<f32>(0.0f, 0.0f, 0.0f, 0.0f);
    //}

    return out;
}

/////
fn getBit(
    iX: i32,
    iY: i32,
    iTotalDimension: i32) -> u32
{
    // 32 bits per int
    let iBitDimensions: i32 = iTotalDimension / 32;

    let iIndexX: i32 = iX / 32;          
    let iIndexY: i32 = iY;

    let iTotalIndex: i32 = iIndexY * iBitDimensions + iIndexX;
    let iBitIndex: u32 = u32(iX % 32);
    let iBitMask: u32 = u32(1u << iBitIndex);
    
    var ret = aiVisibleTiles[iTotalIndex] & iBitMask;
    ret = ret >> iBitIndex;

    return ret;
}