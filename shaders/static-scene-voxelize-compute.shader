const UINT_MAX: u32 = 9999u;

struct InsideTriangleInfo
{
    mbInside: bool,
    mfDepth: f32,
    miAlignment: i32,
};

struct UniformData
{
    mfPositionScale: f32, 
    mfBrickDimension: f32,
    mfBrixelDimension: f32,
    miFrameIndex: u32,
};

struct VertexRange
{
    miStartVertex: u32,
    miEndVertex: u32,
};

struct TriangleSetupInfo
{
    mN0: vec2<f32>,
    mN1: vec2<f32>,
    mN2: vec2<f32>,

    mD: vec3<f32>,
};

struct BrickInfo
{
    mPosition: vec3<f32>,
    miBrixelIndex: u32,
};

struct BrixelInfo
{
    maiBrixelDistances: array<u32, 512>,
    maBrixelBarycentricCoord: array<vec3<f32>, 512>,
};

struct BoundingBox
{
    mMinPosition: vec3<f32>,
    mMaxPosition: vec3<f32>,
};

@group(0) @binding(0)
var<storage, read> aTrianglePositions: array<vec4<f32>>;

@group(0) @binding(1)
var<storage, read> aBBox: array<i32>;

@group(0) @binding(2)
var<storage, read_write> aiCounters: array<atomic<u32>>;

@group(0) @binding(3)
var<storage, read> aBricks: array<BrickInfo>;

@group(0) @binding(4)
var<storage, read_write> aBrixels: array<BrixelInfo>;

@group(0) @binding(5)
var<storage, read_write> aTempBrixels: array<u32>;

@group(1) @binding(0)
var<uniform> uniformData: UniformData;

@group(1) @binding(1)
var<storage, read> aMeshVertexRanges: array<VertexRange>;

@group(1) @binding(2)
var<storage, read> aiTriangleIndices: array<u32>;

@group(1) @binding(3)
var<storage, read> aiTriangleRanges: array<u32>;

@group(1) @binding(4)
var<storage, read> aTotalBrickBBox: array<BoundingBox>;

@group(0) @binding(6)
var<storage, read_write> debugBuffer: array<u32>;

const iNumThreads: u32 = 256u;

/////
@compute
@workgroup_size(iNumThreads)
fn cs_main(
    @builtin(num_workgroups) numWorkGroups: vec3<u32>,
    @builtin(local_invocation_index) iLocalIndex: u32,
    @builtin(workgroup_id) workGroup: vec3<u32>)
{
    // one triangle per thread

    if(uniformData.miFrameIndex != 1u)
    {
        return;
    }

    let iNumTotalThreads: u32 = numWorkGroups.x * iNumThreads;

    let iTotalThreadIndex: u32 = workGroup.x * iNumThreads + iLocalIndex;
    var iCurrTriangleIndex: u32 = iTotalThreadIndex * 3u;
    var iCurrMesh: u32 = 0u;

    var iNumVertices: u32 = aMeshVertexRanges[1].miEndVertex;
    var iNumTriangleIndices: u32 = aiTriangleRanges[0];
    
    let fNumBrixelsPerBrickRow: f32 = 8.0f;
    let fBrickSize: f32 = uniformData.mfBrickDimension;
    let fBrixelSize: f32 = uniformData.mfBrixelDimension;

    let meshBBox0: vec3<f32> = aTotalBrickBBox[0] * uniformData.mfPositionScale;
    let meshBBox1: vec3<f32> = aTotalBrickBBox[1] * uniformData.mfPositionScale;

    let meshBBoxDiff: vec3<f32> = meshBBox1 - meshBBox0;
    let meshBBoxSign: vec3<f32> = vec3<f32>(1.0f, 1.0f, 1.0f) * sign(meshBBoxDiff);

    let numBricks: vec3<f32> = vec3<f32>(
        256.0f / fNumBrixelsPerBrickRow, 
        256.0f / fNumBrixelsPerBrickRow, 
        256.0f / fNumBrixelsPerBrickRow);

    var minBrickWorldPosition: vec3<f32> = 
        vec3<f32>(
            floor(meshBBox0.x),
            floor(meshBBox0.y),
            floor(meshBBox0.z)
        );

    var brickBBox0: vec3<f32> = minBrickWorldPosition;

    let brickSize: vec3<f32> = vec3<f32>(
        fBrickSize, 
        fBrickSize, 
        fBrickSize);
    let brixelDelta: vec3<f32> = vec3<f32>(
        fBrixelSize, 
        fBrixelSize, 
        fBrixelSize) * meshBBoxSign;

    var iNumTriangles: u32 = aiTriangleRanges[0] / 3u;
    var iNumTrianglesPerThread = u32(ceil(f32(iNumTriangles) / f32(iNumTotalThreads)));

    var iCurrTriangle: u32 = iTotalThreadIndex * iNumTrianglesPerThread;
    var iEndTriangle: u32 = iCurrTriangle + iNumTrianglesPerThread;
    if(iEndTriangle > iNumTriangles)
    {
        iEndTriangle = iNumTriangles;
    }

    let bbox0: vec3<f32> = vec3<f32>(
        floor(meshBBox0.x),
        floor(meshBBox0.y),
        floor(meshBBox0.z)
    );

    let bbox1: vec3<f32> = vec3<f32>(
        ceil(meshBBox1.x),
        ceil(meshBBox1.y),
        ceil(meshBBox1.z)
    );

    let bboxDimension = bbox1 - bbox0;
    for(;iCurrTriangle < iEndTriangle; iCurrTriangle++)
    {
        let iCurrTriangleIndex: u32 = iCurrTriangle * 3u;

        let iTriangleIndex0: u32 = aiTriangleIndices[iCurrTriangleIndex];
        let iTriangleIndex1: u32 = aiTriangleIndices[iCurrTriangleIndex + 1u];
        let iTriangleIndex2: u32 = aiTriangleIndices[iCurrTriangleIndex + 2u];

        var pt0: vec3<f32> = aTrianglePositions[iTriangleIndex0].xyz * uniformData.mfPositionScale;
        var pt1: vec3<f32> = aTrianglePositions[iTriangleIndex1].xyz * uniformData.mfPositionScale;
        var pt2: vec3<f32> = aTrianglePositions[iTriangleIndex2].xyz * uniformData.mfPositionScale;

        // triangle normal
        let edge0: vec3<f32> = pt1 - pt0;
        let edge1: vec3<f32> = pt2 - pt1;
        let edge2: vec3<f32> = pt0 - pt2;

        // reject triangles with tiny area
        let normal: vec3<f32> = cross(edge2, edge0);
        if(length(normal) * 0.5f <= 1.0e-2f)
        {
            continue;
        }

        // triangle normal
        let triangleNormal: vec3<f32> = normalize(cross(normalize(edge0), normalize(edge2 * -1.0f)));

        let fAbsX: f32 = abs(triangleNormal.x);
        let fAbsY: f32 = abs(triangleNormal.y);
        let fAbsZ: f32 = abs(triangleNormal.z);
        var iTriangleAxis: u32 = 0u;
        if(fAbsY > fAbsX && fAbsY > fAbsZ)
        {
            iTriangleAxis = 1u;
        }
        else if(fAbsZ > fAbsX && fAbsZ > fAbsY)
        {
            iTriangleAxis = 2u;
        }
        
        // critical pt for checking the bounding points are on the opposite sides of the triangle's plane
        var criticalPt: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        if(triangleNormal.x <= 0.0f)
        {
            criticalPt.x = brixelDelta.x;
        }
        if(triangleNormal.y <= 0.0f)
        {
            criticalPt.y = brixelDelta.y;
        }
        if(triangleNormal.z <= 0.0f)
        {
            criticalPt.z = brixelDelta.z;
        }
        let fPlaneD0: f32 = dot(triangleNormal, criticalPt - pt0);
        let fPlaneD1: f32 = dot(triangleNormal, (brixelDelta - criticalPt) - pt0);
        
        // set up triangle for rasterization
        let triangleSetUpInfo: TriangleSetupInfo = triangleSetup(
            triangleNormal,
            pt0,
            pt1,
            pt2,
            edge0,
            edge1,
            edge2,
            brixelDelta,
            iTriangleAxis
        );

        // get the triangle's brick bounding box in relation to the mesh's local space
        let diff0: vec3<f32> = pt0 - minBrickWorldPosition;
        let diff1: vec3<f32> = pt1 - minBrickWorldPosition;
        let diff2: vec3<f32> = pt2 - minBrickWorldPosition;

        let b0: vec3<f32> = diff0 / brickSize;
        let b1: vec3<f32> = diff1 / brickSize;
        let b2: vec3<f32> = diff2 / brickSize;

        // number of bricks for this triangle
        var triBBox0: vec3<f32> = min(min(b0, b1), b2);
        var triBBox1: vec3<f32> = max(max(b0, b1), b2);
        triBBox0 = floor(triBBox0);
        triBBox1 = ceil(triBBox1);
        let triBBoxDiff: vec3<f32> = triBBox1 - triBBox0;
        let triBBoxDimensions: vec3<u32> = vec3<u32>(
                u32(triBBoxDiff.x + 1.0f),
                u32(triBBoxDiff.y + 1.0f),
                u32(triBBoxDiff.z + 1.0f));

        // scan line the containing bricks for the triangle
        var iCurrBrick: u32 = 0u;
        for(var iZ: u32 = 0u; iZ <= triBBoxDimensions.z; iZ++)
        {
            for(var iY: u32 = 0u; iY < triBBoxDimensions.y; iY++)
            {
                for(var iX: u32 = 0u; iX < triBBoxDimensions.x; iX++)
                {
                    let fX: f32 = triBBox0.x + f32(iX);
                    let fY: f32 = triBBox0.y + f32(iY);
                    let fZ: f32 = triBBox0.z + f32(iZ);
                    let brickPos: vec3<f32> = brickBBox0 + brickSize * vec3<f32>(fX, fY, fZ);
                    
                    var iBrickArrayIndex: u32 = 
                        u32(fX * brickSize.x) + 
                        u32(fY * brickSize.y) * u32(bboxDimension.x) +
                        u32(fZ * brickSize.z) * u32(bboxDimension.x) * u32(bboxDimension.y);

                    // scan line brixels in the brick
                    iCurrBrick += iNumThreads;
                    let bValid: bool = scanLineBrixel(
                        triangleSetUpInfo.mN0,
                        triangleSetUpInfo.mN1,
                        triangleSetUpInfo.mN2,
                        triangleSetUpInfo.mD,
                        triangleNormal,
                        brixelDelta,
                        fNumBrixelsPerBrickRow,
                        brickPos,
                        fPlaneD0,
                        fPlaneD1,
                        iTriangleAxis,
                        iBrickArrayIndex,
                        pt0,
                        pt1,
                        pt2,
                        0u,
                        iCurrTriangleIndex
                    );
                
                }   // for x = 0 triangle bbox x
            
            }   // for y = 0 triangle bbox y

        }   // for z = 0 triangle bbox z

    }   // for curr triangle index = start triangle to start triangle + num triangles per thread
}

/////
// from voxel pipe, see utils.h
fn triangleSetup(
    normal: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>,
    edge0: vec3<f32>,
    edge1: vec3<f32>,
    edge2: vec3<f32>,
    brixelDelta: vec3<f32>,
    iTriangleAxis: u32
) -> TriangleSetupInfo
{
    var ret: TriangleSetupInfo;
    
    var fSignW: f32 = 1.0f;
    switch(iTriangleAxis)
    {
        case 0u:
        {
            // u(const Vec a) { return a.y; }
            // v(const Vec a) { return a.z; }
            // w(const Vec a) { return a.x; }
            // x(const Vec a) { return a.z; }
            // y(const Vec a) { return a.x; }
            // z(const Vec a) { return a.y; }
            // ccw(const float3 n) { return n.x > 0.0f ? 1.0f : -1.0f; }
            
            // n0 = make_float2( -sel::v( edge0 ) * sgn_w, sel::u( edge0 ) * sgn_w );
            // const float  d0 = -(n0.x * sel::u( v0 ) + n0.y * sel::v( v0 )) +
            // fmaxf( 0.0f, sel::u( bbox_delta )*n0.x ) +
            // fmaxf( 0.0f, sel::v( bbox_delta )*n0.y );

            if(normal.x > 0.0f)
            {
                fSignW = 1.0f;
            }
            else 
            {
                fSignW = -1.0f;
            }

            ret.mN0 = vec2<f32>(-edge0.z, edge0.y) * fSignW;
            ret.mN1 = vec2<f32>(-edge1.z, edge1.y) * fSignW;
            ret.mN2 = vec2<f32>(-edge2.z, edge2.y) * fSignW;

            let vAxis0: vec2<f32> = vec2<f32>(v0.y, v0.z);
            let vAxis1: vec2<f32> = vec2<f32>(v1.y, v1.z);
            let vAxis2: vec2<f32> = vec2<f32>(v2.y, v2.z);

            ret.mD.x = -dot(ret.mN0, vAxis0) + max(brixelDelta.y * ret.mN0.x, 0.0f) + max(brixelDelta.z * ret.mN0.y, 0.0f);
            ret.mD.y = -dot(ret.mN1, vAxis1) + max(brixelDelta.y * ret.mN1.x, 0.0f) + max(brixelDelta.z * ret.mN1.y, 0.0f);
            ret.mD.z = -dot(ret.mN2, vAxis2) + max(brixelDelta.y * ret.mN2.x, 0.0f) + max(brixelDelta.z * ret.mN2.y, 0.0f);

            break;
        }
        case 1u:
        {
            // u(const Vec a) { return a.x; }
            // v(const Vec a) { return a.z; }
            // w(const Vec a) { return a.y; }
            // x(const Vec a) { return a.x; }
            // y(const Vec a) { return a.z; }
            // z(const Vec a) { return a.y; }
            // ccw(const float3 n) { return n.y < 0.0f ? 1.0f : -1.0f; }

            // n0 = make_float2( -sel::v( edge0 ) * sgn_w, sel::u( edge0 ) * sgn_w );
            // const float  d0 = -(n0.x * sel::u( v0 ) + n0.y * sel::v( v0 )) +
            // fmaxf( 0.0f, sel::u( bbox_delta )*n0.x ) +
            // fmaxf( 0.0f, sel::v( bbox_delta )*n0.y );

            if(normal.y < 0.0f)
            {
                fSignW = 1.0f;
            }
            else 
            {
                fSignW = -1.0f;
            }

            ret.mN0 = vec2<f32>(-edge0.z, edge0.x) * fSignW;
            ret.mN1 = vec2<f32>(-edge1.z, edge1.x) * fSignW;
            ret.mN2 = vec2<f32>(-edge2.z, edge2.x) * fSignW;

            let vAxis0: vec2<f32> = vec2<f32>(v0.x, v0.z);
            let vAxis1: vec2<f32> = vec2<f32>(v1.x, v1.z);
            let vAxis2: vec2<f32> = vec2<f32>(v2.x, v2.z);

            ret.mD.x = -dot(ret.mN0, vAxis0) + max(brixelDelta.x * ret.mN0.x, 0.0f) + max(brixelDelta.z * ret.mN0.y, 0.0f);
            ret.mD.y = -dot(ret.mN1, vAxis1) + max(brixelDelta.x * ret.mN1.x, 0.0f) + max(brixelDelta.z * ret.mN1.y, 0.0f);
            ret.mD.z = -dot(ret.mN2, vAxis2) + max(brixelDelta.x * ret.mN2.x, 0.0f) + max(brixelDelta.z * ret.mN2.y, 0.0f);

            break;
        }
        case 2u:
        {
            // u(const Vec a) { return a.x; }
            // v(const Vec a) { return a.y; }
            // w(const Vec a) { return a.z; }
            // x(const Vec a) { return a.x; }
            // y(const Vec a) { return a.y; }
            // z(const Vec a) { return a.z; }
            // ccw(const float3 n) { return n.z > 0.0f ? 1.0f : -1.0f; }

            // n0 = make_float2( -sel::v( edge0 ) * sgn_w, sel::u( edge0 ) * sgn_w );
            // const float  d0 = -(n0.x * sel::u( v0 ) + n0.y * sel::v( v0 )) +
            // fmaxf( 0.0f, sel::u( bbox_delta )*n0.x ) +
            // fmaxf( 0.0f, sel::v( bbox_delta )*n0.y );

            if(normal.z > 0.0f)
            {
                fSignW = 1.0f;
            }
            else
            {
                fSignW = -1.0f;
            }

            ret.mN0 = vec2<f32>(-edge0.y, edge0.x) * fSignW;
            ret.mN1 = vec2<f32>(-edge1.y, edge1.x) * fSignW;
            ret.mN2 = vec2<f32>(-edge2.y, edge2.x) * fSignW;

            let vAxis0: vec2<f32> = vec2<f32>(v0.x, v0.y);
            let vAxis1: vec2<f32> = vec2<f32>(v1.x, v1.y);
            let vAxis2: vec2<f32> = vec2<f32>(v2.x, v2.y);

            ret.mD.x = -dot(ret.mN0, vAxis0) + max(brixelDelta.x * ret.mN0.x, 0.0f) + max(brixelDelta.y * ret.mN0.y, 0.0f);
            ret.mD.y = -dot(ret.mN1, vAxis1) + max(brixelDelta.x * ret.mN1.x, 0.0f) + max(brixelDelta.y * ret.mN1.y, 0.0f);
            ret.mD.z = -dot(ret.mN2, vAxis2) + max(brixelDelta.x * ret.mN2.x, 0.0f) + max(brixelDelta.y * ret.mN2.y, 0.0f);

            break;
        }

        default:
        {
            break;
        }
    }

    return ret;
}

/////
fn scanLineBrixel(
    n0: vec2<f32>,
    n1: vec2<f32>,
    n2: vec2<f32>,
    d: vec3<f32>,
    triNormal: vec3<f32>,
    brixelDelta: vec3<f32>,
    fNumBrixelsPerBrickRow: f32,
    brickPosition: vec3<f32>,
    fPlaneD0: f32,
    fPlaneD1: f32,
    iTriangleAxis: u32,
    iBrickIndex: u32,
    triPt0: vec3<f32>,
    triPt1: vec3<f32>,
    triPt2: vec3<f32>,
    iMeshIndex: u32,
    iTriangleIndex: u32
) -> bool
{
    var bRet: bool = false;

    // delta uvw based on dominant axis
    var deltaW: vec3<f32> = vec3<f32>(brixelDelta.x, 0.0f, 0.0f); 
    var deltaV: vec3<f32> = vec3<f32>(0.0f, brixelDelta.y, 0.0f);
    var deltaU: vec3<f32> = vec3<f32>(brixelDelta.x, 0.0f, 0.0f);
    switch(iTriangleAxis)
    {
        case 0u:
        {
            deltaU = vec3<f32>(0.0f, 0.0f, brixelDelta.z);
            break;
        }
        case 1u:
        {
            deltaW = vec3<f32>(0.0f, brixelDelta.y, 0.0f);
            deltaV = vec3<f32>(0.0f, 0.0f, brixelDelta.z);
            break;
        }
        case 2u:
        {
            deltaW = vec3<f32>(0.0f, 0.0f, brixelDelta.z);
            break;
        }
        default:
        {
            break;
        }
    }  
    
    // go through uvw of the triangle bounds
    let iNumBrixelsPerBrickRow: u32 = u32(fNumBrixelsPerBrickRow);
    for(var iW: u32 = 0u; iW < iNumBrixelsPerBrickRow; iW++)
    {
        let brixelCoordW: vec3<f32> = brickPosition + deltaW * f32(iW);
        for(var iV: u32 = 0; iV < iNumBrixelsPerBrickRow; iV++)
        {
            let brixelCoordV: vec3<f32> = brixelCoordW + deltaV * f32(iV);
            for(var iU: u32 = 0; iU < iNumBrixelsPerBrickRow; iU++)
            {
                // check for bounding box points that are on opposite sides of the triangle plane
                let brixelCoordU: vec3<f32> = brixelCoordV + deltaU * f32(iU);
                let fDP: f32 = dot(triNormal, brixelCoordU);
                let bOppositePlane: bool = ((fDP + fPlaneD0) * (fDP + fPlaneD1)) <= 0.0f;

                // check of projected dominant axis voxel position with the edge normals plus d
                var projectedVoxelCoord: vec2<f32> = vec2<f32>(brixelCoordU.y, brixelCoordU.z);
                if(iTriangleAxis == 1u)
                {
                    projectedVoxelCoord = vec2<f32>(brixelCoordU.x, brixelCoordU.z);
                }
                else if(iTriangleAxis == 2u)
                {
                    projectedVoxelCoord = vec2<f32>(brixelCoordU.x, brixelCoordU.y);
                }
                
                let bCriteria0: bool = (dot(projectedVoxelCoord, n0) + d.x >= 0.0f);
                let bCriteria1: bool = (dot(projectedVoxelCoord, n1) + d.y >= 0.0f);
                let bCriteria2: bool = (dot(projectedVoxelCoord, n2) + d.z >= 0.0f);
                if(bCriteria0 && bCriteria1 && bCriteria2 && bOppositePlane)
                {
                    // determine the mapping of u, v, w of the dominant axis

                    var iAxisW: u32 = iU; 
                    var iAxisV: u32 = iV; 
                    var iAxisU: u32 = iW;
                    if(iTriangleAxis == 2u)
                    {
                        iAxisW = iW;
                        iAxisV = iV;
                        iAxisU = iU;
                    } 
                    else if(iTriangleAxis == 1u)
                    { 
                        iAxisW = iV; 
                        iAxisV = iW;
                        iAxisU = iU;
                    }

                    let iBrixelArrayIndex: u32 = iAxisW * iNumBrixelsPerBrickRow * iNumBrixelsPerBrickRow + iAxisV * iNumBrixelsPerBrickRow + iAxisU;
                    aBrixels[iBrickIndex].maiBrixelDistances[iBrixelArrayIndex] |= 1u;
                    
                    // get closest pt on the triangle and the barycentric coordinate of the pt
                    let closestPt: vec3<f32> = closestPointToTriangle(
                        brixelCoordU,
                        triPt0,
                        triPt1,
                        triPt2
                    );

                    // barycentric, encode the mesh and triangle indices in the decimal places
                    var barycentricCoord: vec3<f32> = barycentric(
                        closestPt,
                        triPt0,
                        triPt1,
                        triPt2
                    );
                    barycentricCoord.x = clamp(barycentricCoord.x, 0.0f, 0.99999f);
                    barycentricCoord.y = clamp(barycentricCoord.y, 0.0f, 0.99999f);
                    barycentricCoord.z = clamp(barycentricCoord.z, 0.0f, 0.99999f);
                    barycentricCoord.x += f32(iMeshIndex + 1);
                    barycentricCoord.y += f32(iTriangleIndex + 1);

                    aBrixels[iBrickIndex].maBrixelBarycentricCoord[iBrixelArrayIndex] = barycentricCoord;

                    bRet = true;
                }
            }
        }
    }

    return bRet;
}

/////
//
// https://stackoverflow.com/questions/2924795/fastest-way-to-compute-point-to-triangle-distance-in-3d
//
fn closestPointToTriangle(
    pt: vec3<f32>,
    v0: vec3<f32>,
    v1: vec3<f32>,
    v2: vec3<f32>) -> vec3<f32>
{
    let v10: vec3<f32> = v1 - v0;
    let v20: vec3<f32> = v2 - v0;
    let vp0: vec3<f32> = pt - v0;

    let d1: f32 = dot(v10, vp0);
    let d2: f32 = dot(v20, vp0);
    if(d1 <= 0.f && d2 <= 0.f)
    {
        return v0; //#1
    }

    let bp: vec3<f32> = pt - v1;
    let d3: f32 = dot(v10, bp);
    let d4: f32 = dot(v20, bp);
    if(d3 >= 0.f && d4 <= d3)
    { 
        return v1; //#2
    }

    let cp: vec3<f32> = pt - v2;
    let d5: f32 = dot(v10, cp);
    let d6: f32 = dot(v20, cp);
    if(d6 >= 0.f && d5 <= d6)
    { 
        return v2; //#3
    }

    let vc: f32 = d1 * d4 - d3 * d2;
    if(vc <= 0.f && d1 >= 0.f && d3 <= 0.f)
    {
        let v: f32 = d1 / (d1 - d3);
        return v0 + v10 * v; //#4
    }

    let vb: f32 = d5 * d2 - d1 * d6;
    if(vb <= 0.f && d2 >= 0.f && d6 <= 0.f)
    {
        let v: f32 = d2 / (d2 - d6);
        return v0 + v20 * v; //#5
    }

    let va: f32 = d3 * d6 - d5 * d4;
    if(va <= 0.f && (d4 - d3) >= 0.f && (d5 - d6) >= 0.f)
    {
        let v: f32 = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return v1 + (v2 - v1) * v; //#6
    }

    let denom: f32 = 1.f / (va + vb + vc);
    let v: f32 = vb * denom;
    let w: f32 = vc * denom;
    return v0 + v10 * v + v20 * w; //#0
}

/////
fn barycentric(
    p: vec3<f32>, 
    a: vec3<f32>, 
    b: vec3<f32>, 
    c: vec3<f32>) -> vec3<f32>
{
    let v0: vec3<f32> = b - a;
    let v1: vec3<f32> = c - a;
    let v2: vec3<f32> = p - a;
    let fD00: f32 = dot(v0, v0);
    let fD01: f32 = dot(v0, v1);
    let fD11: f32 = dot(v1, v1);
    let fD20: f32 = dot(v2, v0);
    let fD21: f32 = dot(v2, v1);
    let fOneOverDenom: f32 = 1.0f / (fD00 * fD11 - fD01 * fD01);
    let fV: f32 = (fD11 * fD20 - fD01 * fD21) * fOneOverDenom;
    let fW: f32 = (fD00 * fD21 - fD01 * fD20) * fOneOverDenom;
    let fU: f32 = 1.0f - fV - fW;

    return vec3<f32>(fU, fV, fW);
}

/////
fn murmurHash13(
    src: vec3<u32>) -> u32
{
    var srcCopy: vec3<u32> = src;

    let M: u32 = 0x5bd1e995u;
    var h: u32 = 1190494759u;
    srcCopy *= M; 
    srcCopy ^= vec3<u32>(
        srcCopy.x >> 24u,
        srcCopy.y >> 24u,
        srcCopy.z >> 24u
    ); 
    srcCopy *= M;
    h *= M; h ^= srcCopy.x; 
    h *= M; h ^= srcCopy.y; 
    h *= M; h ^= srcCopy.z;
    h ^= h >> 13u; 
    h *= M; 
    h ^= h >> 15u;

    return h;
}

/////
fn hash13(
    src: vec3<u32>,
    iNumSlots: u32) -> u32
{
    let h: u32 = murmurHash13(src);
    let fValue: f32 = bitcast<f32>((h & 0x007ffffffu) | 0x3f800000u) - 1.0f;
    let iRet: u32 = clamp(
        u32(fValue * f32(iNumSlots - 1u)), 
        0u, 
        iNumSlots - 1u);

    return iRet;
}