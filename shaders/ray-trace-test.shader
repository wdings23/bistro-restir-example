const UINT32_MAX: u32 = 1000000;
const FLT_MAX: f32 = 1.0e+10;

struct Ray
{
    mOrigin: vec4<f32>,
    mDirection: vec4<f32>,
    mfT: vec4<f32>,
};

struct BVHNode
{
    mBoundingBoxMin : vec4<f32>,
    mBoundingBoxMax : vec4<f32>,
    miLeftFirst : u32,
    miNumTriangles : u32,
    miLevel : u32,
    mPadding : u32
};

struct Tri
{
    miV0 : u32,
    miV1 : u32,
    miV2 : u32,
    mPadding : u32,

    mCentroid : vec4<f32>
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
    @location(0) color_output : vec4f
};

@group(0) @binding(0)
var<storage, read> nodeList: array<BVHNode>;

@group(1) @binding(0)
var<storage, read> positions: array<vec4<f32>>;

@group(1) @binding(1)
var<storage, read> triangles: array<Tri>;

@group(1) @binding(2)
var<storage, read> triangleIndices: array<u32>;

@vertex
fn vs_main(in: VertexInput) -> VertexOutput 
{
    var out: VertexOutput;
    out.pos = in.pos;
    out.texcoord = in.texcoord;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> FragmentOutput 
{
    var out: FragmentOutput;

    var ray: Ray;
    let p0: vec3<f32> = vec3<f32>(-1.0f, 1.0f, 2.0f);
    let p1: vec3<f32> = vec3<f32>(1.0f, 1.0f, 2.0f);
    let p2: vec3<f32> = vec3<f32>(-1.0f, -1.0f, 2.0f);

    ray.mOrigin = vec4<f32>(0.0f, 0.0f, -4.0f, 1.0f);
    let pixelPos: vec3<f32> = ray.mOrigin.xyz + p0 + (p1 - p0) * in.texcoord.x + (p2 - p0) * (1.0f - in.texcoord.y);
    ray.mDirection = vec4<f32>(normalize(pixelPos - vec3<f32>(ray.mOrigin.x, ray.mOrigin.y, ray.mOrigin.z)), 1.0f);
    ray.mfT = vec4<f32>(FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX);

    let intersectionPosition: vec3<f32> = intersectBVH(ray);

    out.color_output = vec4f(intersectionPosition.x, intersectionPosition.y, intersectionPosition.z, 1.0f);

    return out;
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
fn rayPlaneIntersection(
    pt0: vec3<f32>,
    pt1: vec3<f32>,
    planeNormal: vec3<f32>,
    fPlaneDistance: f32) -> f32
{
    var fRet: f32 = FLT_MAX;
    let v: vec3<f32> = pt1 - pt0;

    let fDenom: f32 = dot(v, planeNormal);
    fRet = -(dot(pt0, planeNormal) + fPlaneDistance) / (fDenom + 1.0e-4f);

    return fRet;
}

/////
fn rayBoxIntersect(
    rayPosition: vec3<f32>,
    rayDir: vec3<f32>,
    bboxMin: vec3<f32>,
    bboxMax: vec3<f32>) -> bool
{
    let oneOverRay: vec3<f32> = 1.0f / rayDir.xyz;
    let tMin: vec3<f32> = (bboxMin - rayPosition) * oneOverRay;
    let tMax: vec3<f32> = (bboxMax - rayPosition) * oneOverRay;

    var fTMin: f32 = min(tMin.x, tMax.x);
    var fTMax: f32 = max(tMin.x, tMax.x);

    fTMin = max(fTMin, min(tMin.y, tMax.y));
    fTMax = min(fTMax, max(tMin.y, tMax.y));

    fTMin = max(fTMin, min(tMin.z, tMax.z));
    fTMax = min(fTMax, max(tMin.z, tMax.z));

    return fTMax >= fTMin;
}

/////
fn rayTriangleIntersection(
    rayPt0: vec3<f32>, 
    rayPt1: vec3<f32>, 
    triPt0: vec3<f32>, 
    triPt1: vec3<f32>, 
    triPt2: vec3<f32>) -> vec3<f32>
{
    let v0: vec3<f32> = normalize(triPt1 - triPt0);
    let v1: vec3<f32> = normalize(triPt2 - triPt0);
    let cp: vec3<f32> = cross(v0, v1);
    let fLength: f32 = dot(cp, cp);
    //if(fLength <= 0.0001f)
    //{
    //    return vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    //}

    let triNormal: vec3<f32> = normalize(cp);
    let fPlaneDistance: f32 = -dot(triPt0, triNormal);

    //var collisionPtOnTriangle: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    //var bRet: bool = false;
    
    let fT: f32 = rayPlaneIntersection(
        rayPt0, 
        rayPt1, 
        triNormal, 
        fPlaneDistance);
    
    let MAX_MULT: f32 = 1000000000.0f;

    let fMinT: f32 = abs(min(fT, 0.0f)) * MAX_MULT;
    let fMaxT: f32 = abs(1.0f - max(fT, 1.0f)) * MAX_MULT;

    let collision: vec3<f32> = rayPt0 + (rayPt1 - rayPt0) * fT;
    let baryCentricCoord: vec3<f32> = barycentric(collision, triPt0, triPt1, triPt2);
    
    // add large number when barycentric coordinate is out of bounds to avoid branching 
    let fMin: f32 = min(baryCentricCoord.x, min(baryCentricCoord.y, baryCentricCoord.z));     
    let fMax: f32 = max(baryCentricCoord.x, max(baryCentricCoord.y, baryCentricCoord.z));  
    let fAddition: f32 = abs(min(fMin, 0.0f)) * MAX_MULT + abs(1.0f - max(fMax, 1.0)) * MAX_MULT;
    let collisionPtOnTriangle: vec3<f32> = (triPt0 * baryCentricCoord.x + triPt1 * baryCentricCoord.y + triPt2 * baryCentricCoord.z) 
        + fAddition
        + fMinT
        + fMaxT;

    //if(fT >= 0.0f && fT <= 1.0f)
    //{
        //var ret: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
        //bRet = (baryCentricCoord.x >= -0.01f && baryCentricCoord.x <= 1.01f &&
        //    baryCentricCoord.y >= -0.01f && baryCentricCoord.y <= 1.01f &&
        //    baryCentricCoord.z >= -0.01f && baryCentricCoord.z <= 1.01f);
        //if(bRet)
        //{
        //    collisionPtOnTriangle = (triPt0 * baryCentricCoord.x + triPt1 * baryCentricCoord.y + triPt2 * baryCentricCoord.z);
        //}
    //}

    return collisionPtOnTriangle;
}


/////
fn intersectTri(
    ray: Ray,
    iTriangle: u32) -> vec3<f32>
{
    let iTriangleIndex: u32 = triangleIndices[iTriangle];
    let tri: Tri = triangles[iTriangleIndex];

    let pos0: vec4<f32> = positions[tri.miV0];
    let pos1: vec4<f32> = positions[tri.miV1];
    let pos2: vec4<f32> = positions[tri.miV2];

    var iIntersected: u32 = 0;
    var fT: f32 = FLT_MAX;
    let intersectionPosition: vec3<f32> = rayTriangleIntersection(
        ray.mOrigin.xyz,
        ray.mOrigin.xyz + ray.mDirection.xyz * 1000.0f,
        pos0.xyz,
        pos1.xyz,
        pos2.xyz);
    
    return intersectionPosition;
}


/////
fn intersectBVH(
    ray: Ray) -> vec3<f32>
{
    var iStackTop: i32 = 0;
    var aiStack: array<u32, 256>;
    aiStack[iStackTop] = 0u;

    var retIntersectPosition: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
    var fClosestDistance: f32 = FLT_MAX;

    for(;;)
    {
        if(iStackTop < 0)
        {
            break;
        }

        let iNodeIndex: u32 = aiStack[iStackTop];
        iStackTop -= 1;

        let node: BVHNode = nodeList[iNodeIndex];

        if(node.miNumTriangles > 0)
        {
            // leaf node
            for(var iTri: u32 = 0u; iTri < node.miNumTriangles; iTri++)
            {
                var intersectionPosition: vec3<f32> = vec3<f32>(FLT_MAX, FLT_MAX, FLT_MAX);
                let iTriangleIndex: u32 = node.miLeftFirst + iTri;
                intersectionPosition = intersectTri(
                    ray,
                    iTriangleIndex);
                if(abs(intersectionPosition.x) < 10000.0f)
                {
                    let fDistanceToEye: f32 = length(intersectionPosition - ray.mOrigin.xyz);
                    if(fDistanceToEye < fClosestDistance)
                    {
                        fClosestDistance = fDistanceToEye;
                        retIntersectPosition = intersectionPosition;
                    }

                    // hit
                    //retIntersectPosition = intersectionPosition;
                    //iStackTop = -1;
                    //break;
                }
            }
        }
        else
        {
            let bIntersect: bool = rayBoxIntersect(
                ray.mOrigin.xyz,
                ray.mDirection.xyz,
                node.mBoundingBoxMin.xyz,
                node.mBoundingBoxMax.xyz);

            // node left and right child to stack
            if(bIntersect)
            {
                iStackTop += 1;
                aiStack[iStackTop] = node.miLeftFirst;
                iStackTop += 1;
                aiStack[iStackTop] = node.miLeftFirst + 1u;
            }
        }
    }

    return retIntersectPosition;
}
