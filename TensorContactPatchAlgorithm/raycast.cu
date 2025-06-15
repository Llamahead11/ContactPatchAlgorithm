#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>

#include "helpers.h"
#include "raycast.h"
#include "vec_math.h"


extern "C" {
__constant__ Params params;
}


// static __forceinline__ __device__ void setPayload( float3 p )
// {
//     optixSetPayload_0( __float_as_int( p.x ) );
//     optixSetPayload_1( __float_as_int( p.y ) );
//     optixSetPayload_2( __float_as_int( p.z ) );
// }


// static __forceinline__ __device__ void computeRay( uint3 idx, uint3 dim, float3& origin, float3& direction )
// {
//     const float3 U = params.cam_u;
//     const float3 V = params.cam_v;
//     const float3 W = params.cam_w;
//     const float2 d = 2.0f * make_float2(
//             static_cast<float>( idx.x ) / static_cast<float>( dim.x ),
//             static_cast<float>( idx.y ) / static_cast<float>( dim.y )
//             ) - 1.0f;

//     origin    = params.cam_eye;
//     direction = normalize( d.x * U + d.y * V + W );
// }


extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    //const uint3 dim = optixGetLaunchDimensions();

    unsigned int ray_id = idx.x;

    if (ray_id >= params.num_rays)
        return;
    // Map our launch idx to a screen location and create a ray from the camera
    // location through the screen

    float3 ray_origin = params.ray_origins[ray_id]; 
    float3 ray_direction = params.ray_directions[ray_id];

    //computeRay( make_uint3( idx.x, idx.y, 0 ), dim, ray_origin, ray_direction );

    // Trace the ray against our scene hierarchy
    float3 result = make_float3( 0 );
    unsigned int p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11;
    optixTrace(
            params.inner_handle,
            ray_origin,
            -ray_direction,
            0.0f,                // Min intersection distance
            0.5f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2, p3);
    result.x = __int_as_float( p0 );
    result.y = __int_as_float( p1 );
    result.z = __int_as_float( p2 );
    // result = make_float3(__int_as_float( p0 ),__int_as_float( p1 ),__int_as_float( p2 ));
    float t = length(result - ray_origin); // or pass t as payload
    params.hit_positions_inner[ray_id] = result;
    if (t > 0.03f){
        params.hit_positions_inner[ray_id] = make_float3( 0 );
    }
    else{
        params.hit_distances_inner[ray_id] = -t;
    }
    params.hit_tri_indices_inner[ray_id] = p3;

    if (p3 == 0){
        optixTrace(
            params.inner_handle,
            ray_origin,
            ray_direction,
            0.0f,                // Min intersection distance
            0.5f,               // Max intersection distance
            0.0f,                // rayTime -- used for motion blur
            OptixVisibilityMask( 255 ), // Specify always visible
            OPTIX_RAY_FLAG_NONE,
            0,                   // SBT offset   -- See SBT discussion
            1,                   // SBT stride   -- See SBT discussion
            0,                   // missSBTIndex -- See SBT discussion
            p0, p1, p2, p3);

        result.x = __uint_as_float( p0 );
        result.y = __uint_as_float( p1 );
        result.z = __uint_as_float( p2 );
        
        params.hit_positions_inner[ray_id] = result;
        float t = length(result - ray_origin); // or pass t as payload
        if (t > 0.03f){
            params.hit_positions_inner[ray_id] = make_float3( 0 );
        }
        else{
            params.hit_distances_inner[ray_id] = t;
        }
        params.hit_tri_indices_inner[ray_id] = p3;
    }
    if (p3 == 0xFFFFFFFF){
        params.hit_tri_indices_inner[ray_id] = 0;
    }
    if (p3 != 0) {
        float3 result_o = make_float3( 0 );
        optixTrace(
                params.outer_handle,
                result,
                -params.normals_inner[p3],
                0.0f,                // Min intersection distance
                0.3f,               // Max intersection distance
                0.0f,                // rayTime -- used for motion blur
                OptixVisibilityMask( 255 ), // Specify always visible
                OPTIX_RAY_FLAG_NONE,
                0,                   // SBT offset   -- See SBT discussion
                1,                   // SBT stride   -- See SBT discussion
                0,                   // missSBTIndex -- See SBT discussion
                p4, p5, p6, p7);
        result_o.x = __int_as_float( p4 );
        result_o.y = __int_as_float( p5 );
        result_o.z = __int_as_float( p6 );
        // result = make_float3(__int_as_float( p0 ),__int_as_float( p1 ),__int_as_float( p2 ));
        float t_o = length(result_o - ray_origin); // or pass t as payload
        params.hit_positions_outer[ray_id] = result_o;
        if (t_o > 0.06f){
            //|| t_o < 0.045
            params.hit_positions_outer[ray_id] = make_float3( 0 );
        }
        else{
            params.hit_distances_outer[ray_id] = t_o;
        }
        params.hit_tri_indices_outer[ray_id] = p7;
    }
    if (p7 == 0xFFFFFFFF){
        params.hit_tri_indices_outer[ray_id] = 0;
    }

    if (p3 != 0) {
        float3 result_t = make_float3( 0 );
        optixTrace(
                params.tread_handle,
                result,
                -params.normals_inner[p3],
                0.0f,                // Min intersection distance
                0.3f,               // Max intersection distance
                0.0f,                // rayTime -- used for motion blur
                OptixVisibilityMask( 255 ), // Specify always visible
                OPTIX_RAY_FLAG_NONE,
                0,                   // SBT offset   -- See SBT discussion
                1,                   // SBT stride   -- See SBT discussion
                0,                   // missSBTIndex -- See SBT discussion
                p8, p9, p10, p11);
        result_t.x = __int_as_float( p8 );
        result_t.y = __int_as_float( p9 );
        result_t.z = __int_as_float( p10 );
        // result = make_float3(__int_as_float( p0 ),__int_as_float( p1 ),__int_as_float( p2 ));
        float t_t = length(result_t - ray_origin); // or pass t as payload
        params.hit_positions_tread[ray_id] = result_t;
        if (t_t > 0.06f){
            //|| t_o < 0.045
            params.hit_positions_tread[ray_id] = make_float3( 0 );
        }
        else{
            params.hit_distances_tread[ray_id] = t_t;
        }
        params.hit_tri_indices_tread[ray_id] = p11;
    }
    if (p11 == 0xFFFFFFFF){
        params.hit_tri_indices_tread[ray_id] = 0;
    }
    
    // Record results in our output raster
    //params.image[idx.y * params.image_width + idx.x] = make_color( result );
}


extern "C" __global__ void __miss__ms()
{
    // MissData* miss_data  = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    // setPayload(  miss_data->bg_color );

    // unsigned int idx = optixGetLaunchIndex().x;
    
    // params.hit_distances[idx] = 100000;//CUDART_INF_F;
    // params.hit_tri_indices[idx] = 0xFFFFFFFF;
    // params.hit_positions[idx] = make_float3(0.0f, 0.0f, 0.0f);
    float t = optixGetRayTmax();
    float3 origin = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    float3 hit_point = origin + t*direction; 
    unsigned int prim_id = 0;
    optixSetPayload_0(__float_as_int(0.0f));
    optixSetPayload_1(__float_as_int(0.0f));
    optixSetPayload_2(__float_as_int(0.0f));
    optixSetPayload_3(prim_id);
}


extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, indlucing barycentric coordinates.
    // const float2 barycentrics = optixGetTriangleBarycentrics();

    // setPayload( make_float3( barycentrics, 1.0f ) );

    float t = optixGetRayTmax();
    float3 origin = optixGetWorldRayOrigin();
    float3 direction = optixGetWorldRayDirection();
    float3 hit_point = origin + t*direction; 
    unsigned int prim_id = optixGetPrimitiveIndex();
    optixSetPayload_0(__float_as_int(hit_point.x));
    optixSetPayload_1(__float_as_int(hit_point.y));
    optixSetPayload_2(__float_as_int(hit_point.z));
    optixSetPayload_3(prim_id);
    // printf("Hit point: %f %f %f, tri_id: %u\n", hit_point.x, hit_point.y, hit_point.z, prim_id);
    // if (!isfinite(hit_point.x) || !isfinite(hit_point.y) || !isfinite(hit_point.z)) {
    //     printf("Invalid hit point detected\n");
    // }
}


// struct Params {
//     float* rays;         // [ox, oy, oz, dx, dy, dz] per ray (N x 6)
//     float* t_hit;        // Output: hit distance
//     int* tri_id;         // Output: triangle index
//     float3* hit_point;   // Output: hit coordinate
//     unsigned int num_rays;
// };

// extern "C" {
// __constant__ Params params;
// }

// extern "C" __global__ void __raygen__raycast() {
//     unsigned int idx = optixGetLaunchIndex().x;
//     if (idx >= params.num_rays) return;

//     float3 origin = make_float3(params.rays[idx * 6 + 0],
//                                 params.rays[idx * 6 + 1],
//                                 params.rays[idx * 6 + 2]);
//     float3 dir = make_float3(params.rays[idx * 6 + 3],
//                              params.rays[idx * 6 + 4],
//                              params.rays[idx * 6 + 5]);

//     unsigned int u0 = 0, u1 = 0;

//     optixTrace(
//         optixLaunchParams.traversable,
//         origin,
//         dir,
//         0.001f,
//         1e20f,
//         0.0f,
//         OptixVisibilityMask(255),
//         OPTIX_RAY_FLAG_DISABLE_ANYHIT,
//         0, 1, 0,
//         u0, u1
//     );

//     float t = __uint_as_float(u0);
//     int tri = static_cast<int>(u1);
//     params.t_hit[idx] = t;
//     params.tri_id[idx] = tri;
//     params.hit_point[idx] = origin + t * dir;
// }

// extern "C" __global__ void __closesthit__default() {
//     float t = optixGetRayTmax();
//     unsigned int prim_id = optixGetPrimitiveIndex();
//     optixSetPayload_0(__float_as_uint(t));
//     optixSetPayload_1(prim_id);
// }

// extern "C" __global__ void __miss__default() {
//     unsigned int idx = optixGetLaunchIndex().x;
//     params.t_hit[idx] = CUDART_INF_F;
//     params.tri_id[idx] = -1;
//     params.hit_point[idx] = make_float3(0.0f, 0.0f, 0.0f);
// }