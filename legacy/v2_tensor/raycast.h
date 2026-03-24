
struct Params
{
    float3*                 ray_origins;
    float3*                 ray_directions;
    float3*                 hit_positions_inner;
    float*                  hit_distances_inner;
    unsigned int*           hit_tri_indices_inner;
    float3*                 hit_positions_outer;
    float*                  hit_distances_outer;
    unsigned int*           hit_tri_indices_outer;
    float3*                 hit_positions_tread;
    float*                  hit_distances_tread;
    unsigned int*           hit_tri_indices_tread;
    OptixTraversableHandle  inner_handle;
    OptixTraversableHandle  outer_handle;
    OptixTraversableHandle  tread_handle;
    float3*                 normals_inner;
    unsigned int            num_rays;
};


struct RayGenData
{
    // No data needed
};


struct MissData
{
    float3 bg_color;
};


struct HitGroupData
{
    // No data needed
};