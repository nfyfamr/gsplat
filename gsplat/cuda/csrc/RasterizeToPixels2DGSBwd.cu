#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <ATen/cuda/Atomic.cuh>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include "Common.h"
#include "Rasterization.h"
#include "Utils.cuh"

namespace gsplat {

__host__ __device__
inline vec2& operator+=(vec2& a, const vec2& b) {
    a.x += b.x;
    a.y += b.y;
    return a;
}

__host__ __device__
inline vec3& operator+=(vec3& a, const vec3& b) {
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    return a;
}

namespace cg = cooperative_groups;

template <uint32_t CDIM, typename scalar_t>
__global__ void per_gaussian_rasterize_to_pixels_2dgs_bwd_kernel(
    const uint32_t I,        // number of images
    const uint32_t N,        // number of gaussians
    const uint32_t n_isects, // number of ray-primitive intersections.
    const bool packed,       // whether the input tensors are packed
    // fwd inputs
    const vec2
        *__restrict__ means2d, // Projected Gaussian means. [..., N, 2] if
                               // packed is False, [nnz, 2] if packed is True.
    const scalar_t
        *__restrict__ ray_transforms, // transformation matrices that transforms
                                      // xy-planes in pixel spaces into splat
                                      // coordinates. [..., N, 3, 3] if packed is
                                      // False, [nnz, channels] if packed is
                                      // True. This is (KWH)^{-1} in the paper
                                      // (takes screen [x,y] and map to [u,v])
    const scalar_t *__restrict__ colors,  // [..., N, CDIM] or [nnz, CDIM]  //
                                          // Gaussian colors or ND features.
    const scalar_t *__restrict__ normals, // [..., N, 3] or [nnz, 3] // The
                                          // normals in camera space.
    const scalar_t *__restrict__ opacities, // [..., N] or [nnz] // Gaussian
                                            // opacities that support per-view
                                            // values.
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] // Background colors
                                              // on camera basis
    const bool *__restrict__ masks, // [..., tile_height, tile_width]     //
                                    // Optional tile mask to skip rendering GS
                                    // to masked tiles.

    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]

    // fwd outputs
    const scalar_t *__restrict__ render_colors, // [..., image_height,
                                                // image_width, CDIM]
    const scalar_t
        *__restrict__ render_alphas, // [..., image_height, image_width, 1]
    const scalar_t *__restrict__ render_normals, // [..., image_height,
                                                // image_width, 3]
    const scalar_t
        *__restrict__ render_stats, // [..., image_height, image_width, 2]
    const int32_t
        *__restrict__ last_ids, // [..., image_height, image_width]     // the id
                                // to last gaussian that got intersected
    const int32_t *__restrict__ median_ids, // [..., image_height, image_width] //
                                            // the id to the gaussian that
                                            // brings the opacity over 0.5

    // efficient backward
    const uint32_t num_buckets,
    const int32_t *__restrict__ per_tile_bucket_offset,
    const int32_t *__restrict__ bucket_to_tile,
    const scalar_t *__restrict__ sampled_stats,
    const scalar_t *__restrict__ sampled_ar,
    const scalar_t *__restrict__ sampled_an,
    const int32_t *__restrict__ max_contrib,

    // grad outputs
    const scalar_t
        *__restrict__ v_render_colors, // [..., image_height, image_width,     //
                                       // RGB CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]  //
                                       // total opacities.
    const scalar_t
        *__restrict__ v_render_normals, // [..., image_height, image_width, 3]  //
                                        // camera space normals
    const scalar_t
        *__restrict__ v_render_distort, // [..., image_height, image_width, 1]  //
                                        // mip-nerf 360 distorts
    const scalar_t
        *__restrict__ v_render_median, // [..., image_height, image_width, 1]  //
                                       // the median depth

    // grad inputs
    vec2 *__restrict__ v_means2d_abs,        // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,            // [..., N, 2] or [nnz, 2]
    scalar_t *__restrict__ v_ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    scalar_t *__restrict__ v_colors,         // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities,      // [..., N] or [nnz]
    scalar_t *__restrict__ v_normals,        // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_densify
) {
	// global_bucket_idx = warp_idx
    auto block = cg::this_thread_block();
	auto my_warp = cg::tiled_partition<32>(block);
	uint32_t global_bucket_idx = block.group_index().x * my_warp.meta_group_size() + my_warp.meta_group_rank();
    bool valid_bucket = global_bucket_idx < (uint32_t) num_buckets;
	if (!valid_bucket) return;
    
	bool valid_splat = false;

    const uint32_t GLOBAL_TILE_ID = bucket_to_tile[global_bucket_idx];
    int32_t range_start = tile_offsets[GLOBAL_TILE_ID];
    int32_t range_end =
        // see if this is the last tile in the image
        (GLOBAL_TILE_ID == I * tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[GLOBAL_TILE_ID + 1];
    uint32_t num_splats_in_tile = range_end - range_start;
	// What is the number of buckets before me? what is my offset?
    uint32_t bbm_current = (GLOBAL_TILE_ID == 0) ? 0 : per_tile_bucket_offset[GLOBAL_TILE_ID - 1];
    uint32_t bucket_idx_in_tile = global_bucket_idx - bbm_current;
	uint32_t splat_idx_in_tile = bucket_idx_in_tile * 32 + my_warp.thread_rank();
	uint32_t splat_idx_global = range_start + splat_idx_in_tile;
	valid_splat = (splat_idx_in_tile < num_splats_in_tile);

	// if first gaussian in bucket is useless, then others are also useless
	if (bucket_idx_in_tile * 32 > max_contrib[GLOBAL_TILE_ID]) {
		return;
	}

	// tile metadata
    const uint32_t image_id = GLOBAL_TILE_ID / (tile_width * tile_height);
    const uint32_t tile_id = GLOBAL_TILE_ID % (tile_width * tile_height);
    const uint2 tile = {tile_id % tile_width, tile_id / tile_width};
	const uint2 pix_min = {tile.x * tile_size, tile.y * tile_size};

    // Will be indexed by local pix_id. Rebase pointer to image_id.
    // Except for tile_offsets and masks. They are indexed by GLOBAL_TILE_ID.
    render_alphas += image_id * image_height * image_width;
    render_colors += image_id * image_height * image_width * CDIM;
    render_normals += image_id * image_height * image_width * 3;

    last_ids += image_id * image_height * image_width;
    median_ids += image_id * image_height * image_width;

    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    v_render_normals += image_id * image_height * image_width * 3;
    v_render_median += image_id * image_height * image_width;

    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (v_render_distort != nullptr) {
        v_render_distort += image_id * image_height * image_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[GLOBAL_TILE_ID]) {
        return;
    }

    // Load Gaussian properties into registers
	int32_t gaussian_idx = 0;
	vec2 xy = {0.0f, 0.0f};
    float opac = 0.0f;
    vec3 u_M = {0.0f, 0.0f, 0.0f};
    vec3 v_M = {0.0f, 0.0f, 0.0f};
    vec3 w_M = {0.0f, 0.0f, 0.0f};
	float color[CDIM] = {0.0f};
	float normal[3] = {0.0f};
	if (valid_splat) {
		gaussian_idx = flatten_ids[splat_idx_global];
		xy = means2d[gaussian_idx];
        opac = opacities[gaussian_idx];
        u_M = {
            ray_transforms[gaussian_idx * 9],
            ray_transforms[gaussian_idx * 9 + 1],
            ray_transforms[gaussian_idx * 9 + 2]
        };
        v_M = {
            ray_transforms[gaussian_idx * 9 + 3],
            ray_transforms[gaussian_idx * 9 + 4],
            ray_transforms[gaussian_idx * 9 + 5]
        };
        w_M = {
            ray_transforms[gaussian_idx * 9 + 6],
            ray_transforms[gaussian_idx * 9 + 7],
            ray_transforms[gaussian_idx * 9 + 8]
        };
        // float *ray_transforms_ptr = (float *)(ray_transforms) + gaussian_idx * 9;
        // float4 row1_2_part = *(const float4*) (ray_transforms_ptr);        // Loads M11, M12, M13, M21
        // float4 row2_3_part = *(const float4*) (ray_transforms_ptr + 4);    // Loads M22, M23, M31, M32
        // u_M = {row1_2_part.x, row1_2_part.y, row1_2_part.z};
        // v_M = {row1_2_part.w, row2_3_part.x, row2_3_part.y};
        // w_M = {row2_3_part.z, row2_3_part.w, ray_transforms_ptr[8]};
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            color[k] = colors[gaussian_idx * CDIM + k];
        }
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            normal[k] = normals[gaussian_idx * 3 + k];
        }
	}

    // Gradient accumulation variables
	vec2 Register_dL_dmeans2d_abs = {0.0f, 0.0f};
	vec2 Register_dL_dmeans2d = {0.0f, 0.0f};
	vec3 Register_dL_du_M = {0.0f, 0.0f, 0.0f};
	vec3 Register_dL_dv_M = {0.0f, 0.0f, 0.0f};
	vec3 Register_dL_dw_M = {0.0f, 0.0f, 0.0f};
	float Register_dL_dcolors[CDIM] = {0.0f};
	float Register_dL_dopacities = 0.0f;
	float Register_dL_dnormals[3] = {0.0f};
	// float Register_dL_ddensify = 0.0f;

	// values useful for gradient calculation
    int32_t median_idx = 0;
    float v_median;
	float T;
	float T_final;
    float v_render_a = 0;
	float last_contributor;
	float ar[CDIM];
	float an[3];
    float vis_wd_final;
    float v_distort = 0;
    float d_final;
    float w_final;
    float accum_d;
    float accum_w;
    float accum_vis_wd;
	float dL_dpixel[CDIM];
	float dL_dnpixel[3];
	// const float ddelx_dx = 0.5 * image_width;
	// const float ddely_dy = 0.5 * image_height;

    const uint32_t BLOCK_SIZE = tile_size * tile_size;
    const int32_t tr = my_warp.thread_rank();
    const int32_t batch_factor = 8;

    extern __shared__ int s[];
    int32_t *median_id_batch = (int32_t *)s; // [BLOCK_SIZE/batch_factor]
    float *v_median_batch = (float *)&median_id_batch[BLOCK_SIZE/batch_factor]; // [BLOCK_SIZE/batch_factor]
    float4 *sampled_stat_batch =
        reinterpret_cast<float4 *>(&v_median_batch[BLOCK_SIZE/batch_factor]); // [BLOCK_SIZE/batch_factor]
    float *T_final_batch = (float *)&sampled_stat_batch[BLOCK_SIZE/batch_factor];  // [BLOCK_SIZE/batch_factor]
    float *v_render_a_batch = &T_final_batch[BLOCK_SIZE/batch_factor];  // [BLOCK_SIZE/batch_factor]
    float *last_contributor_batch = &v_render_a_batch[BLOCK_SIZE/batch_factor];  // [BLOCK_SIZE/batch_factor]
    float *render_colors_batch = &last_contributor_batch[BLOCK_SIZE/batch_factor];  // [BLOCK_SIZE/batch_factor * CDIM]
    float *sampled_ar_batch = &render_colors_batch[BLOCK_SIZE/batch_factor * CDIM];  // [BLOCK_SIZE/batch_factor * CDIM]
    float *v_render_colors_batch = &sampled_ar_batch[BLOCK_SIZE/batch_factor * CDIM];  // [BLOCK_SIZE/batch_factor * CDIM]
    float3 *sampled_an_batch = 
        reinterpret_cast<float3 *>(&v_render_colors_batch[BLOCK_SIZE/batch_factor * CDIM]);  // [BLOCK_SIZE/batch_factor]
    float *render_normals_batch = (float *)&sampled_an_batch[BLOCK_SIZE/batch_factor];  // [BLOCK_SIZE/batch_factor * 3]
    float *v_render_normals_batch = &render_normals_batch[BLOCK_SIZE/batch_factor * 3];  // [BLOCK_SIZE/batch_factor * 3]
    float *v_render_distort_batch = nullptr;
    float2 *render_stat_batch = nullptr;
    if (v_render_distort != nullptr) {
        v_render_distort_batch = &v_render_normals_batch[BLOCK_SIZE/batch_factor * 3];  // [BLOCK_SIZE/batch_factor]
        render_stat_batch = 
            reinterpret_cast<float2 *>(&v_render_distort_batch[BLOCK_SIZE/batch_factor]);  // [BLOCK_SIZE/batch_factor]
    }

    //////////////////////// batched run
    int sm_batch = (BLOCK_SIZE/batch_factor) / 32;  // 256/8/32 = 1
    uint32_t inner_i = 0;
    for (int bch = 0; bch < batch_factor; ++bch) {
        /*
            * Fetch pixel data
            */
        block.sync();
        for (int i = 0; i < sm_batch; ++i) {
            int idx = tr * sm_batch + i;   // pix index in tile.
            const uint2 pix = {pix_min.x + (idx + BLOCK_SIZE/batch_factor*bch) % tile_size, pix_min.y + (idx + BLOCK_SIZE/batch_factor*bch) / tile_size};
            const uint32_t pix_id = image_width * pix.y + pix.x;
            bool valid_pixel = pix.x < image_width && pix.y < image_height;
            if (valid_pixel) {
                median_id_batch[idx] = median_ids[pix_id];
                v_median_batch[idx] = v_render_median[pix_id];
                float4* sampled_stats_ptr = (float4*)sampled_stats;
                sampled_stat_batch[idx] = sampled_stats_ptr[global_bucket_idx * BLOCK_SIZE + (idx + BLOCK_SIZE/batch_factor*bch)];      // (sampled_T, sampled_avd, sampled_aw, sampled_avwd)
                T_final_batch[idx] = 1 - render_alphas[pix_id];
                v_render_a_batch[idx] = v_render_alphas[pix_id];
                last_contributor_batch[idx] = last_ids[pix_id];

    #pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    render_colors_batch[idx * CDIM + k] = render_colors[pix_id * CDIM + k];
                    sampled_ar_batch[idx * CDIM + k] = 
                        sampled_ar[global_bucket_idx * BLOCK_SIZE * CDIM + k * BLOCK_SIZE + (idx + BLOCK_SIZE/batch_factor*bch)] + (backgrounds == nullptr ? 0 : T_final_batch[idx] * backgrounds[k]);
                    v_render_colors_batch[idx * CDIM + k] = v_render_colors[pix_id * CDIM + k];
                }

                float4* sampled_an_ptr = (float4*) sampled_an;
                float4 an_tmp = sampled_an_ptr[(global_bucket_idx * BLOCK_SIZE) + (idx + BLOCK_SIZE/batch_factor*bch)];
                sampled_an_batch[idx] = {an_tmp.x, an_tmp.y, an_tmp.z};
    #pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    render_normals_batch[idx * 3 + k] = render_normals[pix_id * 3 + k];
                    v_render_normals_batch[idx * 3 + k] = v_render_normals[pix_id * 3 + k];
                }
                
                if (v_render_distort != nullptr) {
                    v_render_distort_batch[idx] = v_render_distort[pix_id];
                    float2* render_stats_ptr = (float2*)render_stats;
                    render_stat_batch[idx] = render_stats_ptr[pix_id];     // (vis_wd, w)
                }
            }
        }
        // wait for other threads to collect the gaussians
        block.sync();

        // loop over all batches of primitives
        for (; inner_i < BLOCK_SIZE + 31; ++inner_i) {
            // SHUFFLING

            // At this point, T already has my (1 - alpha) multiplied.
            // So pass this ready-made T value to next thread.
            median_idx = my_warp.shfl_up(median_idx, 1);
            v_median = my_warp.shfl_up(v_median, 1);
            T = my_warp.shfl_up(T, 1);
            last_contributor = my_warp.shfl_up(last_contributor, 1);
            T_final = my_warp.shfl_up(T_final, 1);
    #pragma unroll
            for (int k = 0; k < CDIM; ++k) {
                ar[k] = my_warp.shfl_up(ar[k], 1);
                dL_dpixel[k] = my_warp.shfl_up(dL_dpixel[k], 1);
            }
    #pragma unroll
            for (int k = 0; k < 3; ++k) {
                an[k] = my_warp.shfl_up(an[k], 1);
                dL_dnpixel[k] = my_warp.shfl_up(dL_dnpixel[k], 1);
            }
            if (v_render_distort != nullptr) {
                v_distort = my_warp.shfl_up(v_distort, 1);
                vis_wd_final = my_warp.shfl_up(vis_wd_final, 1);
                d_final = my_warp.shfl_up(d_final, 1);
                w_final = my_warp.shfl_up(w_final, 1);
                accum_d = my_warp.shfl_up(accum_d, 1);
                accum_w = my_warp.shfl_up(accum_w, 1);
                accum_vis_wd = my_warp.shfl_up(accum_vis_wd, 1);
            }

            // which pixel index should this thread deal with?
            int idx = inner_i - BLOCK_SIZE/batch_factor*bch - my_warp.thread_rank();        // pix index in tile.
            const uint2 pix = {pix_min.x + (idx + BLOCK_SIZE/batch_factor*bch) % tile_size, pix_min.y + (idx + BLOCK_SIZE/batch_factor*bch) / tile_size};
            // const uint32_t pix_id = image_width * pix.y + pix.x;
            const float2 pixf = {(float) pix.x + 0.5f, (float) pix.y + 0.5f};
            bool valid_pixel = pix.x < image_width && pix.y < image_height;
            
            // every 32nd thread should read the stored state from memory
            if (valid_splat && valid_pixel && my_warp.thread_rank() == 0 && idx < BLOCK_SIZE/batch_factor) {
                median_idx = median_id_batch[idx];
                v_median = v_median_batch[idx];
                float4 sampled_stat = sampled_stat_batch[idx];      // (sampled_T, sampled_avd, sampled_aw, sampled_avwd)
                T = sampled_stat.x;
                T_final = T_final_batch[idx];
                v_render_a = v_render_a_batch[idx];
                last_contributor = last_contributor_batch[idx];
    #pragma unroll
                for (int k = 0; k < CDIM; ++k) {
                    ar[k] = -render_colors_batch[idx * CDIM + k] + sampled_ar_batch[idx * CDIM + k];
                    dL_dpixel[k] = v_render_colors_batch[idx * CDIM + k];
                }
                float3 an_tmp = sampled_an_batch[idx];
                float nx = render_normals_batch[idx * 3];
                float ny = render_normals_batch[idx * 3 + 1];
                float nz = render_normals_batch[idx * 3 + 2];
                an[0] = -nx + an_tmp.x;
                an[1] = -ny + an_tmp.y;
                an[2] = -nz + an_tmp.z;
    #pragma unroll
                for (int k = 0; k < 3; ++k) {
                    dL_dnpixel[k] = v_render_normals_batch[idx * 3 + k];
                }
                if (v_render_distort != nullptr) {
                    v_distort = v_render_distort_batch[idx];
                    float2 render_stat = render_stat_batch[idx];     // (vis_wd, w)
                    vis_wd_final = render_stat.x;
                    w_final = render_stat.y;
                    d_final = render_colors_batch[idx * CDIM + CDIM - 1];
                    accum_d = sampled_stat.y;
                    accum_w = sampled_stat.z;
                    accum_vis_wd = sampled_stat.w;
                }
            }

            // do work
            if (valid_splat && valid_pixel && 0 <= idx && idx < BLOCK_SIZE/batch_factor) {
                if (image_width <= pix.x || image_height <= pix.y) continue;

                if (splat_idx_global > last_contributor) continue;

                vec3 h_u = pixf.x * w_M - u_M;
                vec3 h_v = pixf.y * w_M - v_M;
                vec3 ray_cross = glm::cross(h_u, h_v);

                // no ray_crossion
                if (ray_cross.z == 0.0) continue;

                vec2 s = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};

                // GAUSSIAN KERNEL EVALUATION
                float gauss_weight_3d = s.x * s.x + s.y * s.y;
                vec2 d = {xy.x - pixf.x, xy.y - pixf.y};

                // 2D gaussian weight using the projected 2D mean
                float gauss_weight_2d =
                    FILTER_INV_SQUARE_2DGS * (d.x * d.x + d.y * d.y);
                float gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

                // visibility and alpha
                const float sigma = 0.5f * gauss_weight;
                float vis = __expf(-sigma);
                float alpha = min(0.999f, opac * vis); // clipped alpha

                // gaussian throw out
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) continue;

                // gradient contribution from median depth
                if (splat_idx_global == median_idx) {
                    // v_median is a special gradient input from forward pass
                    // not yet clear what this is for
                    Register_dL_dcolors[CDIM - 1] += v_median;
                }

                /**
                    * d(img)/d(rgb) and d(img)/d(alpha)
                    */

                // compute the current T for this gaussian
                // since the output T = coprod (1 - alpha_i), we have T_(inner_i-1) =
                // T_i * 1/(1 - alpha_(inner_i-1)) potential numerical stability issue
                // if alpha -> 1
                float ra = 1.0f / (1.0f - alpha);

                // update v_rgb for this gaussian
                // because the weight is computed as: c_i (a_i G_i) * T : T =
                // prod{1, inner_i-1}(1 - a_j G_j) we have d(img)/d(c_i) = (a_i G_i) *
                // T where alpha_i is a_i * G_i
                const float weight = alpha * T;
                float dL_dalpha = 0.0f;
                for (uint32_t k = 0; k < CDIM; ++k) {
                    ar[k] += weight * color[k];
                    const float &dL_dchannel = dL_dpixel[k];
                    Register_dL_dcolors[k] += weight * dL_dchannel;
                    
                    dL_dalpha += ((color[k] * T) - ra * (-ar[k])) * dL_dchannel;
                }

                for (uint32_t k = 0; k < 3; ++k) {
                    an[k] += weight * normal[k];
                    const float &dL_dnchannel = dL_dnpixel[k];
                    Register_dL_dnormals[k] += weight * dL_dnchannel;

                    dL_dalpha += (normal[k] * T - ra * (-an[k])) * dL_dnchannel;
                }

                /*
                * d(alpha_out) / d(alpha)
                */
                dL_dalpha += T_final * ra * v_render_a;
                
                if (backgrounds != nullptr) {
                    float bg_dot_dpixel = 0.0f;
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        bg_dot_dpixel += backgrounds[k] * dL_dpixel[k];
                    }
                    dL_dalpha += -T_final * ra * bg_dot_dpixel;
                }
                
                // contribution from distortion
                if (v_render_distort != nullptr) {
                    // last channel of colors is depth
                    float depth = color[CDIM - 1];
                    accum_d += weight * depth;
                    accum_w += weight;
                    accum_vis_wd += weight * (depth * accum_w - accum_d);
                    float distort_buffer = 
                        2.0f * 
                        (2.0f * (vis_wd_final - accum_vis_wd) -
                            (d_final * accum_w - w_final * accum_d));
                    float dl_dw =
                        2.0f *
                        (2.0f * (depth * accum_w - accum_d) +
                            (d_final - depth * w_final));
                    // df / d(alpha)
                    dL_dalpha += (dl_dw * T - distort_buffer * ra) * v_distort;
                    // df / d(depth). put it in the last channel of v_rgb
                    Register_dL_dcolors[CDIM - 1] += 2.0f * weight *
                                                (2.0f - 2.0f * T - w_final + weight) *
                                                v_distort;
                }

                T *= (1.0f - alpha);

                /** ==================================================
                    * 2DGS backward pass: compute gradients of d_out / d_G_i and
                    * d_G_i w.r.t geometry parameters
                    * ==================================================
                    */
                if (opac * vis <= 0.999f) {
                    float v_depth = 0.f;
                    // d(a_i * G_i) / d(G_i) = a_i
                    const float dL_dG = opac * dL_dalpha;

                    // case 1: in the forward pass, the proper ray-primitive
                    // intersection is used
                    if (gauss_weight_3d <= gauss_weight_2d) {

                        // derivative of G_i w.r.t. ray-primitive intersection
                        // uv coordinates
                        const vec2 v_s = {
                            dL_dG * -vis * s.x + v_depth * w_M.x,
                            dL_dG * -vis * s.y + v_depth * w_M.y
                        };

                        // backward through the projective transform
                        // @see rasterize_to_pixels_2dgs_fwd.cu to understand
                        // what is going on here
                        const vec3 v_z_w_M = {s.x, s.y, 1.0};
                        const float v_sx_pz = v_s.x / ray_cross.z;
                        const float v_sy_pz = v_s.y / ray_cross.z;
                        const vec3 v_ray_cross = {
                            v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y)
                        };
                        const vec3 v_h_u = glm::cross(h_v, v_ray_cross);
                        const vec3 v_h_v = glm::cross(v_ray_cross, h_u);

                        // derivative of ray-primitive intersection uv
                        // coordinates w.r.t. transformation (geometry)
                        // coefficients
                        Register_dL_du_M += {-v_h_u.x, -v_h_u.y, -v_h_u.z};
                        Register_dL_dv_M += {-v_h_v.x, -v_h_v.y, -v_h_v.z};
                        Register_dL_dw_M += {
                            pixf.x * v_h_u.x + pixf.y * v_h_v.x + v_depth * v_z_w_M.x,
                            pixf.x * v_h_u.y + pixf.y * v_h_v.y + v_depth * v_z_w_M.y,
                            pixf.x * v_h_u.z + pixf.y * v_h_v.z + v_depth * v_z_w_M.z
                        };

                        // case 2: in the forward pass, the 2D gaussian
                        // projected gaussian weight is used
                    } else {
                        // computing the derivative of G_i w.r.t. 2d projected
                        // gaussian parameters (trivial)
                        const float v_G_ddelx =
                            -vis * FILTER_INV_SQUARE_2DGS * d.x;
                        const float v_G_ddely =
                            -vis * FILTER_INV_SQUARE_2DGS * d.y;
                            Register_dL_dmeans2d += {dL_dG * v_G_ddelx, dL_dG * v_G_ddely};
                        if (v_means2d_abs != nullptr) {
                            Register_dL_dmeans2d_abs += {
                                abs(Register_dL_dmeans2d.x), abs(Register_dL_dmeans2d.y)
                            };
                        }
                    }
                    Register_dL_dopacities += vis * dL_dalpha;
                }
            }
        }
    }

    // finally add the gradients using atomics
	if (valid_splat) {
        float *v_rgb_ptr = (float *)(v_colors) + CDIM * gaussian_idx;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            gpuAtomicAdd(v_rgb_ptr + k, Register_dL_dcolors[k]);
        }

        float *v_normal_ptr = (float *)(v_normals) + 3 * gaussian_idx;
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            gpuAtomicAdd(v_normal_ptr + k, Register_dL_dnormals[k]);
        }

        float *v_ray_transforms_ptr =
            (float *)(v_ray_transforms) + 9 * gaussian_idx;
        gpuAtomicAdd(v_ray_transforms_ptr, Register_dL_du_M.x);
        gpuAtomicAdd(v_ray_transforms_ptr + 1, Register_dL_du_M.y);
        gpuAtomicAdd(v_ray_transforms_ptr + 2, Register_dL_du_M.z);
        gpuAtomicAdd(v_ray_transforms_ptr + 3, Register_dL_dv_M.x);
        gpuAtomicAdd(v_ray_transforms_ptr + 4, Register_dL_dv_M.y);
        gpuAtomicAdd(v_ray_transforms_ptr + 5, Register_dL_dv_M.z);
        gpuAtomicAdd(v_ray_transforms_ptr + 6, Register_dL_dw_M.x);
        gpuAtomicAdd(v_ray_transforms_ptr + 7, Register_dL_dw_M.y);
        gpuAtomicAdd(v_ray_transforms_ptr + 8, Register_dL_dw_M.z);
        
        float *v_xy_ptr = (float *)(v_means2d) + 2 * gaussian_idx;
        gpuAtomicAdd(v_xy_ptr, Register_dL_dmeans2d.x);
        gpuAtomicAdd(v_xy_ptr + 1, Register_dL_dmeans2d.y);

        if (v_means2d_abs != nullptr) {
            float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * gaussian_idx;
            gpuAtomicAdd(v_xy_abs_ptr, Register_dL_dmeans2d_abs.x);
            gpuAtomicAdd(v_xy_abs_ptr + 1, Register_dL_dmeans2d_abs.y);
        }

        gpuAtomicAdd(v_opacities + gaussian_idx, Register_dL_dopacities);
        
        float *v_densify_ptr = (float *)(v_densify) + 2 * gaussian_idx;
        // float *v_ray_transforms_ptr =
        //     (float *)(v_ray_transforms) + 9 * gaussian_idx;
        float depth = w_M.z;
        v_densify_ptr[0] = v_ray_transforms_ptr[2] * depth;
        v_densify_ptr[1] = v_ray_transforms_ptr[5] * depth;
	}
}

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_2dgs_bwd_kernel(
    const uint32_t I,        // number of images
    const uint32_t N,        // number of gaussians
    const uint32_t n_isects, // number of ray-primitive intersections.
    const bool packed,       // whether the input tensors are packed
    // fwd inputs
    const vec2
        *__restrict__ means2d, // Projected Gaussian means. [..., N, 2] if
                               // packed is False, [nnz, 2] if packed is True.
    const scalar_t
        *__restrict__ ray_transforms, // transformation matrices that transforms
                                      // xy-planes in pixel spaces into splat
                                      // coordinates. [..., N, 3, 3] if packed is
                                      // False, [nnz, channels] if packed is
                                      // True. This is (KWH)^{-1} in the paper
                                      // (takes screen [x,y] and map to [u,v])
    const scalar_t *__restrict__ colors,  // [..., N, CDIM] or [nnz, CDIM]  //
                                          // Gaussian colors or ND features.
    const scalar_t *__restrict__ normals, // [..., N, 3] or [nnz, 3] // The
                                          // normals in camera space.
    const scalar_t *__restrict__ opacities, // [..., N] or [nnz] // Gaussian
                                            // opacities that support per-view
                                            // values.
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] // Background colors
                                              // on camera basis
    const bool *__restrict__ masks, // [..., tile_height, tile_width]     //
                                    // Optional tile mask to skip rendering GS
                                    // to masked tiles.

    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t *__restrict__ tile_offsets, // [..., tile_height, tile_width]
    const int32_t *__restrict__ flatten_ids,  // [n_isects]

    // fwd outputs
    const scalar_t *__restrict__ render_colors, // [..., image_height,
                                                // image_width, CDIM]
    const scalar_t
        *__restrict__ render_alphas, // [..., image_height, image_width, 1]
    const int32_t
        *__restrict__ last_ids, // [..., image_height, image_width]     // the id
                                // to last gaussian that got intersected
    const int32_t *__restrict__ median_ids, // [..., image_height, image_width] //
                                            // the id to the gaussian that
                                            // brings the opacity over 0.5

    // grad outputs
    const scalar_t
        *__restrict__ v_render_colors, // [..., image_height, image_width,     //
                                       // RGB CDIM]
    const scalar_t
        *__restrict__ v_render_alphas, // [..., image_height, image_width, 1]  //
                                       // total opacities.
    const scalar_t
        *__restrict__ v_render_normals, // [..., image_height, image_width, 3]  //
                                        // camera space normals
    const scalar_t
        *__restrict__ v_render_distort, // [..., image_height, image_width, 1]  //
                                        // mip-nerf 360 distorts
    const scalar_t
        *__restrict__ v_render_median, // [..., image_height, image_width, 1]  //
                                       // the median depth

    // grad inputs
    vec2 *__restrict__ v_means2d_abs,        // [..., N, 2] or [nnz, 2]
    vec2 *__restrict__ v_means2d,            // [..., N, 2] or [nnz, 2]
    scalar_t *__restrict__ v_ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    scalar_t *__restrict__ v_colors,         // [..., N, CDIM] or [nnz, CDIM]
    scalar_t *__restrict__ v_opacities,      // [..., N] or [nnz]
    scalar_t *__restrict__ v_normals,        // [..., N, 3] or [nnz, 3]
    scalar_t *__restrict__ v_densify
) {
    /**
     * ==============================
     * Set up the thread blocks
     * blocks are assigned tilewise, and threads are assigned pixelwise
     * ==============================
     */
    auto block = cg::this_thread_block();
    uint32_t image_id = block.group_index().x;
    uint32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets += image_id * tile_height * tile_width;
    render_alphas += image_id * image_height * image_width;
    render_colors += image_id * image_height * image_width * CDIM;

    last_ids += image_id * image_height * image_width;
    median_ids += image_id * image_height * image_width;

    v_render_colors += image_id * image_height * image_width * CDIM;
    v_render_alphas += image_id * image_height * image_width;
    v_render_normals += image_id * image_height * image_width * 3;
    v_render_median += image_id * image_height * image_width;

    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }
    if (v_render_distort != nullptr) {
        v_render_distort += image_id * image_height * image_width;
    }

    // when the mask is provided, do nothing and return if
    // this tile is labeled as False
    if (masks != nullptr && !masks[tile_id]) {
        return;
    }

    const float px = (float)j + 0.5f;
    const float py = (float)i + 0.5f;
    // clamp this value to the last pixel
    const int32_t pix_id =
        min(i * image_width + j, image_width * image_height - 1);

    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    // number of batches needed to process all gaussians in this tile
    const uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    /**
     * ==============================
     * Memory Allocation
     * Memory is laid out as:
     * | pix_x : pix_y : opac | u_M : v_M : w_M | rgb : normal |
     * ==============================
     */

    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]

    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]
    vec3 *u_Ms_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    vec3 *v_Ms_batch =
        reinterpret_cast<vec3 *>(&u_Ms_batch[block_size]); // [block_size]
    vec3 *w_Ms_batch =
        reinterpret_cast<vec3 *>(&v_Ms_batch[block_size]); // [block_size]

    // extended memory block
    float *rgbs_batch = (float *)&w_Ms_batch[block_size]; // [block_size * CDIM]
    float *normals_batch = &rgbs_batch[block_size * CDIM]; // [block_size * 3]

    // this is the T AFTER the last gaussian in this pixel
    float T_final = 1.0f - render_alphas[pix_id];
    float T = T_final;

    // the contribution from gaussians behind the current one
    // this is used to compute d(alpha)/d(c_i)
    float buffer[CDIM] = {0.f};
    float buffer_normals[3] = {0.f};

    // index of last gaussian to contribute to this pixel
    const int32_t bin_final = inside ? last_ids[pix_id] : 0;

    // index of gaussian that contributes to median depth
    const int32_t median_idx = inside ? median_ids[pix_id] : 0;

    /**
     * ==============================
     * Fetching Data
     * ==============================
     */

    // df/d_out for this pixel (within register)
    // FETCH COLOR GRADIENT
    float v_render_c[CDIM];
#pragma unroll
    for (uint32_t k = 0; k < CDIM; ++k) {
        v_render_c[k] = v_render_colors[pix_id * CDIM + k];
    }

    // FETCH ALPHA GRADIENT
    const float v_render_a = v_render_alphas[pix_id];
    float v_render_n[3];

// FETCH NORMAL GRADIENT (NORMALIZATION FOR 2DGS)
#pragma unroll
    for (uint32_t k = 0; k < 3; ++k) {
        v_render_n[k] = v_render_normals[pix_id * 3 + k];
    }

    // PREPARE FOR DISTORTION (IF DISTORSION LOSS ENABLED)
    float v_distort = 0.f;
    float accum_d, accum_w;
    float accum_d_buffer, accum_w_buffer, distort_buffer;
    if (v_render_distort != nullptr) {
        v_distort = v_render_distort[pix_id];
        // last channel of render_colors is accumulated depth
        accum_d_buffer = render_colors[pix_id * CDIM + CDIM - 1];
        accum_d = accum_d_buffer;
        accum_w_buffer = render_alphas[pix_id];
        accum_w = accum_w_buffer;
        distort_buffer = 0.f;
    }

    // median depth gradients
    float v_median = v_render_median[pix_id];

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing
    const uint32_t tr = block.thread_rank();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    // find the maximum final gaussian ids in the thread warp.
    // this gives the last gaussian id that have intersected with any pixels in
    // the warp
    const int32_t warp_bin_final =
        cg::reduce(warp, bin_final, cg::greater<int>());

    /**
     * =======================================================
     * Calculating Derivatives
     * =======================================================
     */
    // loop over all batches of primitives
    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before writing next batch of shared mem
        block.sync();

        // each thread fetch 1 gaussian from back to front
        // 0 index will be furthest back in batch
        // index of gaussian to load
        // batch end is the index of the last gaussian in the batch
        // These values can be negative so must be int32 instead of uint32

        // loop factors:
        // we start with loop end and interate backwards
        const int32_t batch_end = range_end - 1 - block_size * b;
        const int32_t batch_size = min(block_size, batch_end + 1 - range_start);

        // VERY IMPORTANT HERE!
        // we are looping from back to front
        // so we are processing the gaussians in the order of closest to
        // furthest if you use symbolic solver on splatting rendering equations
        // you will see
        const int32_t idx = batch_end - tr;

        /*
         * Fetch Gaussian Primitives and STORE THEM IN REVERSE ORDER
         */
        if (idx >= range_start) {
            int32_t g = flatten_ids[idx]; // flatten index in [I * N] or [nnz]
            id_batch[tr] = g;
            const vec2 xy = means2d[g];
            const float opac = opacities[g];
            xy_opacity_batch[tr] = {xy.x, xy.y, opac};

            u_Ms_batch[tr] = {
                ray_transforms[g * 9],
                ray_transforms[g * 9 + 1],
                ray_transforms[g * 9 + 2]
            };
            v_Ms_batch[tr] = {
                ray_transforms[g * 9 + 3],
                ray_transforms[g * 9 + 4],
                ray_transforms[g * 9 + 5]
            };
            w_Ms_batch[tr] = {
                ray_transforms[g * 9 + 6],
                ray_transforms[g * 9 + 7],
                ray_transforms[g * 9 + 8]
            };
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                rgbs_batch[tr * CDIM + k] = colors[g * CDIM + k];
            }
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                normals_batch[tr * 3 + k] = normals[g * 3 + k];
            }
        }
        // wait for other threads to collect the gaussians in batch
        block.sync();
        // loops through the gaussians in the current batch for this pixel
        // 0 index is the furthest back gaussian in the batch
        /**
         * ==================================================
         * BACKWARD LOOPING THROUGH PRIMITIVES
         * ==================================================
         */
        for (uint32_t t = max(0, batch_end - warp_bin_final); t < batch_size;
             ++t) {

            bool valid = inside;
            if (batch_end - t > bin_final) {
                valid = 0;
            }

            /**
             * ==================================================
             * Forward pass variables
             * ==================================================
             */
            float alpha; // for the currently processed gaussian, per pixel
            float
                opac; // opacity of the currently processed gaussian, per pixel
            float
                vis; // visibility of the currently processed gaussian (the pure
                     // gaussian weight, not multiplied by opacity), per pixel
            float gauss_weight_3d; // 3D gaussian weight (using the proper
                                   // intersection of UV space), per pixel
            float gauss_weight_2d; // 2D gaussian weight (using the projected 2D
                                   // mean), per pixel
            float gauss_weight;    // minimum of 3D and 2D gaussian weights, per
                                   // pixel

            vec2 s;   // normalized point of intersection on the uv, per pixel
            vec2 d;   // position on uv plane with respect to the primitive
                      // center, per pixel
            vec3 h_u; // homogeneous plane parameter for us, per pixel
            vec3 h_v; // homogeneous plane parameter for vs, per pixel
            vec3 ray_cross; // ray cross product, the ray of plane intersection,
                            // per pixel
            vec3 w_M; // depth component of the ray transform matrix, per pixel

            /**
             * ==================================================
             * Run through the forward pass, but only for the t-th primitive
             * ==================================================
             */
            if (valid) {
                vec3 xy_opac = xy_opacity_batch[t];

                opac = xy_opac.z;

                const vec3 u_M = u_Ms_batch[t];
                const vec3 v_M = v_Ms_batch[t];

                w_M = w_Ms_batch[t];

                h_u = px * w_M - u_M;
                h_v = py * w_M - v_M;

                ray_cross = glm::cross(h_u, h_v);

                // no ray_crossion
                if (ray_cross.z == 0.0)
                    valid = false;
                s = {ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z};

                // GAUSSIAN KERNEL EVALUATION
                gauss_weight_3d = s.x * s.x + s.y * s.y;
                d = {xy_opac.x - px, xy_opac.y - py};

                // 2D gaussian weight using the projected 2D mean
                gauss_weight_2d =
                    FILTER_INV_SQUARE_2DGS * (d.x * d.x + d.y * d.y);
                gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

                // visibility and alpha
                const float sigma = 0.5f * gauss_weight;
                vis = __expf(-sigma);
                alpha = min(0.999f, opac * vis); // clipped alpha

                // gaussian throw out
                if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                    valid = false;
                }
            }

            // if all threads are inactive in this warp, skip this loop
            if (!warp.any(valid)) {
                continue;
            }

            /**
             * ==================================================
             * Gradient variables
             *
             * note: the "local" suffix means this is the gradient of a
             * primitive for a pixel in the end of the loops, we will reduce sum
             * over all threads in the block to get the final gradient
             * ==================================================
             */
            // rgb gradients
            float v_rgb_local[CDIM] = {0.f};
            // normal gradients
            float v_normal_local[3] = {0.f};

            // ray transform gradients
            vec3 v_u_M_local = {0.f, 0.f, 0.f};
            vec3 v_v_M_local = {0.f, 0.f, 0.f};
            vec3 v_w_M_local = {0.f, 0.f, 0.f};

            // 2D mean gradients, used if 2d gaussian weight is applied
            vec2 v_xy_local = {0.f, 0.f};

            // absolute 2D mean gradients, used if 2d gaussian weight is applied
            vec2 v_xy_abs_local = {0.f, 0.f};

            // opacity gradients
            float v_opacity_local = 0.f;

            // initialize everything to 0, only set if the lane is valid
            /**
             * ==================================================
             * Calculating Derivatives w.r.t current primitive / gaussian
             * ==================================================
             */
            if (valid) {

                // gradient contribution from median depth
                if (batch_end - t == median_idx) {
                    // v_median is a special gradient input from forward pass
                    // not yet clear what this is for
                    v_rgb_local[CDIM - 1] += v_median;
                }

                /**
                 * d(img)/d(rgb) and d(img)/d(alpha)
                 */

                // compute the current T for this gaussian
                // since the output T = coprod (1 - alpha_i), we have T_(i-1) =
                // T_i * 1/(1 - alpha_(i-1)) potential numerical stability issue
                // if alpha -> 1
                float ra = 1.0f / (1.0f - alpha);
                T *= ra;

                // update v_rgb for this gaussian
                // because the weight is computed as: c_i (a_i G_i) * T : T =
                // prod{1, i-1}(1 - a_j G_j) we have d(img)/d(c_i) = (a_i G_i) *
                // T where alpha_i is a_i * G_i
                const float fac = alpha * T;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_rgb_local[k] += fac * v_render_c[k];
                }

                // contribution from this pixel to alpha
                // we have d(alpha)/d(c_i) = c_i * G_i * T + [grad contribution
                // from following gaussians in T term] this can be proven by
                // symbolic differentiation of a_i with respect to c_out
                float v_alpha = 0.f;
                for (uint32_t k = 0; k < CDIM; ++k) {
                    v_alpha += (rgbs_batch[t * CDIM + k] * T - buffer[k] * ra) *
                               v_render_c[k];
                }

/*
 * d(normal_out) / d(rgb) and d(normal_out) / d(alpha)
 */

// update v_normal for this gaussian
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    v_normal_local[k] = fac * v_render_n[k];
                }

                for (uint32_t k = 0; k < 3; ++k) {
                    v_alpha += (normals_batch[t * 3 + k] * T -
                                buffer_normals[k] * ra) *
                               v_render_n[k];
                }

                /*
                 * d(alpha_out) / d(alpha)
                 */
                v_alpha += T_final * ra * v_render_a;

                // adjust the alpha gradients by background color
                // this prevents the background rendered in the fwd pass being
                // considered as inaccuracies in primitives this allows us to
                // swtich background colors to prevent overfitting to particular
                // backgrounds i.e. black
                if (backgrounds != nullptr) {
                    float accum = 0.f;
#pragma unroll
                    for (uint32_t k = 0; k < CDIM; ++k) {
                        accum += backgrounds[k] * v_render_c[k];
                    }
                    v_alpha += -T_final * ra * accum;
                }

                // contribution from distortion
                if (v_render_distort != nullptr) {
                    // last channel of colors is depth
                    float depth = rgbs_batch[t * CDIM + CDIM - 1];
                    float dl_dw =
                        2.0f *
                        (2.0f * (depth * accum_w_buffer - accum_d_buffer) +
                         (accum_d - depth * accum_w));
                    // df / d(alpha)
                    v_alpha += (dl_dw * T - distort_buffer * ra) * v_distort;
                    accum_d_buffer -= fac * depth;
                    accum_w_buffer -= fac;
                    distort_buffer += dl_dw * fac;
                    // df / d(depth). put it in the last channel of v_rgb
                    v_rgb_local[CDIM - 1] += 2.0f * fac *
                                             (2.0f - 2.0f * T - accum_w + fac) *
                                             v_distort;
                }

                /** ==================================================
                 * 2DGS backward pass: compute gradients of d_out / d_G_i and
                 * d_G_i w.r.t geometry parameters
                 * ==================================================
                 */
                if (opac * vis <= 0.999f) {
                    float v_depth = 0.f;
                    // d(a_i * G_i) / d(G_i) = a_i
                    const float v_G = opac * v_alpha;

                    // case 1: in the forward pass, the proper ray-primitive
                    // intersection is used
                    if (gauss_weight_3d <= gauss_weight_2d) {

                        // derivative of G_i w.r.t. ray-primitive intersection
                        // uv coordinates
                        const vec2 v_s = {
                            v_G * -vis * s.x + v_depth * w_M.x,
                            v_G * -vis * s.y + v_depth * w_M.y
                        };

                        // backward through the projective transform
                        // @see rasterize_to_pixels_2dgs_fwd.cu to understand
                        // what is going on here
                        const vec3 v_z_w_M = {s.x, s.y, 1.0};
                        const float v_sx_pz = v_s.x / ray_cross.z;
                        const float v_sy_pz = v_s.y / ray_cross.z;
                        const vec3 v_ray_cross = {
                            v_sx_pz, v_sy_pz, -(v_sx_pz * s.x + v_sy_pz * s.y)
                        };
                        const vec3 v_h_u = glm::cross(h_v, v_ray_cross);
                        const vec3 v_h_v = glm::cross(v_ray_cross, h_u);

                        // derivative of ray-primitive intersection uv
                        // coordinates w.r.t. transformation (geometry)
                        // coefficients
                        v_u_M_local = {-v_h_u.x, -v_h_u.y, -v_h_u.z};
                        v_v_M_local = {-v_h_v.x, -v_h_v.y, -v_h_v.z};
                        v_w_M_local = {
                            px * v_h_u.x + py * v_h_v.x + v_depth * v_z_w_M.x,
                            px * v_h_u.y + py * v_h_v.y + v_depth * v_z_w_M.y,
                            px * v_h_u.z + py * v_h_v.z + v_depth * v_z_w_M.z
                        };

                        // case 2: in the forward pass, the 2D gaussian
                        // projected gaussian weight is used
                    } else {
                        // computing the derivative of G_i w.r.t. 2d projected
                        // gaussian parameters (trivial)
                        const float v_G_ddelx =
                            -vis * FILTER_INV_SQUARE_2DGS * d.x;
                        const float v_G_ddely =
                            -vis * FILTER_INV_SQUARE_2DGS * d.y;
                        v_xy_local = {v_G * v_G_ddelx, v_G * v_G_ddely};
                        if (v_means2d_abs != nullptr) {
                            v_xy_abs_local = {
                                abs(v_xy_local.x), abs(v_xy_local.y)
                            };
                        }
                    }
                    v_opacity_local = vis * v_alpha;
                }

/**
 * Update the cumulative "later" gaussian contributions, used in derivatives of
 * render with respect to alphas
 */
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    buffer[k] += rgbs_batch[t * CDIM + k] * fac;
                }

/**
 * Update the cumulative "later" gaussian contributions, used in derivatives of
 * output normals w.r.t. alphas
 */
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    buffer_normals[k] += normals_batch[t * 3 + k] * fac;
                }
            }

            /**
             * ==================================================
             * Warp-level reduction to compute the sum of the gradients for each
             * gaussian
             * ==================================================
             */
            warpSum<CDIM>(v_rgb_local, warp);
            warpSum<3>(v_normal_local, warp);
            warpSum(v_xy_local, warp);
            warpSum(v_u_M_local, warp);
            warpSum(v_v_M_local, warp);
            warpSum(v_w_M_local, warp);
            if (v_means2d_abs != nullptr) {
                warpSum(v_xy_abs_local, warp);
            }
            warpSum(v_opacity_local, warp);
            int32_t g = id_batch[t]; // flatten index in [I * N] or [nnz]

            /**
             * ==================================================
             * Write the gradients to the global memory
             * ==================================================
             */
            if (warp.thread_rank() == 0) {
                float *v_rgb_ptr = (float *)(v_colors) + CDIM * g;
#pragma unroll
                for (uint32_t k = 0; k < CDIM; ++k) {
                    gpuAtomicAdd(v_rgb_ptr + k, v_rgb_local[k]);
                }

                float *v_normal_ptr = (float *)(v_normals) + 3 * g;
#pragma unroll
                for (uint32_t k = 0; k < 3; ++k) {
                    gpuAtomicAdd(v_normal_ptr + k, v_normal_local[k]);
                }

                float *v_ray_transforms_ptr =
                    (float *)(v_ray_transforms) + 9 * g;
                gpuAtomicAdd(v_ray_transforms_ptr, v_u_M_local.x);
                gpuAtomicAdd(v_ray_transforms_ptr + 1, v_u_M_local.y);
                gpuAtomicAdd(v_ray_transforms_ptr + 2, v_u_M_local.z);
                gpuAtomicAdd(v_ray_transforms_ptr + 3, v_v_M_local.x);
                gpuAtomicAdd(v_ray_transforms_ptr + 4, v_v_M_local.y);
                gpuAtomicAdd(v_ray_transforms_ptr + 5, v_v_M_local.z);
                gpuAtomicAdd(v_ray_transforms_ptr + 6, v_w_M_local.x);
                gpuAtomicAdd(v_ray_transforms_ptr + 7, v_w_M_local.y);
                gpuAtomicAdd(v_ray_transforms_ptr + 8, v_w_M_local.z);

                float *v_xy_ptr = (float *)(v_means2d) + 2 * g;
                gpuAtomicAdd(v_xy_ptr, v_xy_local.x);
                gpuAtomicAdd(v_xy_ptr + 1, v_xy_local.y);

                if (v_means2d_abs != nullptr) {
                    float *v_xy_abs_ptr = (float *)(v_means2d_abs) + 2 * g;
                    gpuAtomicAdd(v_xy_abs_ptr, v_xy_abs_local.x);
                    gpuAtomicAdd(v_xy_abs_ptr + 1, v_xy_abs_local.y);
                }

                gpuAtomicAdd(v_opacities + g, v_opacity_local);
            }

            if (valid) {
                float *v_densify_ptr = (float *)(v_densify) + 2 * g;
                float *v_ray_transforms_ptr =
                    (float *)(v_ray_transforms) + 9 * g;
                float depth = w_M.z;
                v_densify_ptr[0] = v_ray_transforms_ptr[2] * depth;
                v_densify_ptr[1] = v_ray_transforms_ptr[5] * depth;
            }
        }
    }
}

bool all_close_custom(const at::Tensor& a, const at::Tensor& b, float atol) {
    // 1. Calculate the element-wise absolute difference: |a - b|
    at::Tensor diff = at::abs(a - b);
    
    // 2. Compare the difference to the tolerance: |a - b| <= atol
    at::Tensor comparison = at::le(diff, atol); // element-wise Less than or Equal
    
    // 3. Check if ALL elements are True
    return at::all(comparison).item<bool>();
}

void tensor_diff_check(const at::Tensor& a, const at::Tensor& b, float atol) {
    // 1. Prepare and calculate difference
    at::Tensor flat_a = a.flatten();
    at::Tensor flat_b = b.flatten();

    at::Tensor diff = at::abs(flat_a - flat_b);
    
    // 2. Create the non-matching mask and find indices
    at::Tensor comparison = at::le(diff, atol);
    at::Tensor non_matching_mask = at::logical_not(comparison);
    
    // NOTE: false_indices contains the linear indices of non-matching elements
    at::Tensor false_indices = at::where(non_matching_mask)[0];
    int64_t count = false_indices.numel();

    if (count == 0) {
        printf("Tensors A and B are all within the tolerance (atol=%.7f).\n", atol);
        return;
    }

    // 3. Index and gather problematic values
    at::Tensor problematic_values_a = at::index(flat_a, {false_indices});
    at::Tensor problematic_values_b = at::index(flat_b, {false_indices});

    // 4. Move data to CPU (Standard for printing)
    at::Tensor a_cpu = problematic_values_a.cpu();
    at::Tensor b_cpu = problematic_values_b.cpu();
    
    // --- FIX for Segfault: Ensure Index Tensor is CPU int64_t ---
    // Cast index tensor to int64_t (kLong) and move to CPU for safe access
    at::Tensor indices_cpu_long = false_indices.to(at::kLong).cpu();
    const int64_t* original_indices = indices_cpu_long.data_ptr<int64_t>();
    // -----------------------------------------------------------

    // Get data pointers for float values
    const float* a_data = a_cpu.data_ptr<float>();
    const float* b_data = b_cpu.data_ptr<float>();

    printf("\n--- Problematic Values (A vs B) ---\n");
    printf("Linear Index | Value A (%.7f) | Value B (%.7f) | Difference\n");
    printf("----------------------------------------------------------\n");

    // 5. Iterate and print
    for (int64_t i = 0; i < count; ++i) {
        float abs_diff = fabsf(a_data[i] - b_data[i]);
        
        // Use the safely accessed pointer
        int64_t original_index = original_indices[i]; 

        printf("%12lld | %15.7f | %15.7f | %.7f\n", 
               (long long)original_index, a_data[i], b_data[i], abs_diff);
    }

    // --- FIX for last printf line ---
    // The previous line printed garbage due to wrong format specifiers (%d for int64_t).
    printf("Total non-matching count: %lld (Size of Problematic Tensors: %lld)\n", 
           (long long)count, (long long)a_cpu.numel());
}

// template <uint32_t CDIM>
// void launch_rasterize_to_pixels_2dgs_bwd_kernel(
//     // Gaussian parameters
//     const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
//     const at::Tensor ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
//     const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
//     const at::Tensor opacities,                 // [..., N] or [nnz]
//     const at::Tensor normals,                   // [..., N, 3] or [nnz, 3]
//     const at::Tensor densify,                   // [..., N, 2] or [nnz, 2]
//     const at::optional<at::Tensor> backgrounds, // [..., CDIM]
//     const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
//     // image size
//     const uint32_t image_width,
//     const uint32_t image_height,
//     const uint32_t tile_size,
//     // ray_crossions
//     const at::Tensor tile_offsets, // [..., tile_height, tile_width]
//     const at::Tensor flatten_ids,  // [n_isects]
//     // forward outputs
//     const at::Tensor render_colors, // [..., image_height, image_width, CDIM]
//     const at::Tensor render_alphas, // [..., image_height, image_width, 1]
//     const at::Tensor render_normals, // [..., image_height, image_width, 3]
//     const at::Tensor render_stats, // [..., image_height, image_width, 2]
//     const at::Tensor last_ids,      // [..., image_height, image_width]
//     const at::Tensor median_ids,    // [..., image_height, image_width]
//     // efficient backward
//     const uint32_t num_buckets,
//     const at::Tensor per_tile_bucket_offset,
//     const at::Tensor bucket_to_tile,
//     const at::Tensor sampled_stats,
//     const at::Tensor sampled_ar,
//     const at::Tensor sampled_an,
//     const at::Tensor max_contrib,
//     // gradients of outputs
//     const at::Tensor v_render_colors,  // [..., image_height, image_width, 3]
//     const at::Tensor v_render_alphas,  // [..., image_height, image_width, 1]
//     const at::Tensor v_render_normals, // [..., image_height, image_width, 3]
//     const at::Tensor v_render_distort, // [..., image_height, image_width, 1]
//     const at::Tensor v_render_median,  // [..., image_height, image_width, 1]
//     // outputs
//     at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
//     at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
//     at::Tensor v_ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
//     at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
//     at::Tensor v_opacities,                 // [..., N] or [nnz]
//     at::Tensor v_normals,                   // [..., N, 3] or [nnz, 3]
//     at::Tensor v_densify                    // [..., N, 2] or [nnz, 2]
// ) {
//     bool packed = means2d.dim() == 2;

//     uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
//     uint32_t I = render_alphas.numel() / (image_height * image_width); // number of images
//     uint32_t tile_height = tile_offsets.size(-2);
//     uint32_t tile_width = tile_offsets.size(-1);
//     uint32_t n_isects = flatten_ids.size(0);

//     // Each block covers a tile on the image. In total there are
//     // I * tile_height * tile_width blocks.
//     dim3 threads = {tile_size, tile_size, 1};
//     dim3 grid = {I, tile_height, tile_width};

//     int64_t shmem_size =
//         tile_size * tile_size *
//         (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) + sizeof(vec3) +
//          sizeof(vec3) + sizeof(float) * CDIM + sizeof(float) * 3);

//     if (n_isects == 0) {
//         // skip the kernel launch if there are no elements
//         return;
//     }

//     // TODO: an optimization can be done by passing the actual number of
//     // channels into the kernel functions and avoid necessary global memory
//     // writes. This requires moving the channel padding from python to C side.
//     if (cudaFuncSetAttribute(
//             rasterize_to_pixels_2dgs_bwd_kernel<CDIM, float>,
//             cudaFuncAttributeMaxDynamicSharedMemorySize,
//             shmem_size
//         ) != cudaSuccess) {
//         AT_ERROR(
//             "Failed to set maximum shared memory size (requested ",
//             shmem_size,
//             " bytes), try lowering tile_size."
//         );
//     }

//     rasterize_to_pixels_2dgs_bwd_kernel<CDIM, float>
//         <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
//             I,
//             N,
//             n_isects,
//             packed,
//             reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
//             ray_transforms.data_ptr<float>(),
//             colors.data_ptr<float>(),
//             normals.data_ptr<float>(),
//             opacities.data_ptr<float>(),
//             backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
//                                     : nullptr,
//             masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
//             image_width,
//             image_height,
//             tile_size,
//             tile_width,
//             tile_height,
//             tile_offsets.data_ptr<int32_t>(),
//             flatten_ids.data_ptr<int32_t>(),
//             render_colors.data_ptr<float>(),
//             render_alphas.data_ptr<float>(),
//             last_ids.data_ptr<int32_t>(),
//             median_ids.data_ptr<int32_t>(),
//             v_render_colors.data_ptr<float>(),
//             v_render_alphas.data_ptr<float>(),
//             v_render_normals.data_ptr<float>(),
//             v_render_distort.data_ptr<float>(),
//             v_render_median.data_ptr<float>(),
//             v_means2d_abs.has_value()
//                 ? reinterpret_cast<vec2 *>(
//                       v_means2d_abs.value().data_ptr<float>()
//                   )
//                 : nullptr,
//             reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
//             v_ray_transforms.data_ptr<float>(),
//             v_colors.data_ptr<float>(),
//             v_opacities.data_ptr<float>(),
//             v_normals.data_ptr<float>(),
//             v_densify.data_ptr<float>()
//         );




//     // for test
//     shmem_size =
//         tile_size * tile_size / 8 *   // batch_factor = 8
//         (sizeof(int32_t) + sizeof(float) + sizeof(float4) + sizeof(float) +
//          sizeof(float) + sizeof(float) + sizeof(float) * CDIM +
//          sizeof(float) * CDIM + sizeof(float) * CDIM + sizeof(float3) +
//          sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) + sizeof(float2));
         
//     if (cudaFuncSetAttribute(
//         per_gaussian_rasterize_to_pixels_2dgs_bwd_kernel<CDIM, float>,
//             cudaFuncAttributeMaxDynamicSharedMemorySize,
//             shmem_size
//         ) != cudaSuccess) {
//         AT_ERROR(
//             "Failed to set maximum shared memory size (requested ",
//             shmem_size,
//             " bytes), try lowering tile_size."
//         );
//     }

//     at::Tensor copy_v_means2d_abs = at::zeros_like(v_means2d);
//     at::Tensor copy_v_means2d = at::zeros_like(v_means2d);
//     at::Tensor copy_v_ray_transforms = at::zeros_like(v_ray_transforms);
//     at::Tensor copy_v_colors = at::zeros_like(v_colors);
//     at::Tensor copy_v_opacities = at::zeros_like(v_opacities);
//     at::Tensor copy_v_normals = at::zeros_like(v_normals);
//     at::Tensor copy_v_densify = at::zeros_like(v_densify);
    
// 	const int THREADS = 32;
//     per_gaussian_rasterize_to_pixels_2dgs_bwd_kernel<CDIM, float>
//         <<<((num_buckets * 32) + THREADS - 1) / THREADS, THREADS, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
//             I,
//             N,
//             n_isects,
//             packed,
//             reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
//             ray_transforms.data_ptr<float>(),
//             colors.data_ptr<float>(),
//             normals.data_ptr<float>(),
//             opacities.data_ptr<float>(),
//             backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
//                                     : nullptr,
//             masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
//             image_width,
//             image_height,
//             tile_size,
//             tile_width,
//             tile_height,
//             tile_offsets.data_ptr<int32_t>(),
//             flatten_ids.data_ptr<int32_t>(),
//             render_colors.data_ptr<float>(),
//             render_alphas.data_ptr<float>(),
//             render_normals.data_ptr<float>(),
//             render_stats.data_ptr<float>(),
//             last_ids.data_ptr<int32_t>(),
//             median_ids.data_ptr<int32_t>(),
//             num_buckets,
//             per_tile_bucket_offset.data_ptr<int32_t>(),
//             bucket_to_tile.data_ptr<int32_t>(),
//             sampled_stats.data_ptr<float>(),
//             sampled_ar.data_ptr<float>(),
//             sampled_an.data_ptr<float>(),
//             max_contrib.data_ptr<int32_t>(),
//             v_render_colors.data_ptr<float>(),
//             v_render_alphas.data_ptr<float>(),
//             v_render_normals.data_ptr<float>(),
//             v_render_distort.data_ptr<float>(),
//             v_render_median.data_ptr<float>(),
//             v_means2d_abs.has_value()
//                 ? reinterpret_cast<vec2 *>(
//                       copy_v_means2d_abs.data_ptr<float>()
//                   )
//                 : nullptr,
//             reinterpret_cast<vec2 *>(copy_v_means2d.data_ptr<float>()),
//             copy_v_ray_transforms.data_ptr<float>(),
//             copy_v_colors.data_ptr<float>(),
//             copy_v_opacities.data_ptr<float>(),
//             copy_v_normals.data_ptr<float>(),
//             copy_v_densify.data_ptr<float>()
//         );

//     // const float TOLERANCE = 0.00001f;
//     // bool b0 = v_means2d_abs.has_value() ? all_close_custom(copy_v_means2d_abs, v_means2d_abs.value(), TOLERANCE): 1;
//     // bool b1 = all_close_custom(copy_v_means2d, v_means2d, TOLERANCE);
//     // bool b2 = all_close_custom(copy_v_ray_transforms, v_ray_transforms, TOLERANCE);
//     // bool b3 = all_close_custom(copy_v_colors, v_colors, TOLERANCE);
//     // bool b4 = all_close_custom(copy_v_opacities, v_opacities, TOLERANCE);
//     // bool b5 = all_close_custom(copy_v_normals, v_normals, TOLERANCE);
//     // bool b6 = all_close_custom(copy_v_densify, v_densify, TOLERANCE);
//     // if (!b0 || !b1 || !b2 || !b3 || !b4 || !b5 || !b6) {
//     //     printf("v_means2d_abs %d, v_means2d %d, v_ray_transforms %d, v_colors %d, v_opacities %d, v_normals %d, v_densify %d \n",
//     //         b0, b1, b2, b3, b4, b5, b6);
//     //     printf("detect diff!");
//     //     if (!b0) tensor_diff_check(copy_v_means2d_abs, v_means2d_abs.value(), TOLERANCE);
//     //     if (!b1) tensor_diff_check(copy_v_means2d, v_means2d, TOLERANCE);
//     //     if (!b2) tensor_diff_check(copy_v_ray_transforms, v_ray_transforms, TOLERANCE);
//     //     if (!b3) tensor_diff_check(copy_v_colors, v_colors, TOLERANCE);
//     //     if (!b4) tensor_diff_check(copy_v_opacities, v_opacities, TOLERANCE);
//     //     if (!b5) tensor_diff_check(copy_v_normals, v_normals, TOLERANCE);
//     //     if (!b6) tensor_diff_check(copy_v_densify, v_densify, TOLERANCE);
//     // }
// }

// per-gaussian backward
template <uint32_t CDIM>
void launch_rasterize_to_pixels_2dgs_bwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,                   // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,                    // [..., N, 3] or [nnz, 3]
    const at::Tensor opacities,                 // [..., N] or [nnz]
    const at::Tensor normals,                   // [..., N, 3] or [nnz, 3]
    const at::Tensor densify,                   // [..., N, 2] or [nnz, 2]
    const at::optional<at::Tensor> backgrounds, // [..., CDIM]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // ray_crossions
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // forward outputs
    const at::Tensor render_colors, // [..., image_height, image_width, CDIM]
    const at::Tensor render_alphas, // [..., image_height, image_width, 1]
    const at::Tensor render_normals, // [..., image_height, image_width, 3]
    const at::Tensor render_stats, // [..., image_height, image_width, 2]
    const at::Tensor last_ids,      // [..., image_height, image_width]
    const at::Tensor median_ids,    // [..., image_height, image_width]
    // efficient backward
    const uint32_t num_buckets,
    const at::Tensor per_tile_bucket_offset,
    const at::Tensor bucket_to_tile,
    const at::Tensor sampled_stats,
    const at::Tensor sampled_ar,
    const at::Tensor sampled_an,
    const at::Tensor max_contrib,
    // gradients of outputs
    const at::Tensor v_render_colors,  // [..., image_height, image_width, 3]
    const at::Tensor v_render_alphas,  // [..., image_height, image_width, 1]
    const at::Tensor v_render_normals, // [..., image_height, image_width, 3]
    const at::Tensor v_render_distort, // [..., image_height, image_width, 1]
    const at::Tensor v_render_median,  // [..., image_height, image_width, 1]
    // outputs
    at::optional<at::Tensor> v_means2d_abs, // [..., N, 2] or [nnz, 2]
    at::Tensor v_means2d,                   // [..., N, 2] or [nnz, 2]
    at::Tensor v_ray_transforms,            // [..., N, 3, 3] or [nnz, 3, 3]
    at::Tensor v_colors,                    // [..., N, 3] or [nnz, 3]
    at::Tensor v_opacities,                 // [..., N] or [nnz]
    at::Tensor v_normals,                   // [..., N, 3] or [nnz, 3]
    at::Tensor v_densify                    // [..., N, 2] or [nnz, 2]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = render_alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size =
        tile_size * tile_size / 8 *   // batch_factor = 8
        (sizeof(int32_t) + sizeof(float) + sizeof(float4) + sizeof(float) +
         sizeof(float) + sizeof(float) + sizeof(float) * CDIM +
         sizeof(float) * CDIM + sizeof(float) * CDIM + sizeof(float3) +
         sizeof(float) * 3 + sizeof(float) * 3 + sizeof(float) + sizeof(float2));

    if (n_isects == 0) {
        // skip the kernel launch if there are no elements
        return;
    }

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
        per_gaussian_rasterize_to_pixels_2dgs_bwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }
    
	const int THREADS = 32;
    per_gaussian_rasterize_to_pixels_2dgs_bwd_kernel<CDIM, float>
        <<<((num_buckets * 32) + THREADS - 1) / THREADS, THREADS, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            ray_transforms.data_ptr<float>(),
            colors.data_ptr<float>(),
            normals.data_ptr<float>(),
            opacities.data_ptr<float>(),
            backgrounds.has_value() ? backgrounds.value().data_ptr<float>()
                                    : nullptr,
            masks.has_value() ? masks.value().data_ptr<bool>() : nullptr,
            image_width,
            image_height,
            tile_size,
            tile_width,
            tile_height,
            tile_offsets.data_ptr<int32_t>(),
            flatten_ids.data_ptr<int32_t>(),
            render_colors.data_ptr<float>(),
            render_alphas.data_ptr<float>(),
            render_normals.data_ptr<float>(),
            render_stats.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>(),
            num_buckets,
            per_tile_bucket_offset.data_ptr<int32_t>(),
            bucket_to_tile.data_ptr<int32_t>(),
            sampled_stats.data_ptr<float>(),
            sampled_ar.data_ptr<float>(),
            sampled_an.data_ptr<float>(),
            max_contrib.data_ptr<int32_t>(),
            v_render_colors.data_ptr<float>(),
            v_render_alphas.data_ptr<float>(),
            v_render_normals.data_ptr<float>(),
            v_render_distort.data_ptr<float>(),
            v_render_median.data_ptr<float>(),
            v_means2d_abs.has_value()
                ? reinterpret_cast<vec2 *>(
                      v_means2d_abs.value().data_ptr<float>()
                  )
                : nullptr,
            reinterpret_cast<vec2 *>(v_means2d.data_ptr<float>()),
            v_ray_transforms.data_ptr<float>(),
            v_colors.data_ptr<float>(),
            v_opacities.data_ptr<float>(),
            v_normals.data_ptr<float>(),
            v_densify.data_ptr<float>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_2dgs_bwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor ray_transforms,                                       \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::Tensor normals,                                              \
        const at::Tensor densify,                                              \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        const uint32_t image_width,                                            \
        const uint32_t image_height,                                           \
        const uint32_t tile_size,                                              \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor render_colors,                                        \
        const at::Tensor render_alphas,                                        \
        const at::Tensor render_normals,                                       \
        const at::Tensor render_stats,                                         \
        const at::Tensor last_ids,                                             \
        const at::Tensor median_ids,                                           \
        const uint32_t num_buckets,                                            \
        const at::Tensor per_tile_bucket_offset,                               \
        const at::Tensor bucket_to_tile,                                       \
        const at::Tensor sampled_stats,                                        \
        const at::Tensor sampled_ar,                                           \
        const at::Tensor sampled_an,                                           \
        const at::Tensor max_contrib,                                          \
        const at::Tensor v_render_colors,                                      \
        const at::Tensor v_render_alphas,                                      \
        const at::Tensor v_render_normals,                                     \
        const at::Tensor v_render_distort,                                     \
        const at::Tensor v_render_median,                                      \
        at::optional<at::Tensor> v_means2d_abs,                                \
        const at::Tensor v_means2d,                                            \
        const at::Tensor v_ray_transforms,                                     \
        const at::Tensor v_colors,                                             \
        const at::Tensor v_opacities,                                          \
        const at::Tensor v_normals,                                            \
        const at::Tensor v_densify                                             \
    );

__INS__(1)
__INS__(2)
__INS__(3)
__INS__(4)
__INS__(5)
__INS__(8)
__INS__(9)
__INS__(16)
__INS__(17)
__INS__(32)
__INS__(33)
__INS__(64)
__INS__(65)
__INS__(128)
__INS__(129)
__INS__(256)
__INS__(257)
__INS__(512)
__INS__(513)
#undef __INS__

} // namespace gsplat
