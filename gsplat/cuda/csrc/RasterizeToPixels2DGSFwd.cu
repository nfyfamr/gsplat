#include <ATen/Dispatch.h>
#include <ATen/core/Tensor.h>
#include <c10/cuda/CUDAStream.h>
#include <cooperative_groups.h>

#include <cub/cub.cuh>

#include "Common.h"
#include "Rasterization.h"

namespace gsplat {

namespace cg = cooperative_groups;

////////////////////////////////////////////////////////////////
// Forward
////////////////////////////////////////////////////////////////

template <uint32_t CDIM, typename scalar_t>
__global__ void rasterize_to_pixels_2dgs_fwd_kernel(
    const uint32_t I,        // number of images
    const uint32_t N,        // number of gaussians
    const uint32_t n_isects, // number of ray-primitive intersections.
    const bool packed,       // whether the input tensors are packed
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
    const scalar_t *__restrict__ colors,    // [..., N, CDIM] or [nnz, CDIM]  //
                                            // Gaussian colors or ND features.
    const scalar_t *__restrict__ opacities, // [..., N] or [nnz] // Gaussian
                                            // opacities that support per-view
                                            // values.
    const scalar_t *__restrict__ normals, // [..., N, 3] or [nnz, 3] // The
                                          // normals in camera space.
    const scalar_t *__restrict__ backgrounds, // [..., CDIM] // Background colors
                                              // on camera basis
    const bool *__restrict__ masks, // [..., tile_height, tile_width] // Optional
                                    // tile mask to skip rendering GS to masked
                                    // tiles.
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    const uint32_t tile_width,
    const uint32_t tile_height,
    const int32_t
        *__restrict__ tile_offsets, // [..., tile_height, tile_width]    //
                                    // Intersection offsets outputs from
                                    // `isect_offset_encode()`, this is the
                                    // result of a prefix sum, and gives the
                                    // interval that our gaussians are gonna
                                    // use.
    const int32_t *__restrict__ flatten_ids, // [n_isects] // The global flatten
                                             // indices in [I * N] or [nnz] from
                                             // `isect_tiles()`.

    // efficient backward
    const int32_t *__restrict__ per_tile_bucket_offset,
    int32_t *__restrict__ bucket_to_tile,
    scalar_t *__restrict__ sampled_stats,   // (sampled_T, sampled_avd, sampled_aw, sampled_avwd)
    scalar_t *__restrict__ sampled_ar,
    scalar_t *__restrict__ sampled_an,
    int32_t *__restrict__ max_contrib,

    // outputs
    scalar_t
        *__restrict__ render_colors, // [..., image_height, image_width, CDIM]
    scalar_t *__restrict__ render_alphas,  // [..., image_height, image_width, 1]
    scalar_t *__restrict__ render_normals, // [..., image_height, image_width, 3]
    scalar_t *__restrict__ render_stats, // [..., image_height, image_width, 2]     // (vis_wd, w)
    scalar_t *__restrict__ render_distort, // [..., image_height, image_width, 1]
                                           // // Stores the per-pixel distortion
                                           // error proposed in Mip-NeRF 360.
    scalar_t
        *__restrict__ render_median, // [..., image_height, image_width, 1]  //
                                     // Stores the median depth contribution for
                                     // each pixel "set to the depth of the
                                     // Gaussian that brings the accumulated
                                     // opacity over 0.5."
    int32_t *__restrict__ last_ids,  // [..., image_height, image_width]     //
                                     // Stores the index of the last Gaussian
                                     // that contributed to each pixel.
    int32_t *__restrict__ median_ids // [..., image_height, image_width]    //
                                     // Stores the index of the Gaussian that
                                     // contributes to the median depth for each
                                     // pixel (bring over 0.5).
) {
    // each thread draws one pixel, but also timeshares caching gaussians in a
    // shared tile

    /**
     * ==============================
     * Thread and block setup:
     * This sets up the thread and block indices, determining which image,
     * tile, and pixel each thread will process. The grid structure is assigend
     * as: I * tile_height * tile_width blocks (3d grid), each block is a tile.
     * Each thread is responsible for one pixel. (blockSize = tile_size *
     * tile_size)
     * ==============================
     */
    auto block = cg::this_thread_block();
    int32_t image_id = block.group_index().x;
    int32_t tile_id =
        block.group_index().y * tile_width + block.group_index().z;
    uint32_t i = block.group_index().y * tile_size + block.thread_index().y;
    uint32_t j = block.group_index().z * tile_size + block.thread_index().x;

    tile_offsets +=
        image_id * tile_height *
        tile_width; // get the global offset of the tile w.r.t the image
    render_colors +=
        image_id * image_height * image_width *
        CDIM; // get the global offset of the pixel w.r.t the image
    render_alphas +=
        image_id * image_height *
        image_width; // get the global offset of the pixel w.r.t the image
    last_ids +=
        image_id * image_height *
        image_width; // get the global offset of the pixel w.r.t the image
    render_normals += image_id * image_height * image_width * 3;
    render_distort += image_id * image_height * image_width;
    render_median += image_id * image_height * image_width;
    median_ids += image_id * image_height * image_width;

    // get the global offset of the background and mask
    if (backgrounds != nullptr) {
        backgrounds += image_id * CDIM;
    }
    if (masks != nullptr) {
        masks += image_id * tile_height * tile_width;
    }

    // find the center of the pixel
    float px = (float)j + 0.5f;
    float py = (float)i + 0.5f;
    int32_t pix_id = i * image_width + j;

    // return if out of bounds
    // keep not rasterizing threads around for reading data
    bool inside = (i < image_height && j < image_width);
    bool done = !inside;

    // when the mask is provided, render the background color and return
    // if this tile is labeled as False
    if (masks != nullptr && inside && !masks[tile_id]) {
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? 0.0f : backgrounds[k];
        }
        return;
    }

    // have all threads in tile process the same gaussians in batches
    // first collect gaussians between range.x and range.y in batches
    // which gaussians to look through in this tile

    // print
    int32_t range_start = tile_offsets[tile_id];
    int32_t range_end =
        // see if this is the last tile in the image
        (image_id == I - 1) && (tile_id == tile_width * tile_height - 1)
            ? n_isects
            : tile_offsets[tile_id + 1];
    const uint32_t block_size = block.size();
    uint32_t num_batches =
        (range_end - range_start + block_size - 1) / block_size;

    /**
     * ==============================
     * Register computing variables:
     * For each pixel, we need to find its uv intersection with the gaussian
     * primitives. then we retrieve the kernel's parameters and kernel weights
     * do the splatting rendering equation.
     * ==============================
     */
    // Shared memory layout:
    // This memory is laid out as follows:
    // | gaussian indices | x : y : alpha | u | v | w |
    extern __shared__ int s[];
    int32_t *id_batch = (int32_t *)s; // [block_size]

    // stores the concatination for projected primitive source (x, y) and
    // opacity alpha
    vec3 *xy_opacity_batch =
        reinterpret_cast<vec3 *>(&id_batch[block_size]); // [block_size]

    // these are row vectors of the ray transformation matrices for the current
    // batch of gaussians
    vec3 *u_Ms_batch =
        reinterpret_cast<vec3 *>(&xy_opacity_batch[block_size]); // [block_size]
    vec3 *v_Ms_batch =
        reinterpret_cast<vec3 *>(&u_Ms_batch[block_size]); // [block_size]
    vec3 *w_Ms_batch =
        reinterpret_cast<vec3 *>(&v_Ms_batch[block_size]); // [block_size]

    // current visibility left to render
    // transmittance is gonna be used in the backward pass which requires a high
    // numerical precision so we use double for it. However double make bwd 1.5x
    // slower so we stick with float for now.
    // The coefficient for volumetric rendering for our responsible pixel.
    float T = 1.0f;
    // index of most recent gaussian to write to this thread's pixel
    uint32_t cur_idx = 0;

    // collect and process batches of gaussians
    // each thread loads one gaussian at a time before rasterizing its
    // designated pixel
    uint32_t tr = block.thread_rank();

    // Per-pixel distortion error proposed in Mip-NeRF 360.
    // Implemented reference:
    // https://github.com/nerfstudio-project/nerfacc/blob/master/nerfacc/losses.py#L7
    float distort = 0.f;
    float accum_vis_depth = 0.f; // accumulate vis * depth
    float accum_w = 0.f;         // accumulate vis
    float accum_vis_wd = 0.f;    // accumulate vis * (depth * accum_w - accum_vis_depth)

    // keep track of median depth contribution
    float median_depth = 0.f;
    uint32_t median_idx = 0.f;

    /**
     * ==============================
     * Per-pixel rendering: (2DGS Differntiable Rasterizer Forward Pass)
     * This section is responsible for rendering a single pixel.
     * It processes batches of gaussians and accumulates the pixel color and
     * normal.
     * ==============================
     */

    // TODO (WZ): merge pix_out and normal_out to
    //  float pix_out[CDIM + 3] = {0.f}
    float pix_out[CDIM] = {0.f};
    float normal_out[3] = {0.f};
    
    // efficient backward
    const uint32_t GLOBAL_TILE_ID = image_id * tile_height * tile_width + tile_id;
    const uint32_t NUM_GAUSSIANS_IN_TILE = range_end - range_start;
    const uint32_t NUM_BUCKETS_IN_TILE = (NUM_GAUSSIANS_IN_TILE + 31) / 32;
    uint32_t bbm_current = (GLOBAL_TILE_ID == 0) ? 0 : per_tile_bucket_offset[GLOBAL_TILE_ID - 1];
    uint32_t bucket_idx_counter = 0;

    // bucket-to-tile mapping
    for (int i_map = 0; i_map < (NUM_BUCKETS_IN_TILE + block_size - 1) / block_size; ++i_map) {
        int bucket_idx = i_map * block_size + tr;
        if (bucket_idx < NUM_BUCKETS_IN_TILE) {
            bucket_to_tile[bbm_current + bucket_idx] = GLOBAL_TILE_ID;
        }
    }
    block.sync();

    for (uint32_t b = 0; b < num_batches; ++b) {
        // resync all threads before beginning next batch
        // end early if entire tile is done
        if (__syncthreads_count(done) >= block_size) {
            break;
        }

        // each thread fetch 1 gaussian from front to back
        // index of gaussian to load
        uint32_t batch_start = range_start + block_size * b;
        uint32_t idx = batch_start + tr;

        // only threads within the range of the tile will fetch gaussians
        /**
         * Launch this block with each thread responsible for one gaussian.
         */
        if (idx < range_end) {
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
        }

        // wait for other threads to collect the gaussians in batch
        block.sync();

        /**
         * ==================================================
         * Forward rasterization pass:
         * ==================================================
         *
         * GSplat computes rasterization point of intersection as:
         * 1. Generate 2 homogeneous plane parameter vectors as sets of points
         * in UV space
         * 2. Find the set of points that satisfy both conditions with the cross
         * product
         * 3. Find where this solution set intersects with UV plane using
         * projective flattening
         *
         * For each gaussian G_i and pixel q_xy:
         *
         * 1. Compute homogeneous plane parameters:
         *    h_u = p_x * M_w - M_u
         *    h_v = p_y * M_w - M_v
         *    where M_u, M_v, M_w are rows of the KWH transform
         *
         * Note: this works because:
         *    for any vector q_uv [u, v, 1], applying co-vector h_u will yield
         * the following expression: h_u * [u, v, 1]^T = P_x * (M_w * q_uv) -
         * M_u * q_uv = P_x * q_ray.z - q_ray.x * q_ray.z
         *    - where P_x is the x-coordinate of the ray origin
         *    Thus: h_u  defines a set of q_uv where q_uv's projected x
         * coordinate in ray space is P_x which aligns with the homogeneous
         * plane definition in original 2DGS paper (similar for h_v)
         *
         * 2. Compute intersection:
         *    zeta = h_u × h_v
         *    This cross product is the only solution that satisfies both
         * homogeneous plane equations (dot product == 0)
         *
         * 3. Project to UV space:
         *    s_uv = [zeta_1/zeta_3, zeta_2/zeta_3]
         *    - since UV space is essentially another ray space, and arbitrary
         * scale of q_uv will not change the result of dot product over
         * orthogonality
         *    - thus, the result is the point of intersection in UV space
         *
         * 4. Evaluate gaussian kernel:
         *    G_i = exp(-(s_u^2 + s_v^2)/2)
         *
         * 5. Accumulate color:
         *    p_xy += alpha_i * c_i * G_i * prod(1 - alpha_j * G_j)
         *
         * This method efficiently computes the point of intersection and
         * evaluates the gaussian kernel in UV space.
         * Note: in some cases, we use the minimum of ray-intersection kernels
         * and 2D projected gaussian kernels
         */
        // process gaussians in the current batch for this pixel
        uint32_t batch_size = min(block_size, range_end - batch_start);
        for (uint32_t t = 0; (t < batch_size) && !done; ++t) {
            // add incoming T and accumulated radiance (ar) value for every 32nd gaussian
            // uint32_t global_gaussian_idx_in_tile = b * block_size + t;
            if (t % 32 == 0) {
                uint32_t sample_bucket_idx = bbm_current + bucket_idx_counter;
                float4* sampled_stats_ptr = (float4*)sampled_stats;
                sampled_stats_ptr[(sample_bucket_idx * block_size) + tr] = make_float4(T, accum_vis_depth, accum_w, accum_vis_wd);
                for (uint32_t ch = 0; ch < CDIM; ++ch) {
                    uint32_t ar_idx = (sample_bucket_idx * block_size * CDIM) + (ch * block_size) + tr;
                    sampled_ar[ar_idx] = pix_out[ch];
                }
                float4* sampled_an_ptr = (float4*) sampled_an;
                sampled_an_ptr[(sample_bucket_idx * block_size) + tr] = make_float4(normal_out[0], normal_out[1], normal_out[2], 0);
                bucket_idx_counter++;
            }

            const vec3 xy_opac = xy_opacity_batch[t];
            const float opac = xy_opac.z;

            const vec3 u_M = u_Ms_batch[t];
            const vec3 v_M = v_Ms_batch[t];
            const vec3 w_M = w_Ms_batch[t];

            // h_u and h_v are the homogeneous plane representations (they are
            // contravariant to the points on the primitive plane)
            const vec3 h_u = px * w_M - u_M;
            const vec3 h_v = py * w_M - v_M;

            const vec3 ray_cross = glm::cross(h_u, h_v);
            if (ray_cross.z == 0.0)
                continue;

            const vec2 s =
                vec2(ray_cross.x / ray_cross.z, ray_cross.y / ray_cross.z);

            // IMPORTANT: This is where the gaussian kernel is evaluated!!!!!

            // point of interseciton in uv space
            const float gauss_weight_3d = s.x * s.x + s.y * s.y;

            // projected gaussian kernel
            const vec2 d = {xy_opac.x - px, xy_opac.y - py};
            // #define FILTER_INV_SQUARE_2DGS 2.0f
            const float gauss_weight_2d =
                FILTER_INV_SQUARE_2DGS * (d.x * d.x + d.y * d.y);

            // merge ray-intersection kernel and 2d gaussian kernel
            const float gauss_weight = min(gauss_weight_3d, gauss_weight_2d);

            const float sigma = 0.5f * gauss_weight;
            // evaluation of the gaussian exponential term
            float alpha = min(0.999f, opac * __expf(-sigma));

            // ignore transparent gaussians
            if (sigma < 0.f || alpha < ALPHA_THRESHOLD) {
                continue;
            }

            const float next_T = T * (1.0f - alpha);
            if (next_T <= 1e-4) { // this pixel is done: exclusive
                done = true;
                break;
            }

            // run volumetric rendering..
            int32_t g = id_batch[t];
            const float vis = alpha * T;
            const float *c_ptr = colors + g * CDIM;
#pragma unroll
            for (uint32_t k = 0; k < CDIM; ++k) {
                pix_out[k] += c_ptr[k] * vis;
            }

            const float *n_ptr = normals + g * 3;
#pragma unroll
            for (uint32_t k = 0; k < 3; ++k) {
                normal_out[k] += n_ptr[k] * vis;
            }

            if (render_distort != nullptr) {
                // the last channel of colors is depth
                const float depth = c_ptr[CDIM - 1];
                // in nerfacc, loss_bi_0 = weights * t_mids *
                // exclusive_sum(weights)
                const float distort_bi_0 = vis * depth * (1.0f - T);
                // in nerfacc, loss_bi_1 = weights * exclusive_sum(weights *
                // t_mids)
                const float distort_bi_1 = vis * accum_vis_depth;
                distort += 2.0f * (distort_bi_0 - distort_bi_1);
                accum_vis_depth += vis * depth;
                accum_w += vis;
                accum_vis_wd += vis * (depth * accum_w - accum_vis_depth);
            }

            // compute median depth
            if (T > 0.5) {
                median_depth = c_ptr[CDIM - 1];
                median_idx = batch_start + t;
            }

            cur_idx = batch_start + t;

            T = next_T;
        }
    }
    if (inside) {
        // Here T is the transmittance AFTER the last gaussian in this pixel.
        // We (should) store double precision as T would be used in backward
        // pass and it can be very small and causing large diff in gradients
        // with float32. However, double precision makes the backward pass 1.5x
        // slower so we stick with float for now.
        render_alphas[pix_id] = 1.0f - T;
#pragma unroll
        for (uint32_t k = 0; k < CDIM; ++k) {
            render_colors[pix_id * CDIM + k] =
                backgrounds == nullptr ? pix_out[k]
                                       : (pix_out[k] + T * backgrounds[k]);
        }
#pragma unroll
        for (uint32_t k = 0; k < 3; ++k) {
            render_normals[pix_id * 3 + k] = normal_out[k];
        }
        // index in bin of last gaussian in this pixel
        last_ids[pix_id] = static_cast<int32_t>(cur_idx);

        if (render_distort != nullptr) {
            render_distort[pix_id] = distort;
            float2* render_stats_ptr = (float2*)render_stats;
            render_stats_ptr[pix_id] = make_float2(accum_vis_wd, accum_w);
        }

        render_median[pix_id] = median_depth;
        // index in bin of gaussian that contributes to median depth
        median_ids[pix_id] = static_cast<int32_t>(median_idx);
    }

    // max reduce the last contributor
    typedef cub::BlockReduce<uint32_t, 256> BlockReduce;  // hard-typed block_size 
    __shared__ typename BlockReduce::TempStorage temp_storage;
    cur_idx = BlockReduce(temp_storage).Reduce(cur_idx, cub::Max());
    if (tr == 0) {
        max_contrib[GLOBAL_TILE_ID] = cur_idx;
    }
}

template <uint32_t CDIM>
void launch_rasterize_to_pixels_2dgs_fwd_kernel(
    // Gaussian parameters
    const at::Tensor means2d,        // [..., N, 2] or [nnz, 2]
    const at::Tensor ray_transforms, // [..., N, 3, 3] or [nnz, 3, 3]
    const at::Tensor colors,         // [..., N, channels] or [nnz, channels]
    const at::Tensor opacities,      // [..., N]  or [nnz]
    const at::Tensor normals,        // [..., N, 3] or [nnz, 3]
    const at::optional<at::Tensor> backgrounds, // [..., channels]
    const at::optional<at::Tensor> masks,       // [..., tile_height, tile_width]
    // image size
    const uint32_t image_width,
    const uint32_t image_height,
    const uint32_t tile_size,
    // intersections
    const at::Tensor tile_offsets, // [..., tile_height, tile_width]
    const at::Tensor flatten_ids,  // [n_isects]
    // efficient backward
    const at::Tensor per_tile_bucket_offset,
    at::Tensor bucket_to_tile,
    at::Tensor sampled_stats,
    at::Tensor sampled_ar,
    at::Tensor sampled_an,
    at::Tensor max_contrib,
    // outputs
    at::Tensor renders,        // [..., image_height, image_width, channels]
    at::Tensor alphas,         // [..., image_height, image_width]
    at::Tensor render_normals, // [..., image_height, image_width, 3]
    at::Tensor render_stats,  // [..., image_height, image_width, 2]
    at::Tensor render_distort, // [..., image_height, image_width]
    at::Tensor render_median,  // [..., image_height, image_width]
    at::Tensor last_ids,       // [..., image_height, image_width]
    at::Tensor median_ids      // [..., image_height, image_width]
) {
    bool packed = means2d.dim() == 2;

    uint32_t N = packed ? 0 : means2d.size(-2); // number of gaussians
    uint32_t I = alphas.numel() / (image_height * image_width); // number of images
    uint32_t tile_height = tile_offsets.size(-2);
    uint32_t tile_width = tile_offsets.size(-1);
    uint32_t n_isects = flatten_ids.size(0);

    // Each block covers a tile on the image. In total there are
    // I * tile_height * tile_width blocks.
    dim3 threads = {tile_size, tile_size, 1};
    dim3 grid = {I, tile_height, tile_width};

    int64_t shmem_size = tile_size * tile_size *
                         (sizeof(int32_t) + sizeof(vec3) + sizeof(vec3) +
                          sizeof(vec3) + sizeof(vec3));

    // TODO: an optimization can be done by passing the actual number of
    // channels into the kernel functions and avoid necessary global memory
    // writes. This requires moving the channel padding from python to C side.
    if (cudaFuncSetAttribute(
            rasterize_to_pixels_2dgs_fwd_kernel<CDIM, float>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shmem_size
        ) != cudaSuccess) {
        AT_ERROR(
            "Failed to set maximum shared memory size (requested ",
            shmem_size,
            " bytes), try lowering tile_size."
        );
    }

    rasterize_to_pixels_2dgs_fwd_kernel<CDIM, float>
        <<<grid, threads, shmem_size, at::cuda::getCurrentCUDAStream()>>>(
            I,
            N,
            n_isects,
            packed,
            reinterpret_cast<vec2 *>(means2d.data_ptr<float>()),
            ray_transforms.data_ptr<float>(),
            colors.data_ptr<float>(),
            opacities.data_ptr<float>(),
            normals.data_ptr<float>(),
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
            per_tile_bucket_offset.data_ptr<int32_t>(),
            bucket_to_tile.data_ptr<int32_t>(),
            sampled_stats.data_ptr<float>(),
            sampled_ar.data_ptr<float>(),
            sampled_an.data_ptr<float>(),
            max_contrib.data_ptr<int32_t>(),
            renders.data_ptr<float>(),
            alphas.data_ptr<float>(),
            render_normals.data_ptr<float>(),
            render_stats.data_ptr<float>(),
            render_distort.data_ptr<float>(),
            render_median.data_ptr<float>(),
            last_ids.data_ptr<int32_t>(),
            median_ids.data_ptr<int32_t>()
        );
}

// Explicit Instantiation: this should match how it is being called in .cpp
// file.
// TODO: this is slow to compile, can we do something about it?
#define __INS__(CDIM)                                                          \
    template void launch_rasterize_to_pixels_2dgs_fwd_kernel<CDIM>(            \
        const at::Tensor means2d,                                              \
        const at::Tensor ray_transforms,                                       \
        const at::Tensor colors,                                               \
        const at::Tensor opacities,                                            \
        const at::Tensor normals,                                              \
        const at::optional<at::Tensor> backgrounds,                            \
        const at::optional<at::Tensor> masks,                                  \
        uint32_t image_width,                                                  \
        uint32_t image_height,                                                 \
        uint32_t tile_size,                                                    \
        const at::Tensor tile_offsets,                                         \
        const at::Tensor flatten_ids,                                          \
        const at::Tensor per_tile_bucket_offset,                               \
        at::Tensor bucket_to_tile,                                             \
        at::Tensor sampled_stats,                                              \
        at::Tensor sampled_ar,                                                 \
        at::Tensor sampled_an,                                                 \
        at::Tensor max_contrib,                                                \
        at::Tensor renders,                                                    \
        at::Tensor alphas,                                                     \
        at::Tensor render_normals,                                             \
        at::Tensor render_stats,                                               \
        at::Tensor render_distort,                                             \
        at::Tensor render_median,                                              \
        at::Tensor last_ids,                                                   \
        at::Tensor median_ids                                                  \
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
