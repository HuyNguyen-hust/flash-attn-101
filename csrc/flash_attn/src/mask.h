#pragma once

#include <cuda_runtime.h>
#include <cute/tensor.hpp>

namespace flash
{

using namespace cute;

struct Mask
{  
    template <typename Engine, typename Layout>
    __forceinline__ __device__ void apply_mask(
        Tensor<Engine, Layout> &tensor_,
        const int col_idx_offset_,
        const int row_idx_offset_,
        const int warp_row_stride,
        const int warp_col_stride
    )
    {   
        // what is warp_row_stride, warp_col_stride?
        // Think of the tiled_mma as it is repeated MMA_M times along M-dimension, and MMA_N times along N-dimension
        // So one thread needs to take care of MMA_M x MMA_N repetitions
        // for one thread to advance to the next row (of the next repetition on M-dimension), we need to increment it by warp_row_stride
        // for one thread to advance to the next column (of the next repetition on N-dimension), we need to increment it by warp_col_stride

        // tensor is reshaped from (MMA=4, MMA_M, MMA_N) to ((2, MMA_M), (2, MMA_N))
        // print latex of the MMA Atom: SM80_16x8x16_F32F16F16F32_TN to see the layout, where each thread takes care of (2, 2) elements
        Tensor tensor = make_tensor(tensor_.data(), flash::convert_layout_acc_rowcol(tensor_.layout()));

        // get the thread index
        const int tidx = threadIdx.x;

        // now find which rows and columns this thread is responsible for
        const int lane_id = tidx % 32;
        const int warp_id = tidx / 32;

        const int col_idx_offset = col_idx_offset_ + (lane_id % 4) * 2;
        const int row_idx_offset = row_idx_offset_ + (warp_id * 16) + (lane_id / 4);

        // iterate over MMA_M size<0, 1>(tensor)
        for (int mi = 0; mi < size<0, 1>(tensor); mi++)
        {
            const int row_base_idx = row_idx_offset + mi * warp_row_stride;
            // iterate over 2 size<0, 0>(tensor)
            for (int i = 0; i < size<0, 0>(tensor); i++)
            {   
                const int row_idx = row_base_idx + i * 8;   // Inside 1 MMA Atom, 1 thread is responsible for (2, 2) elements, the distance between 2 rows is 8
                // iterate over MMA_N size<1, 1>(tensor)
                for (int nj = 0; nj < size<1, 1>(tensor); nj++)
                {
                    const int col_base_idx = col_idx_offset + nj * warp_col_stride;
                    // iterate over 2 size<1, 0>(tensor)
                    {
                        for (int j = 0; j < size<1, 0>(tensor); j++)
                        {
                            const int col_idx = col_base_idx + j;   // Inside 1 MMA Atom, 1 thread is responsible for (2, 2) elements, the distance between 2 columns is 1

                            // causal mask
                            if (col_idx > row_idx)
                            {
                                tensor(make_coord(i, mi), make_coord(j, nj)) = -INFINITY;
                            }
                        }
                    }
                }
            }
        }
    }
};

} // end namespace flash