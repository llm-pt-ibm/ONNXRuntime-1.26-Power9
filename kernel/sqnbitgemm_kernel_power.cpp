/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    sqnbitgemm_kernel_power.cpp

Abstract:

    This module implements the SQNBIT_CompFp32 kernels for POWER.

    MLAS previously registered QNBitGemmDispatch only for AVX2/AVX512/NEON/
    LASX, so on POWER it stayed null, MlasIsQNBitGemmAvailable() returned
    false, and MatMulNBits fell back to ComputeBUnpacked() -- which
    dequantizes the whole of B on every call. Providing the two CompFp32
    entry points is enough to put POWER on the MLAS path, where the existing
    POWER9 sgemm kernel (SgemmKernelPower.cpp) does the multiplication.

    B layout consumed here is the unpacked one: quantized values are stored
    column major in blocks of BlkLen values, two 4-bit values per byte, with
    value 2i in the low nibble of byte i and value 2i+1 in the high nibble.
    Packing is the identity (see SQ4BitGemmPackQuantBData below).

--*/

#include <altivec.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>

#include "qnbitgemm.h"

namespace sqnbitgemm_power
{

namespace
{

constexpr size_t BlkBitWidth = 4;

// <altivec.h> defines `vector`, `bool` and `pixel` as keywords; use the
// underscore-prefixed spellings so this stays valid C++.
typedef __vector float VFloat;
typedef __vector unsigned char VUChar;
typedef __vector unsigned short VUShort;
typedef __vector unsigned int VUInt;

//
// Dequantize 16 consecutive 4-bit values (8 bytes) into 4 float vectors,
// in value order: byte i gives value 2i (low nibble) then 2i+1 (high nibble).
//
// Widen 16 unsigned chars to 4 float vectors. On little endian, merging with
// zeros is the zero-extension uchar -> ushort -> uint.
MLAS_FORCEINLINE void
WidenToFloat(VUChar values, VFloat out[4])
{
    const VUChar zero_uc = vec_splats(static_cast<unsigned char>(0));
    const VUShort zero_us = vec_splats(static_cast<unsigned short>(0));

    const VUShort s_lo = reinterpret_cast<VUShort>(vec_mergeh(values, zero_uc));
    const VUShort s_hi = reinterpret_cast<VUShort>(vec_mergel(values, zero_uc));

    out[0] = vec_ctf(reinterpret_cast<VUInt>(vec_mergeh(s_lo, zero_us)), 0);
    out[1] = vec_ctf(reinterpret_cast<VUInt>(vec_mergel(s_lo, zero_us)), 0);
    out[2] = vec_ctf(reinterpret_cast<VUInt>(vec_mergeh(s_hi, zero_us)), 0);
    out[3] = vec_ctf(reinterpret_cast<VUInt>(vec_mergel(s_hi, zero_us)), 0);
}

//
// Dequantize 32 consecutive 4-bit values (one full 16-byte load) into 8 float
// vectors. block_size=32 is the common quantization choice -- and what
// Phi-3-mini ships -- so this covers a whole block per load, halving the
// per-value load/mask/shift overhead relative to the 16-value path.
//
MLAS_FORCEINLINE void
Unpack32Nibbles(const std::byte* src, VFloat out[8])
{
    const VUChar bytes = vec_xl(0, reinterpret_cast<const unsigned char*>(src));
    const VUChar lo = vec_and(bytes, vec_splats(static_cast<unsigned char>(0x0F)));
    const VUChar hi = vec_sr(bytes, vec_splats(static_cast<unsigned char>(4)));

    WidenToFloat(vec_mergeh(lo, hi), out);      // values 0..15
    WidenToFloat(vec_mergel(lo, hi), out + 4);  // values 16..31
}

MLAS_FORCEINLINE void
Unpack16Nibbles(const std::byte* src, VFloat out[4])
{
    uint64_t packed;
    std::memcpy(&packed, src, sizeof(packed));

    const VUChar bytes = reinterpret_cast<VUChar>(vec_splats(packed));
    const VUChar lo = vec_and(bytes, vec_splats(static_cast<unsigned char>(0x0F)));
    const VUChar hi = vec_sr(bytes, vec_splats(static_cast<unsigned char>(4)));

    // Interleaving low/high nibbles of bytes 0..7 yields values 0..15 in order.
    const VUChar values = vec_mergeh(lo, hi);

    // Zero-extend uchar -> ushort -> uint, then convert. On little endian,
    // merging with zeros is the zero-extension.
    const VUChar zero_uc = vec_splats(static_cast<unsigned char>(0));
    const VUShort zero_us = vec_splats(static_cast<unsigned short>(0));

    const VUShort s_lo = reinterpret_cast<VUShort>(vec_mergeh(values, zero_uc));
    const VUShort s_hi = reinterpret_cast<VUShort>(vec_mergel(values, zero_uc));

    out[0] = vec_ctf(reinterpret_cast<VUInt>(vec_mergeh(s_lo, zero_us)), 0);
    out[1] = vec_ctf(reinterpret_cast<VUInt>(vec_mergel(s_lo, zero_us)), 0);
    out[2] = vec_ctf(reinterpret_cast<VUInt>(vec_mergeh(s_hi, zero_us)), 0);
    out[3] = vec_ctf(reinterpret_cast<VUInt>(vec_mergel(s_hi, zero_us)), 0);
}

// Width of the column tile the sgemm kernel expects in its packed B: it loads
// 16 consecutive floats per K step (FgemmKernelpower.h loads B, B+4, B+8, B+12).
constexpr size_t SgemmPackBWidth = 16;

// Columns processed together in the M=1 kernel. Four fills a float vector, so
// the per-column block sums reduce into one store.
constexpr size_t NCols = 4;

MLAS_FORCEINLINE float
DequantZeroPoint(const std::byte* QuantBZeroPointCol, size_t k_blk_idx)
{
    // Zero points are 4 bits each, two per byte, indexed by block.
    if (QuantBZeroPointCol == nullptr) {
        return 8.0f;  // implicit zero point
    }
    const std::byte packed = QuantBZeroPointCol[k_blk_idx / 2];
    const std::byte zp = ((k_blk_idx & 1) == 1) ? (packed >> 4) : (packed & std::byte{0x0F});
    return static_cast<float>(std::to_integer<uint8_t>(zp));
}

MLAS_FORCEINLINE float
DequantValue(const std::byte* BlockData, size_t kk)
{
    // Value kk lives in byte kk/2: low nibble when kk is even, high nibble when odd.
    const std::byte b = BlockData[kk / 2];
    const uint8_t v = ((kk & 1) == 1)
                          ? std::to_integer<uint8_t>(b >> 4)
                          : std::to_integer<uint8_t>(b & std::byte{0x0F});
    return static_cast<float>(v);
}

}  // namespace

//
// Packing: the kernels below read the quantized data in its original layout,
// so packing is a straight copy. It still has to be registered -- MatMulNBits
// passes packed_b_ as PackedQuantBData unconditionally, and that buffer only
// exists if PackQuantBDataSize returns non-zero.
//

size_t
QNBitGemmPackQuantBDataSize(
    size_t N,
    size_t K,
    size_t BlkLen,
    bool /*HasZeroPoint*/,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    return N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
}

void
SQ4BitGemmPackQuantBData(
    size_t N,
    size_t K,
    size_t BlkLen,
    MLAS_QNBIT_GEMM_COMPUTE_TYPE /*ComputeType*/,
    const std::byte* QuantBDataBegin,
    std::byte* PackedQuantBDataBegin,
    MLAS_THREADPOOL* /*ThreadPool*/,
    const MLAS_BACKEND_KERNEL_SELECTOR_CONFIG* /*BackendKernelSelectorConfig*/
)
{
    const size_t BlockCountK = MlasDivRoundup(K, BlkLen);
    const size_t Size = N * BlockCountK * MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    std::memcpy(PackedQuantBDataBegin, QuantBDataBegin, Size);
}

//
// SQNBIT_CompFp32 kernels.
//

void
SQ4BitGemmM1Kernel_CompFp32(
    size_t BlkLen,
    const float* A,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    float* C,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK,
    const float* Bias
)
{
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBData = BlockCountK * BlkDataSize;
    const size_t StrideQuantBZeroPoint =
        MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    size_t n = 0;

    // Handle NCols columns at a time so each A vector is loaded once and reused
    // across all of them. A is tiny (K floats) and stays in cache, but with one
    // column at a time it is still re-read CountN times, and the loads dominate.
    for (; n + NCols <= CountN; n += NCols) {
        // Per-column running totals, kept in vector form: a block's accumulator
        // is folded in with a single FMA (acc * scale + total) instead of being
        // horizontally reduced. With BlkLen=32 there are only 8 vectors of work
        // per block, so a per-block horizontal reduce -- four extracts and three
        // adds on POWER -- was a large share of the cost.
        VFloat total[NCols];
        float tail_total[NCols];
        for (size_t c = 0; c < NCols; ++c) {
            total[c] = vec_splats(0.0f);
            tail_total[c] = 0.0f;
        }

        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, ++k_blk_idx) {
            const size_t kklen = std::min(CountK - k, BlkLen);

            const std::byte* BlockData[NCols];
            VFloat zp_v[NCols];
            VFloat acc[NCols];
            float scale[NCols];

            for (size_t c = 0; c < NCols; ++c) {
                const size_t col = n + c;
                BlockData[c] = QuantBData + col * StrideQuantBData + k_blk_idx * BlkDataSize;
                scale[c] = QuantBScale[col * BlockCountK + k_blk_idx];
                const std::byte* zp_col =
                    (QuantBZeroPoint == nullptr)
                        ? nullptr
                        : QuantBZeroPoint + col * StrideQuantBZeroPoint;
                zp_v[c] = vec_splats(DequantZeroPoint(zp_col, k_blk_idx));
                acc[c] = vec_splats(0.0f);
            }

            size_t kk = 0;
            for (; kk + 32 <= kklen; kk += 32) {
                const float* a = A + k + kk;
                VFloat av[8];
                for (size_t j = 0; j < 8; ++j) {
                    av[j] = vec_xl(static_cast<long>(16 * j), a);
                }

                for (size_t c = 0; c < NCols; ++c) {
                    VFloat vals[8];
                    Unpack32Nibbles(BlockData[c] + kk / 2, vals);
                    for (size_t j = 0; j < 8; ++j) {
                        acc[c] = vec_madd(av[j], vec_sub(vals[j], zp_v[c]), acc[c]);
                    }
                }
            }
            for (; kk + 16 <= kklen; kk += 16) {
                const float* a = A + k + kk;
                const VFloat a0 = vec_xl(0, a);
                const VFloat a1 = vec_xl(16, a);
                const VFloat a2 = vec_xl(32, a);
                const VFloat a3 = vec_xl(48, a);

                for (size_t c = 0; c < NCols; ++c) {
                    VFloat vals[4];
                    Unpack16Nibbles(BlockData[c] + kk / 2, vals);
                    acc[c] = vec_madd(a0, vec_sub(vals[0], zp_v[c]), acc[c]);
                    acc[c] = vec_madd(a1, vec_sub(vals[1], zp_v[c]), acc[c]);
                    acc[c] = vec_madd(a2, vec_sub(vals[2], zp_v[c]), acc[c]);
                    acc[c] = vec_madd(a3, vec_sub(vals[3], zp_v[c]), acc[c]);
                }
            }

            for (size_t c = 0; c < NCols; ++c) {
                total[c] = vec_madd(acc[c], vec_splats(scale[c]), total[c]);

                // Values past the last full group of 16 (only the final block
                // when K is not a multiple of BlkLen).
                if (kk < kklen) {
                    const float zp = vec_extract(zp_v[c], 0);
                    float s = 0.0f;
                    for (size_t kt = kk; kt < kklen; ++kt) {
                        s += A[k + kt] * (DequantValue(BlockData[c], kt) - zp);
                    }
                    tail_total[c] += s * scale[c];
                }
            }
        }

        for (size_t c = 0; c < NCols; ++c) {
            float sum = total[c][0] + total[c][1] + total[c][2] + total[c][3] + tail_total[c];
            C[n + c] = Bias != nullptr ? sum + Bias[n + c] : sum;
        }
    }

    // Remaining columns, one at a time.
    for (; n < CountN; ++n) {
        const std::byte* QuantBDataCol = QuantBData + n * StrideQuantBData;
        const float* QuantBScaleCol = QuantBScale + n * BlockCountK;
        const std::byte* QuantBZeroPointCol =
            (QuantBZeroPoint == nullptr) ? nullptr : QuantBZeroPoint + n * StrideQuantBZeroPoint;

        float sum = 0.0f;

        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, ++k_blk_idx) {
            const float scale = QuantBScaleCol[k_blk_idx];
            const float zp = DequantZeroPoint(QuantBZeroPointCol, k_blk_idx);
            const std::byte* BlockData = QuantBDataCol + k_blk_idx * BlkDataSize;

            const size_t kklen = std::min(CountK - k, BlkLen);

            // The scale is constant across the block, so accumulate
            // sum(A * (v - zp)) first and scale once at the end.
            const VFloat zp_v = vec_splats(zp);
            VFloat acc = vec_splats(0.0f);

            size_t kk = 0;
            for (; kk + 16 <= kklen; kk += 16) {
                VFloat vals[4];
                Unpack16Nibbles(BlockData + kk / 2, vals);

                const float* a = A + k + kk;
                acc = vec_madd(vec_xl(0, a), vec_sub(vals[0], zp_v), acc);
                acc = vec_madd(vec_xl(16, a), vec_sub(vals[1], zp_v), acc);
                acc = vec_madd(vec_xl(32, a), vec_sub(vals[2], zp_v), acc);
                acc = vec_madd(vec_xl(48, a), vec_sub(vals[3], zp_v), acc);
            }

            float blk_sum = acc[0] + acc[1] + acc[2] + acc[3];

            for (; kk < kklen; ++kk) {
                blk_sum += A[k + kk] * (DequantValue(BlockData, kk) - zp);
            }

            sum += blk_sum * scale;
        }

        C[n] = Bias != nullptr ? sum + Bias[n] : sum;
    }
}

void
SQ4BitBlkDequantBForSgemm_CompFp32(
    size_t BlkLen,
    float* FpData,
    const std::byte* QuantBData,
    const float* QuantBScale,
    const std::byte* QuantBZeroPoint,
    size_t CountN,
    size_t CountK,
    size_t BlockCountK
)
{
    const size_t BlkDataSize = MlasQNBitBlkDataSizeInBytes(BlkBitWidth, BlkLen);
    const size_t StrideQuantBData = BlockCountK * BlkDataSize;
    const size_t StrideQuantBZeroPoint =
        MlasQNBitZeroPointsForBlksSizeInBytes<BlkBitWidth>(BlockCountK);

    float* Dst = FpData;

    // One tile of SgemmPackBWidth columns at a time; within a tile the sgemm
    // kernel wants, for each k, the SgemmPackBWidth column values contiguous.
    for (size_t n0 = 0; n0 < CountN; n0 += SgemmPackBWidth) {
        const size_t ncols = std::min(SgemmPackBWidth, CountN - n0);

        for (size_t k = 0, k_blk_idx = 0; k < CountK; k += BlkLen, ++k_blk_idx) {
            const size_t kklen = std::min(CountK - k, BlkLen);

            for (size_t nn = 0; nn < SgemmPackBWidth; ++nn) {
                if (nn >= ncols) {
                    // Pad the tile so the kernel's fixed-width loads stay valid.
                    for (size_t kk = 0; kk < kklen; ++kk) {
                        Dst[kk * SgemmPackBWidth + nn] = 0.0f;
                    }
                    continue;
                }

                const size_t n = n0 + nn;
                const std::byte* BlockData =
                    QuantBData + n * StrideQuantBData + k_blk_idx * BlkDataSize;
                const float scale = QuantBScale[n * BlockCountK + k_blk_idx];
                const std::byte* QuantBZeroPointCol =
                    (QuantBZeroPoint == nullptr)
                        ? nullptr
                        : QuantBZeroPoint + n * StrideQuantBZeroPoint;
                const float zp = DequantZeroPoint(QuantBZeroPointCol, k_blk_idx);

                for (size_t kk = 0; kk < kklen; ++kk) {
                    Dst[kk * SgemmPackBWidth + nn] =
                        (DequantValue(BlockData, kk) - zp) * scale;
                }
            }

            Dst += kklen * SgemmPackBWidth;
        }
    }
}

}  // namespace sqnbitgemm_power

//
// Dispatch structure.
//

const MLAS_QNBIT_GEMM_DISPATCH MlasQNBitGemmDispatchPower = []() {
    MLAS_QNBIT_GEMM_DISPATCH d;

    d.Q4BitGemmPackQuantBDataSize = sqnbitgemm_power::QNBitGemmPackQuantBDataSize;
    d.SQ4BitGemmPackQuantBData = sqnbitgemm_power::SQ4BitGemmPackQuantBData;

    d.SQ4BitGemmM1Kernel_CompFp32 = sqnbitgemm_power::SQ4BitGemmM1Kernel_CompFp32;
    d.SQ4BitBlkDequantBForSgemm_CompFp32 = sqnbitgemm_power::SQ4BitBlkDequantBForSgemm_CompFp32;

    return d;
}();
