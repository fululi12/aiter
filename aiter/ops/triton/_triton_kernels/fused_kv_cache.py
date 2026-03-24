import triton
import triton.language as tl
from aiter.ops.triton.rope import _get_gptj_rotated_x_1D, _get_neox_rotated_x_1D


@triton.jit
def _unit_cat(
    x1_ptr,
    x2_ptr,
    x_out_ptr,
    b_in,
    b_out,
    h,
    d1_offs,
    d2_offs,
    x1_stride_b,
    x1_stride_h,
    x1_stride_d,
    x2_stride_b,
    x2_stride_h,
    x2_stride_d,
    x_out_stride_b,
    x_out_stride_h,
    x_out_stride_d,
    k_scale,
    BLOCK_D1: tl.constexpr,
):
    x1_offs = b_in * x1_stride_b + h * x1_stride_h + d1_offs * x1_stride_d
    x2_offs = b_in * x2_stride_b + h * x2_stride_h + d2_offs * x2_stride_d
    x_out_offs = b_out * x_out_stride_b + h * x_out_stride_h

    x1 = tl.load(x1_ptr + x1_offs)
    x2 = tl.load(x2_ptr + x2_offs)

    x1 = (x1 / k_scale).to(x_out_ptr.dtype.element_ty)
    x2 = (x2 / k_scale).to(x_out_ptr.dtype.element_ty)
    tl.store(x_out_ptr + x_out_offs + d1_offs * x_out_stride_d, x1)
    tl.store(x_out_ptr + x_out_offs + (d2_offs + BLOCK_D1) * x_out_stride_d, x2)


@triton.jit
def _unit_rope_cat(
    x_nope_ptr,
    x_pe_ptr,
    cos,
    sin,
    x_out_ptr,
    b_in,
    b_out,
    h,
    d_nope_offs,
    d_pe_offs,
    x_nope_stride_b,
    x_nope_stride_h,
    x_nope_stride_d,
    x_pe_stride_b,
    x_pe_stride_h,
    x_pe_stride_d,
    x_out_stride_b,
    x_out_stride_h,
    x_out_stride_d,
    k_scale,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    x_nope_offs = (
        b_in * x_nope_stride_b + h * x_nope_stride_h + d_nope_offs * x_nope_stride_d
    )
    x_pe_offs = b_in * x_pe_stride_b + h * x_pe_stride_h + d_pe_offs * x_pe_stride_d
    x_out_offs = b_out * x_out_stride_b + h * x_out_stride_h

    x_nope = tl.load(x_nope_ptr + x_nope_offs)
    x_pe = tl.load(x_pe_ptr + x_pe_offs)

    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    x_pe = x_pe * cos + x_pe_rotated * sin
    x_pe = x_pe / k_scale
    x_nope = x_nope / k_scale
    x_nope = x_nope.to(x_out_ptr.dtype.element_ty)
    x_pe = x_pe.to(x_out_ptr.dtype.element_ty)

    tl.store(x_out_ptr + x_out_offs + d_nope_offs * x_out_stride_d, x_nope)
    tl.store(x_out_ptr + x_out_offs + (d_pe_offs + BLOCK_D_nope) * x_out_stride_d, x_pe)


@triton.jit
def _fused_qk_rope_cat_and_cache_mla_kernel(
    q_nope_ptr,
    q_pe_ptr,
    k_nope_ptr,
    k_pe_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    q_out_ptr,
    decode_q_pe_out_ptr,
    k_pe_out_ptr,
    q_nope_zeros_out_ptr,
    kv_cache_ptr,
    slot_mapping_ptr,
    B,
    B_slot,
    num_decode_toks_for_zeros,
    q_nope_stride_b,
    q_nope_stride_h,
    q_nope_stride_d,
    q_pe_stride_b,
    q_pe_stride_h,
    q_pe_stride_d,
    k_nope_stride_b,
    k_nope_stride_h,
    k_nope_stride_d,
    k_pe_stride_b,
    k_pe_stride_h,
    k_pe_stride_d,
    pos_stride_b,
    cos_stride_b,
    cos_stride_d,
    q_out_stride_b,
    q_out_stride_h,
    q_out_stride_d,
    decode_q_pe_out_stride_b,
    decode_q_pe_out_stride_h,
    decode_q_pe_out_stride_d,
    k_pe_out_stride_b,
    k_pe_out_stride_h,
    k_pe_out_stride_d,
    q_nope_zeros_out_stride_b,
    q_nope_zeros_out_stride_h,
    q_nope_zeros_out_stride_d,
    kv_cache_stride_b,
    kv_cache_stride_h,
    kv_cache_stride_d,
    k_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_nope: tl.constexpr,
    BLOCK_DK_nope: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    OUTPUT_Q_NOPE_ZEROS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)

    d_nope_offs = tl.arange(0, BLOCK_D_nope).to(tl.int64)
    dk_nope_offs = tl.arange(0, BLOCK_DK_nope).to(tl.int64)
    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    if pid < B * QH:
        pid_b = pid // QH
        pid_hq = pid % QH
        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
                # d_cos_mask = d_cos_offs < BLOCK_D_pe
            else:
                d_cos_offs = d_pe_offs // 2
                # d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe
        else:
            d_cos_offs = d_pe_offs
            # d_cos_mask = d_cos_offs < BLOCK_D_pe

        pos = tl.load(pos_ptr + pid_b * pos_stride_b)
        cos_offs = pos * cos_stride_b + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs)
        sin = tl.load(sin_ptr + cos_offs)

        q_nope_ptrs = (
            q_nope_ptr
            + pid_b * q_nope_stride_b
            + pid_hq * q_nope_stride_h
            + d_nope_offs * q_nope_stride_d
        )
        q_pe_ptrs = (
            q_pe_ptr
            + pid_b * q_pe_stride_b
            + pid_hq * q_pe_stride_h
            + d_pe_offs * q_pe_stride_d
        )
        q_out_ptrs = q_out_ptr + pid_b * q_out_stride_b + pid_hq * q_out_stride_h
        q_nope = tl.load(q_nope_ptrs)
        q_pe = _unit_rope(
            q_pe_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )
        tl.store(
            q_out_ptrs + d_nope_offs * q_out_stride_d,
            q_nope.to(q_out_ptr.dtype.element_ty),
        )
        tl.store(
            q_out_ptrs + (d_pe_offs + BLOCK_D_nope) * q_out_stride_d,
            q_pe.to(q_out_ptr.dtype.element_ty),
        )

        if pid < num_decode_toks_for_zeros * QH:
            decode_q_pe_out_ptrs = (
                decode_q_pe_out_ptr
                + pid_b * decode_q_pe_out_stride_b
                + pid_hq * decode_q_pe_out_stride_h
            )
            tl.store(
                decode_q_pe_out_ptrs + d_pe_offs * decode_q_pe_out_stride_d,
                q_pe.to(decode_q_pe_out_ptr.dtype.element_ty),
            )

        if OUTPUT_Q_NOPE_ZEROS:
            if pid < num_decode_toks_for_zeros * QH:
                z = tl.zeros(
                    (BLOCK_DK_nope,), dtype=q_nope_zeros_out_ptr.dtype.element_ty
                )
                tl.store(
                    q_nope_zeros_out_ptr
                    + pid_b * q_nope_zeros_out_stride_b
                    + pid_hq * q_nope_zeros_out_stride_h
                    + dk_nope_offs * q_nope_zeros_out_stride_d,
                    z,
                )

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                pid_hk = pid_hq // QH_PER_KH
                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                kv_cache_ptrs = (
                    kv_cache_ptr
                    + pid_slot * kv_cache_stride_b
                    + pid_hk * kv_cache_stride_h
                )
                k_nope = tl.load(k_nope_ptrs)
                k_pe = _unit_rope(
                    k_pe_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))
                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope)
                tl.store(
                    kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                    k_pe,
                )
    else:
        pid = pid - B * QH + B * KH
        if pid < B_slot * KH:
            pid_b = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_b).to(tl.int64)
            if pid_slot >= 0:
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1

                k_nope_ptrs = (
                    k_nope_ptr
                    + pid_b * k_nope_stride_b
                    + pid_hk * k_nope_stride_h
                    + dk_nope_offs * k_nope_stride_d
                )
                k_pe_ptrs = (
                    k_pe_ptr
                    + pid_b * k_pe_stride_b
                    + pid_hk * k_pe_stride_h
                    + d_pe_offs * k_pe_stride_d
                )
                k_pe_out_ptrs = (
                    k_pe_out_ptr
                    + pid_b * k_pe_out_stride_b
                    + pid_hk * k_pe_out_stride_h
                    + d_pe_offs * k_pe_out_stride_d
                )
                kv_cache_ptrs = (
                    kv_cache_ptr
                    + pid_slot * kv_cache_stride_b
                    + pid_hk * kv_cache_stride_h
                )
                k_nope = tl.load(k_nope_ptrs)
                k_pe = tl.load(k_pe_ptrs)
                tl.store(k_pe_out_ptrs, k_pe.to(k_pe_out_ptr.dtype.element_ty))
                k_scale_rcprl = (1 / k_scale).to(tl.float32)
                k_nope = (k_nope.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                k_pe = (k_pe.to(tl.float32) * k_scale_rcprl).to(
                    kv_cache_ptr.dtype.element_ty
                )
                tl.store(kv_cache_ptrs + dk_nope_offs * kv_cache_stride_d, k_nope)
                tl.store(
                    kv_cache_ptrs + (d_pe_offs + BLOCK_DK_nope) * kv_cache_stride_d,
                    k_pe,
                )


@triton.jit
def _unit_rope(
    x_ptrs,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    x_pe = tl.load(x_ptrs)

    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    x_pe = x_pe * cos + x_pe_rotated * sin

    return x_pe


@triton.jit
def _fp4_quantize_nibbles(scaled):
    """Convert pre-scaled float32 values to FP4 E2M1 nibbles."""
    absval = tl.abs(scaled)
    sign = tl.where(scaled < 0.0, 8, 0)
    mag = tl.where(absval < 0.25, 0,
          tl.where(absval < 0.75, 1,
          tl.where(absval < 1.25, 2,
          tl.where(absval < 1.75, 3,
          tl.where(absval < 2.5, 4,
          tl.where(absval < 3.5, 5,
          tl.where(absval < 5.0, 6, 7)))))))
    return sign | mag


@triton.jit
def _fp4_pack_nibbles(nibbles, BLOCK_D: tl.constexpr):
    """Pack FP4 nibble pairs into uint8 bytes."""
    d_offs = tl.arange(0, BLOCK_D)
    nibbles_shifted = tl.where(d_offs % 2 == 0, nibbles, nibbles << 4)
    nibbles_2d = tl.reshape(nibbles_shifted, (BLOCK_D // 2, 2))
    return tl.sum(nibbles_2d, axis=1).to(tl.uint8)


@triton.jit
def _f32_to_fp4_e2m1_packed(
    data,
    BLOCK_D: tl.constexpr,
    FP4_CLIP_SIGMA: tl.constexpr = 0,
):
    """Quantize float32 vector to packed FP4 E2M1 with per-token scaling."""
    data_f32 = data.to(tl.float32)
    absmax = tl.max(tl.abs(data_f32))

    effective_max = absmax
    if FP4_CLIP_SIGMA > 0:
        sum_sq = tl.sum(data_f32 * data_f32)
        rms = tl.sqrt(sum_sq / BLOCK_D)
        clip_val = rms * FP4_CLIP_SIGMA
        effective_max = tl.minimum(absmax, clip_val)

    scale = tl.where(effective_max > 0.0, effective_max / 6.0, 1.0)
    scaled = data_f32 / scale
    nibbles = _fp4_quantize_nibbles(scaled)
    packed = _fp4_pack_nibbles(nibbles, BLOCK_D)
    return packed, scale


@triton.jit
def _fp4_dequant_nibbles(nibbles):
    """Dequantize FP4 E2M1 nibbles to float32 magnitude * sign."""
    mag_idx = nibbles & 7
    mag = tl.where(mag_idx == 0, 0.0,
          tl.where(mag_idx == 1, 0.5,
          tl.where(mag_idx == 2, 1.0,
          tl.where(mag_idx == 3, 1.5,
          tl.where(mag_idx == 4, 2.0,
          tl.where(mag_idx == 5, 3.0,
          tl.where(mag_idx == 6, 4.0, 6.0)))))))
    sign = tl.where((nibbles & 8) != 0, -1.0, 1.0)
    return mag * sign


@triton.jit
def _f32_to_fp4_e2m1_packed_block32(
    data,
    BLOCK_D: tl.constexpr,
    FP4_BLOCK_SIZE: tl.constexpr = 32,
    FP4_CLIP_SIGMA: tl.constexpr = 0,
    FP4_MSE_REFINE_ITERS: tl.constexpr = 0,
):
    """Quantize float32 vector to packed FP4 E2M1 with per-block-32 scaling.

    Supports optional outlier clipping (FP4_CLIP_SIGMA > 0) and MSE-optimal
    scale refinement (FP4_MSE_REFINE_ITERS > 0) per block.

    Returns (packed_bytes, block_scales) where block_scales has shape
    [BLOCK_D // FP4_BLOCK_SIZE].
    """
    NUM_BLOCKS: tl.constexpr = BLOCK_D // FP4_BLOCK_SIZE

    data_f32 = data.to(tl.float32)
    data_2d = tl.reshape(data_f32, (NUM_BLOCKS, FP4_BLOCK_SIZE))
    abs_2d = tl.abs(data_2d)
    block_absmax = tl.max(abs_2d, axis=1)

    effective_max = block_absmax
    if FP4_CLIP_SIGMA > 0:
        block_sum_sq = tl.sum(data_2d * data_2d, axis=1)
        block_rms = tl.sqrt(block_sum_sq / FP4_BLOCK_SIZE)
        clip_val = block_rms * FP4_CLIP_SIGMA
        effective_max = tl.minimum(block_absmax, clip_val)

    block_scales = tl.where(effective_max > 0.0, effective_max / 6.0, 1.0)

    if FP4_MSE_REFINE_ITERS > 0:
        for _iter in range(FP4_MSE_REFINE_ITERS):
            scaled_2d = data_2d / block_scales[:, None]
            scaled_flat = tl.reshape(scaled_2d, (BLOCK_D,))
            q_nibbles = _fp4_quantize_nibbles(scaled_flat)
            q_vals = _fp4_dequant_nibbles(q_nibbles)
            q_2d = tl.reshape(q_vals, (NUM_BLOCKS, FP4_BLOCK_SIZE))
            xq_sum = tl.sum(data_2d * q_2d, axis=1)
            qq_sum = tl.sum(q_2d * q_2d, axis=1)
            block_scales = tl.where(qq_sum > 0.0,
                                    tl.maximum(xq_sum / qq_sum, 1e-12),
                                    block_scales)

    scaled_2d = data_2d / block_scales[:, None]
    scaled = tl.reshape(scaled_2d, (BLOCK_D,))
    nibbles = _fp4_quantize_nibbles(scaled)
    packed = _fp4_pack_nibbles(nibbles, BLOCK_D)
    return packed, block_scales


@triton.jit
def _f32_to_fp4_e2m1_packed_per_channel_k(
    data,
    ch_scales,
    BLOCK_D: tl.constexpr,
):
    """Quantize float32 vector to packed FP4 E2M1 using pre-computed
    per-channel scales (one scale per head-dim element).

    ``ch_scales`` has shape ``[BLOCK_D]`` and each element ``d`` is used
    as the quantization scale for ``data[d]``.

    Returns packed_bytes (shape ``[BLOCK_D // 2]``).
    """
    data_f32 = data.to(tl.float32)
    ch_s = ch_scales.to(tl.float32)
    scaled = data_f32 / ch_s
    nibbles = _fp4_quantize_nibbles(scaled)
    packed = _fp4_pack_nibbles(nibbles, BLOCK_D)
    return packed


@triton.jit
def _compute_k_channel_scales_triton(
    data_flat,
    BLOCK_D: tl.constexpr,
    N_TOKENS: tl.constexpr,
    FP4_CLIP_SIGMA: tl.constexpr = 0,
    FP4_MSE_REFINE_ITERS: tl.constexpr = 2,
    SAFETY_MARGIN: tl.constexpr = 1,
):
    """Compute per-channel K scales from a batch of tokens.

    ``data_flat`` is a 2D view ``[N_TOKENS, BLOCK_D]`` of K data.
    ``SAFETY_MARGIN`` (integer, >= 1): multiplied into scales after
    computation.  Use 1 for no margin.
    Returns ``ch_scales`` of shape ``[BLOCK_D]``.
    """
    data_2d = tl.reshape(data_flat, (N_TOKENS, BLOCK_D))
    abs_2d = tl.abs(data_2d)
    ch_max = tl.max(abs_2d, axis=0)

    effective_max = ch_max
    if FP4_CLIP_SIGMA > 0:
        sq_2d = data_2d * data_2d
        ch_mean_sq = tl.sum(sq_2d, axis=0) / N_TOKENS
        ch_rms = tl.sqrt(ch_mean_sq)
        effective_max = tl.minimum(ch_max, ch_rms * FP4_CLIP_SIGMA)

    ch_scales = tl.where(effective_max > 0.0, effective_max / 6.0, 1e-12)

    if FP4_MSE_REFINE_ITERS > 0:
        for _iter in range(FP4_MSE_REFINE_ITERS):
            scaled_2d = data_2d / ch_scales[None, :]
            q_flat = _fp4_quantize_nibbles(tl.reshape(scaled_2d, (N_TOKENS * BLOCK_D,)))
            q_vals = _fp4_dequant_nibbles(q_flat)
            q_2d = tl.reshape(q_vals, (N_TOKENS, BLOCK_D))
            xq_sum = tl.sum(data_2d * q_2d, axis=0)
            qq_sum = tl.sum(q_2d * q_2d, axis=0)
            ch_scales = tl.where(qq_sum > 0.0,
                                 tl.maximum(xq_sum / qq_sum, 1e-12),
                                 ch_scales)

    if SAFETY_MARGIN > 1:
        ch_scales = ch_scales * SAFETY_MARGIN

    return ch_scales


@triton.jit
def _f32_to_fp4_e2m1_packed_per_token(
    data,
    BLOCK_D: tl.constexpr,
    FP4_CLIP_SIGMA: tl.constexpr = 0,
    FP4_MSE_REFINE_ITERS: tl.constexpr = 0,
):
    """Quantize float32 vector to packed FP4 E2M1 with a single per-token
    scale (shared across all ``BLOCK_D`` elements).

    Returns (packed_bytes, token_scale) where token_scale is a scalar.
    """
    data_f32 = data.to(tl.float32)
    abs_data = tl.abs(data_f32)
    absmax = tl.max(abs_data, axis=0)

    effective_max = absmax
    if FP4_CLIP_SIGMA > 0:
        sum_sq = tl.sum(data_f32 * data_f32, axis=0)
        rms = tl.sqrt(sum_sq / BLOCK_D)
        effective_max = tl.minimum(absmax, rms * FP4_CLIP_SIGMA)

    token_scale = tl.where(effective_max > 0.0, effective_max / 6.0, 1e-12)

    if FP4_MSE_REFINE_ITERS > 0:
        for _iter in range(FP4_MSE_REFINE_ITERS):
            scaled = data_f32 / token_scale
            q_nib = _fp4_quantize_nibbles(scaled)
            q_vals = _fp4_dequant_nibbles(q_nib)
            xq_sum = tl.sum(data_f32 * q_vals, axis=0)
            qq_sum = tl.sum(q_vals * q_vals, axis=0)
            token_scale = tl.where(qq_sum > 0.0,
                                   tl.maximum(xq_sum / qq_sum, 1e-12),
                                   token_scale)

    scaled = data_f32 / token_scale
    nibbles = _fp4_quantize_nibbles(scaled)
    packed = _fp4_pack_nibbles(nibbles, BLOCK_D)
    return packed, token_scale


@triton.jit
def _float_to_e8m0(x):
    """Convert positive float to E8M0 byte (power-of-2 scale).

    E8M0 encoding: e8m0 = round(log2(x)) + 127.
    Value decoded as: 2^(e8m0 - 127).
    """
    log2_x = tl.log2(x)
    biased = tl.libdevice.rint(log2_x).to(tl.int32) + 127
    biased = tl.maximum(biased, 0)
    biased = tl.minimum(biased, 254)
    return biased.to(tl.uint8)


@triton.jit
def _e8m0_to_float(e8m0):
    """Convert E8M0 byte to float scale: value = 2^(e8m0 - 127)."""
    return tl.exp2((e8m0.to(tl.float32) - 127.0))


@triton.jit
def _f32_to_fp4_e2m1_packed_mxfp4(
    data,
    BLOCK_D: tl.constexpr,
    FP4_BLOCK_SIZE: tl.constexpr = 32,
):
    """Quantize float32 vector to packed FP4 E2M1 with MXFP4 (OCP MX)
    per-block-32 E8M0 (power-of-2) scales.

    Returns (packed_bytes, e8m0_scales) where e8m0_scales is uint8 with
    shape [BLOCK_D // FP4_BLOCK_SIZE].
    """
    NUM_BLOCKS: tl.constexpr = BLOCK_D // FP4_BLOCK_SIZE

    data_f32 = data.to(tl.float32)
    data_2d = tl.reshape(data_f32, (NUM_BLOCKS, FP4_BLOCK_SIZE))
    abs_2d = tl.abs(data_2d)
    block_absmax = tl.max(abs_2d, axis=1)

    ideal_scales = tl.where(block_absmax > 0.0, block_absmax / 6.0, 1e-30)
    e8m0_scales = _float_to_e8m0(ideal_scales)
    float_scales = _e8m0_to_float(e8m0_scales)
    float_scales = tl.where(float_scales > 0.0, float_scales, 1e-30)

    scaled_2d = data_2d / float_scales[:, None]
    scaled = tl.reshape(scaled_2d, (BLOCK_D,))
    nibbles = _fp4_quantize_nibbles(scaled)
    packed = _fp4_pack_nibbles(nibbles, BLOCK_D)
    return packed, e8m0_scales


@triton.jit
def _float_to_fp8_e4m3(x):
    """Convert positive float to FP8 E4M3 FNUZ byte (1 sign + 4 exp + 3 mant).

    E4M3 FNUZ: bias=8, max=240.
    Normal: 2^(E-8) * (1 + M/8)   for E > 0
    Subnormal: 2^(-7) * (M/8)     for E = 0
    """
    x = tl.maximum(x, 0.0)
    x = tl.minimum(x, 240.0)

    log2_x = tl.log2(tl.maximum(x, 1e-30))
    exp_unbiased = tl.libdevice.floor(log2_x).to(tl.int32)
    biased_exp = exp_unbiased + 8

    power = tl.exp2(exp_unbiased.to(tl.float32))
    frac = x / tl.maximum(power, 1e-30) - 1.0
    mantissa = tl.libdevice.rint(frac * 8.0).to(tl.int32)
    mantissa = tl.maximum(mantissa, 0)
    mantissa = tl.minimum(mantissa, 7)

    biased_exp = tl.maximum(biased_exp, 0)
    biased_exp = tl.minimum(biased_exp, 15)

    result = (biased_exp << 3) | mantissa
    result = tl.where(x <= 0.0, 0, result)
    return result.to(tl.uint8)


@triton.jit
def _fp8_e4m3_to_float(fp8):
    """Convert FP8 E4M3 FNUZ byte to float scale."""
    fp8_i32 = fp8.to(tl.int32)
    exp_bits = (fp8_i32 >> 3) & 0xF
    mantissa = fp8_i32 & 0x7
    normal = tl.exp2((exp_bits - 8).to(tl.float32)) * (
        1.0 + mantissa.to(tl.float32) / 8.0
    )
    subnormal = tl.exp2(tl.full(mantissa.shape, -7.0, tl.float32)) * (
        mantissa.to(tl.float32) / 8.0
    )
    result = tl.where(exp_bits == 0, subnormal, normal)
    result = tl.where(fp8_i32 == 0, 0.0, result)
    return result


@triton.jit
def _f32_to_fp4_e2m1_packed_nvfp4(
    data,
    BLOCK_D: tl.constexpr,
    FP4_BLOCK_SIZE: tl.constexpr = 32,
):
    """Quantize float32 vector to packed FP4 E2M1 with NVFP4-style
    per-block-32 FP8 E4M3 scales (3 mantissa bits of precision).

    Returns (packed_bytes, fp8_scales) where fp8_scales is uint8 with
    shape [BLOCK_D // FP4_BLOCK_SIZE].
    """
    NUM_BLOCKS: tl.constexpr = BLOCK_D // FP4_BLOCK_SIZE

    data_f32 = data.to(tl.float32)
    data_2d = tl.reshape(data_f32, (NUM_BLOCKS, FP4_BLOCK_SIZE))
    abs_2d = tl.abs(data_2d)
    block_absmax = tl.max(abs_2d, axis=1)

    ideal_scales = tl.where(block_absmax > 0.0, block_absmax / 6.0, 1e-30)
    fp8_scales = _float_to_fp8_e4m3(ideal_scales)
    float_scales = _fp8_e4m3_to_float(fp8_scales)
    float_scales = tl.where(float_scales > 0.0, float_scales, 1e-30)

    scaled_2d = data_2d / float_scales[:, None]
    scaled = tl.reshape(scaled_2d, (BLOCK_D,))
    nibbles = _fp4_quantize_nibbles(scaled)
    packed = _fp4_pack_nibbles(nibbles, BLOCK_D)
    return packed, fp8_scales


@triton.jit
def _fused_qk_rope_reshape_and_cache_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    offs_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    q_out_ptr,
    k_out_ptr,
    zeros_out_ptr,
    T,
    T_slot,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    cos_stride_t,
    cos_stride_d,
    q_out_stride_t,
    q_out_stride_h,
    q_out_stride_d,
    k_out_stride_t,
    k_out_stride_h,
    k_out_stride_d,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    key_cache_stride_x,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    zeros_out_stride_t,
    zeros_out_stride_h,
    zeros_out_stride_d,
    k_scale_ptr,
    v_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    FLASH_LAYOUT: tl.constexpr,
    HAVE_POS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
    HAVE_ZEROS: tl.constexpr = False,
    k_scale_stride_h=0,
    v_scale_stride_h=0,
):

    tl.assume(q_stride_t >= 0)
    tl.assume(q_stride_h >= 0)
    tl.assume(q_stride_d >= 0)
    tl.assume(k_stride_t >= 0)
    tl.assume(k_stride_h >= 0)
    tl.assume(k_stride_d >= 0)
    tl.assume(v_stride_t >= 0)
    tl.assume(v_stride_h >= 0)
    tl.assume(v_stride_d >= 0)
    tl.assume(cos_stride_t >= 0)
    tl.assume(cos_stride_d >= 0)
    tl.assume(q_out_stride_t >= 0)
    tl.assume(q_out_stride_h >= 0)
    tl.assume(q_out_stride_d >= 0)
    tl.assume(k_out_stride_t >= 0)
    tl.assume(k_out_stride_h >= 0)
    tl.assume(k_out_stride_d >= 0)
    tl.assume(key_cache_stride_t >= 0)
    tl.assume(key_cache_stride_h >= 0)
    tl.assume(key_cache_stride_d >= 0)
    tl.assume(key_cache_stride_b >= 0)
    tl.assume(key_cache_stride_x >= 0)
    tl.assume(value_cache_stride_t >= 0)
    tl.assume(value_cache_stride_h >= 0)
    tl.assume(value_cache_stride_d >= 0)
    tl.assume(value_cache_stride_b >= 0)
    tl.assume(zeros_out_stride_t >= 0)
    tl.assume(zeros_out_stride_h >= 0)
    tl.assume(zeros_out_stride_d >= 0)

    pid = tl.program_id(0)
    tl.assume(pid >= 0)

    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    if pid < T * QH:
        pid_t = pid // QH
        pid_hq = pid % QH
        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
                # d_cos_mask = d_cos_offs < BLOCK_D_pe
            else:
                d_cos_offs = d_pe_offs // 2
                # d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe
        else:
            d_cos_offs = d_pe_offs
            # d_cos_mask = d_cos_offs < BLOCK_D_pe

        pos = tl.load(pos_ptr + pid_t)
        if HAVE_POS:
            offset = tl.load(offs_ptr + pid_t)
            pos = pos + offset
        cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs)
        sin = tl.load(sin_ptr + cos_offs)

        q_ptrs = (
            q_ptr + pid_t * q_stride_t + pid_hq * q_stride_h + d_pe_offs * q_stride_d
        )
        q_pe = _unit_rope(
            q_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )
        q_out_ptrs = (
            q_out_ptr
            + pid_t * q_out_stride_t
            + pid_hq * q_out_stride_h
            + d_pe_offs * q_out_stride_d
        )
        tl.store(q_out_ptrs, q_pe.to(q_out_ptr.dtype.element_ty))

        if HAVE_ZEROS:
            z = tl.zeros((BLOCK_D_pe,), dtype=zeros_out_ptr.dtype.element_ty)
            zeros_out_ptrs = (
                zeros_out_ptr
                + pid_t * zeros_out_stride_t
                + pid_hq * zeros_out_stride_h
                + d_pe_offs * zeros_out_stride_d
            )
            tl.store(zeros_out_ptrs, z)

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
                pid_hk = pid_hq // QH_PER_KH
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )
                k_pe = _unit_rope(
                    k_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))

                if key_cache_ptr.dtype.element_ty == tl.uint8:
                    FP4_BLK: tl.constexpr = 32
                    NUM_K_BLOCKS: tl.constexpr = BLOCK_D_pe // FP4_BLK
                    k_packed, k_block_scales = _f32_to_fp4_e2m1_packed_block32(
                        k_pe, BLOCK_D_pe, FP4_BLK,
                        FP4_MSE_REFINE_ITERS=1)
                    # Store per-block K scales: layout [num_heads * num_k_blocks, total_tokens]
                    k_blk_stride = k_scale_stride_h // NUM_K_BLOCKS
                    k_blk_offs = tl.arange(0, NUM_K_BLOCKS).to(tl.int64)
                    k_scale_ptrs = (
                        k_scale_ptr
                        + pid_hk * k_scale_stride_h
                        + k_blk_offs * k_blk_stride
                        + pid_slot
                    )
                    tl.store(k_scale_ptrs, k_block_scales)
                    d_half_offs = tl.arange(0, BLOCK_D_pe // 2).to(tl.int64)
                    if FLASH_LAYOUT:
                        k_fp4_ptrs = (
                            key_cache_ptr
                            + pid_t_slot * key_cache_stride_t
                            + pid_b * key_cache_stride_b
                            + pid_hk * key_cache_stride_h
                            + d_half_offs * key_cache_stride_d
                        )
                    else:
                        k_packed_2d = tl.reshape(k_packed, (BLOCK_D_pe // 2 // X_SIZE, X_SIZE))
                        dx_offs = tl.arange(0, BLOCK_D_pe // 2 // X_SIZE).to(tl.int64)
                        x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                        k_fp4_ptrs = (
                            key_cache_ptr
                            + pid_t_slot * key_cache_stride_t
                            + pid_hk * key_cache_stride_h
                            + dx_offs[:, None] * key_cache_stride_d
                            + pid_b * key_cache_stride_b
                            + x_offs[None, :] * key_cache_stride_x
                        )
                        k_packed = k_packed_2d
                    tl.store(k_fp4_ptrs, k_packed)
                    v_ptrs = (
                        v_ptr
                        + pid_t * v_stride_t
                        + pid_hk * v_stride_h
                        + d_pe_offs * v_stride_d
                    )
                    v_data = tl.load(v_ptrs)
                    FP4_BLK_V: tl.constexpr = 32
                    NUM_V_BLOCKS: tl.constexpr = BLOCK_D_pe // FP4_BLK_V
                    v_packed, v_block_scales = _f32_to_fp4_e2m1_packed_block32(
                        v_data, BLOCK_D_pe, FP4_BLK_V,
                        FP4_MSE_REFINE_ITERS=1)
                    v_blk_stride = v_scale_stride_h // NUM_V_BLOCKS
                    v_blk_offs = tl.arange(0, NUM_V_BLOCKS).to(tl.int64)
                    v_scale_ptrs = (
                        v_scale_ptr
                        + pid_hk * v_scale_stride_h
                        + v_blk_offs * v_blk_stride
                        + pid_slot
                    )
                    tl.store(v_scale_ptrs, v_block_scales)
                    d_half_offs_v = tl.arange(0, BLOCK_D_pe // 2).to(tl.int64)
                    v_fp4_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_half_offs_v * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                    tl.store(v_fp4_ptrs, v_packed)
                else:
                    k_scale_rcprl = 1 / k_scale
                    k_pe = k_pe * k_scale_rcprl

                    if FLASH_LAYOUT:
                        k_out_ptrs = (
                            key_cache_ptr
                            + pid_t_slot * key_cache_stride_t
                            + pid_b * key_cache_stride_b
                            + pid_hk * key_cache_stride_h
                            + d_pe_offs * key_cache_stride_d
                        )
                    else:
                        k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                        dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                        x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                        k_out_ptrs = (
                            key_cache_ptr
                            + pid_t_slot * key_cache_stride_t
                            + pid_hk * key_cache_stride_h
                            + dx_offs[:, None] * key_cache_stride_d
                            + pid_b * key_cache_stride_b
                            + x_offs[None, :] * key_cache_stride_x
                        )

                    tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                    v_ptrs = (
                        v_ptr
                        + pid_t * v_stride_t
                        + pid_hk * v_stride_h
                        + d_pe_offs * v_stride_d
                    )
                    if HAVE_V_SCALE:
                        v_scale = tl.load(v_scale_ptr)
                    else:
                        v_scale = 1
                    v_scale_rcprl = 1 / v_scale
                    v = tl.load(v_ptrs) * v_scale_rcprl
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_pe_offs.to(tl.int64) * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                    tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))
    else:
        pid = pid - T * QH + T * KH
        if pid < T_slot * KH:
            pid_t = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_slot // BLOCK_SIZE
                pid_b = pid_slot % BLOCK_SIZE
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )

                k_pe = tl.load(k_ptrs)

                k_out_ptrs = (
                    k_out_ptr
                    + pid_t * k_out_stride_t
                    + pid_hk * k_out_stride_h
                    + d_pe_offs * k_out_stride_d
                )
                tl.store(k_out_ptrs, k_pe.to(k_out_ptr.dtype.element_ty))

                if key_cache_ptr.dtype.element_ty == tl.uint8:
                    FP4_BLK2: tl.constexpr = 32
                    NUM_K_BLOCKS2: tl.constexpr = BLOCK_D_pe // FP4_BLK2
                    k_packed, k_block_scales2 = _f32_to_fp4_e2m1_packed_block32(
                        k_pe, BLOCK_D_pe, FP4_BLK2)
                    k_blk_stride2 = k_scale_stride_h // NUM_K_BLOCKS2
                    k_blk_offs2 = tl.arange(0, NUM_K_BLOCKS2).to(tl.int64)
                    k_scale_ptrs2 = (
                        k_scale_ptr
                        + pid_hk * k_scale_stride_h
                        + k_blk_offs2 * k_blk_stride2
                        + pid_slot
                    )
                    tl.store(k_scale_ptrs2, k_block_scales2)
                    d_half_offs = tl.arange(0, BLOCK_D_pe // 2).to(tl.int64)
                    if FLASH_LAYOUT:
                        k_fp4_ptrs = (
                            key_cache_ptr
                            + pid_t_slot * key_cache_stride_t
                            + d_half_offs * key_cache_stride_d
                            + pid_b * key_cache_stride_b
                            + pid_hk * key_cache_stride_h
                        )
                    else:
                        k_packed_2d = tl.reshape(k_packed, (BLOCK_D_pe // 2 // X_SIZE, X_SIZE))
                        dx_offs = tl.arange(0, BLOCK_D_pe // 2 // X_SIZE).to(tl.int64)
                        x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                        k_fp4_ptrs = (
                            key_cache_ptr
                            + pid_t_slot * key_cache_stride_t
                            + pid_hk * key_cache_stride_h
                            + dx_offs[:, None] * key_cache_stride_d
                            + pid_b * key_cache_stride_b
                            + x_offs[None, :] * key_cache_stride_x
                        )
                        k_packed = k_packed_2d
                    tl.store(k_fp4_ptrs, k_packed)
                    v_ptrs = (
                        v_ptr
                        + pid_t * v_stride_t
                        + pid_hk * v_stride_h
                        + d_pe_offs * v_stride_d
                    )
                    v_data = tl.load(v_ptrs)
                    FP4_BLK_V2: tl.constexpr = 32
                    NUM_V_BLOCKS2: tl.constexpr = BLOCK_D_pe // FP4_BLK_V2
                    v_packed, v_block_scales2 = _f32_to_fp4_e2m1_packed_block32(
                        v_data, BLOCK_D_pe, FP4_BLK_V2)
                    v_blk_stride2 = v_scale_stride_h // NUM_V_BLOCKS2
                    v_blk_offs2 = tl.arange(0, NUM_V_BLOCKS2).to(tl.int64)
                    v_scale_ptrs2 = (
                        v_scale_ptr
                        + pid_hk * v_scale_stride_h
                        + v_blk_offs2 * v_blk_stride2
                        + pid_slot
                    )
                    tl.store(v_scale_ptrs2, v_block_scales2)
                    d_half_offs_v = tl.arange(0, BLOCK_D_pe // 2).to(tl.int64)
                    v_fp4_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_half_offs_v * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                    tl.store(v_fp4_ptrs, v_packed)
                else:
                    k_scale_rcprl = 1 / k_scale
                    k_pe = k_pe * k_scale_rcprl

                    if FLASH_LAYOUT:
                        k_out_ptrs = (
                            key_cache_ptr
                            + pid_t_slot * key_cache_stride_t
                            + d_pe_offs * key_cache_stride_d
                            + pid_b * key_cache_stride_b
                            + pid_hk * key_cache_stride_h
                        )
                    else:
                        k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                        dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                        x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                        k_out_ptrs = (
                            key_cache_ptr
                            + pid_t_slot * key_cache_stride_t
                            + pid_hk * key_cache_stride_h
                            + dx_offs[:, None] * key_cache_stride_d
                            + pid_b * key_cache_stride_b
                            + x_offs[None, :] * key_cache_stride_x
                        )
                    tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                    v_ptrs = (
                        v_ptr
                        + pid_t * v_stride_t
                        + pid_hk * v_stride_h
                        + d_pe_offs * v_stride_d
                    )
                    if HAVE_V_SCALE:
                        v_scale = tl.load(v_scale_ptr)
                    else:
                        v_scale = 1
                    v_scale_rcprl = 1 / v_scale
                    v = tl.load(v_ptrs) * v_scale_rcprl
                    v_out_ptrs = (
                        value_cache_ptr
                        + pid_t_slot * value_cache_stride_t
                        + pid_hk * value_cache_stride_h
                        + d_pe_offs * value_cache_stride_d
                        + pid_b * value_cache_stride_b
                    )
                    tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))


@triton.jit
def _fused_qk_rope_cosine_cache_llama_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    pos_ptr,
    cos_ptr,
    sin_ptr,
    offs_ptr,
    key_cache_ptr,
    value_cache_ptr,
    slot_mapping_ptr,
    q_out_ptr,
    T,
    T_slot,
    q_stride_t,
    q_stride_h,
    q_stride_d,
    k_stride_t,
    k_stride_h,
    k_stride_d,
    v_stride_t,
    v_stride_h,
    v_stride_d,
    cos_stride_t,
    cos_stride_d,
    q_out_stride_t,
    q_out_stride_h,
    q_out_stride_d,
    key_cache_stride_t,
    key_cache_stride_h,
    key_cache_stride_d,
    key_cache_stride_b,
    key_cache_stride_x,
    value_cache_stride_t,
    value_cache_stride_h,
    value_cache_stride_d,
    value_cache_stride_b,
    k_scale_ptr,
    v_scale_ptr,
    QH_PER_KH: tl.constexpr,
    QH: tl.constexpr,
    KH: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    X_SIZE: tl.constexpr,
    FLASH_LAYOUT: tl.constexpr,
    HAVE_POS: tl.constexpr = False,
    HAVE_K_SCALE: tl.constexpr = False,
    HAVE_V_SCALE: tl.constexpr = False,
):
    pid = tl.program_id(0)

    d_pe_offs = tl.arange(0, BLOCK_D_pe).to(tl.int64)

    if pid < T * QH:
        pid_t = pid // QH
        pid_hq = pid % QH
        if REUSE_FREQS_FRONT_PART:
            if IS_NEOX:
                d_cos_offs = d_pe_offs
                d_cos_offs = tl.where(
                    (d_cos_offs >= BLOCK_D_HALF_pe) & (d_cos_offs < BLOCK_D_pe),
                    d_cos_offs - BLOCK_D_HALF_pe,
                    d_cos_offs,
                ).to(d_cos_offs.dtype)
            else:
                d_cos_offs = d_pe_offs // 2
                d_cos_mask = d_cos_offs < BLOCK_D_HALF_pe

        else:
            d_cos_offs = d_pe_offs

        pos = tl.load(pos_ptr + pid_t)
        if HAVE_POS:
            offset = tl.load(offs_ptr + pid_t)
            pos = pos + offset
        cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs).to(tl.float64)
        sin = tl.load(sin_ptr + cos_offs).to(tl.float64)

        q_ptrs = (
            q_ptr + pid_t * q_stride_t + pid_hq * q_stride_h + d_pe_offs * q_stride_d
        )
        q_pe = _unit_rope(
            q_ptrs,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_D_pe,
            BLOCK_D_HALF_pe,
        )
        q_out_ptrs = (
            q_out_ptr
            + pid_t * q_out_stride_t
            + pid_hq * q_out_stride_h
            + d_pe_offs * q_out_stride_d
        )
        tl.store(q_out_ptrs, q_pe.to(q_out_ptr.dtype.element_ty))

        if pid_hq % QH_PER_KH == 0:
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_t
                pid_b = pid_slot
                pid_hk = pid_hq // QH_PER_KH
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )
                k_pe = _unit_rope(
                    k_ptrs,
                    cos,
                    sin,
                    d_pe_offs,
                    IS_NEOX,
                    BLOCK_D_pe,
                    BLOCK_D_HALF_pe,
                )

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                        + d_pe_offs * key_cache_stride_d
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )

                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                v_out_ptrs = (
                    value_cache_ptr
                    + pid_t_slot * value_cache_stride_t
                    + pid_hk * value_cache_stride_h
                    + d_pe_offs * value_cache_stride_d
                    + pid_b * value_cache_stride_b
                )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))
    else:
        pid = pid - T * QH + T * KH
        if pid < T_slot * KH:
            pid_t = pid // KH
            pid_hk = pid % KH
            pid_slot = tl.load(slot_mapping_ptr + pid_t).to(tl.int64)
            if pid_slot >= 0:
                pid_t_slot = pid_t
                pid_b = pid_slot
                if HAVE_K_SCALE:
                    k_scale = tl.load(k_scale_ptr)
                else:
                    k_scale = 1
                k_ptrs = (
                    k_ptr
                    + pid_t * k_stride_t
                    + pid_hk * k_stride_h
                    + d_pe_offs * k_stride_d
                )

                k_pe = tl.load(k_ptrs)

                k_scale_rcprl = 1 / k_scale
                k_pe = k_pe * k_scale_rcprl

                if FLASH_LAYOUT:
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + d_pe_offs * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + pid_hk * key_cache_stride_h
                    )
                else:
                    k_pe = tl.reshape(k_pe, (BLOCK_D_pe // X_SIZE, X_SIZE))
                    dx_offs = tl.arange(0, BLOCK_D_pe // X_SIZE).to(tl.int64)
                    x_offs = tl.arange(0, X_SIZE).to(tl.int64)
                    k_out_ptrs = (
                        key_cache_ptr
                        + pid_t_slot * key_cache_stride_t
                        + pid_hk * key_cache_stride_h
                        + dx_offs[:, None] * key_cache_stride_d
                        + pid_b * key_cache_stride_b
                        + x_offs[None, :] * key_cache_stride_x
                    )
                tl.store(k_out_ptrs, k_pe.to(key_cache_ptr.dtype.element_ty))

                v_ptrs = (
                    v_ptr
                    + pid_t * v_stride_t
                    + pid_hk * v_stride_h
                    + d_pe_offs * v_stride_d
                )
                if HAVE_V_SCALE:
                    v_scale = tl.load(v_scale_ptr)
                else:
                    v_scale = 1
                v_scale_rcprl = 1 / v_scale
                v = tl.load(v_ptrs) * v_scale_rcprl
                v_out_ptrs = (
                    value_cache_ptr
                    + pid_t_slot * value_cache_stride_t
                    + pid_hk * value_cache_stride_h
                    + d_pe_offs * value_cache_stride_d
                    + pid_b * value_cache_stride_b
                )
                tl.store(v_out_ptrs, v.to(value_cache_ptr.dtype.element_ty))
