
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import random
from typing import List, Optional, Tuple, Union
import itertools
import torch
import pytest
from aiter.test_common import checkAllclose, perftest, tensor_dump, tensor_load
from aiter import pertoken_quant
from aiter import dtypes

from enum import Enum
from einops import rearrange
import argparse
import pandas as pd
import aiter

uniform_range = (-1, 1)
STR_DTYPE_TO_TORCH_DTYPE = {
    "half": torch.half,
    "bfloat16": dtypes.bf16,
    "float": dtypes.fp32,
    "fp8": torch.uint8,
    "fp8_e4m3": torch.uint8,
    "fp8_e5m2": torch.uint8,
}


def get_kv_cache_torch_dtype(
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
) -> torch.dtype:
    if isinstance(cache_dtype, str):
        if cache_dtype == "auto":
            if isinstance(model_dtype, str):
                torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[model_dtype]
            elif isinstance(model_dtype, torch.dtype):
                torch_dtype = model_dtype
            else:
                raise ValueError(f"Invalid model dtype: {model_dtype}")
        elif cache_dtype in ["half", "bfloat16", "float"]:
            torch_dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]
        elif cache_dtype == "fp8":
            torch_dtype = torch.uint8
        else:
            raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    elif isinstance(cache_dtype, torch.dtype):
        torch_dtype = cache_dtype
    else:
        raise ValueError(f"Invalid kv cache dtype: {cache_dtype}")
    return torch_dtype


def kv_cache_factory(
    num_blocks: int,
    block_size: int,
    num_layers: int,
    num_heads: int,
    head_size: int,
    cache_dtype: Optional[Union[str, torch.dtype]],
    model_dtype: Optional[Union[str, torch.dtype]] = None,
    seed: int = 0,
    device: Optional[str] = "cuda",
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:

    if cache_dtype == "fp8" and head_size % 16:
        raise ValueError(
            f"Does not support key cache of type fp8 with head_size {head_size}"
        )

    torch_dtype = get_kv_cache_torch_dtype(cache_dtype, model_dtype)

    x = 16 // torch_dtype.itemsize
    key_cache_shape = (num_blocks, num_heads, head_size // x, block_size, x)
    key_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        key_cache = torch.empty(size=key_cache_shape, dtype=torch_dtype, device=device)
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            key_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support key cache of type {cache_dtype}")
        key_caches.append(key_cache)

    value_cache_shape = (num_blocks, num_heads, head_size, block_size)
    value_caches: List[torch.Tensor] = []
    for _ in range(num_layers):
        value_cache = torch.empty(
            size=value_cache_shape, dtype=torch_dtype, device=device
        )
        if cache_dtype in ["auto", "half", "bfloat16", "float"]:
            value_cache.uniform_(*uniform_range)
        else:
            raise ValueError(f"Does not support value cache of type {cache_dtype}")
        value_caches.append(value_cache)
    return key_caches, value_caches


FLOAT32_BYTES = torch.finfo(dtypes.fp32).bits // 8
# This will change depending on the compute capability.
# - 512 as a buffer
MAX_SEQ_LEN = 65536
# There may not be enough gpu memory due to large NUM_BLOCKS.
# Reduce NUM_BLOCKS when it happens.
NUM_BLOCKS = 32768  # Arbitrary values for testing
PARTITION_SIZE = 512
# flshattF and tritonflashattF supported: {dtypes.fp16, dtypes.bf16}
DTYPES = [torch.half, dtypes.bf16]
NUM_GEN_SEQS = [7]  # Arbitrary values for testing
NUM_PREFILL_SEQS = [3]  # Arbitrary values for testing
NUM_HEADS = [(40, 40), (64, 8)]  # Arbitrary values for testing

# FlashAttention forward only supports head dimension at most 128
# https://github.com/ROCmSoftwarePlatform/flash-attention/blob/3d2b6f5d037782cc2c906909a46fb7e2e1b48b25/csrc/flash_attn_rocm/flash_api.cpp#L62
HEAD_SIZES = [64, 80, 96, 112, 120, 128, 192, 256]

BLOCK_SIZES = [16, 32]
USE_ALIBI = [False, True]
KV_CACHE_DTYPE = ["auto", "fp8"]
SEEDS = [0]
CUDA_DEVICES = [f"cuda:{i}" for i in range(1 if torch.cuda.device_count() == 1 else 2)]

# 0: no quant. 1: (ignore this), FP8, 2: K/V per-token(prefer this)
PA_QUANT = 2


def ref_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    attn_mask: Optional[torch.Tensor] = None,
    logits_soft_cap: float = 0.0,
    sliding_window: int = 0,
) -> torch.Tensor:
    attn_weights = scale * torch.einsum("qhd,khd->hqk", query, key).float()
    if attn_mask is not None:
        attn_weights = attn_weights + attn_mask.float()
    if sliding_window:
        attn_weights[:, :, :-sliding_window] = -1e38
    if 0 < logits_soft_cap:
        attn_weights = logits_soft_cap * torch.tanh(attn_weights / logits_soft_cap)
    attn_weights = torch.softmax(attn_weights, dim=-1).to(value.dtype)
    out = torch.einsum("hqk,khd->qhd", attn_weights, value)
    return out


def pertoken_quant_kvcache_symm(
    # [num_blocks, num_heads, head_size // x, block_size, x]
    key_cache: torch.Tensor,
    # [num_blocks, num_heads, head_size, block_size]
    value_cache: torch.Tensor,
    quant_dtype: torch.dtype,  # e.g. dtypes.fp8
    scale_dtype: torch.dtype = dtypes.fp32,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_blocks = key_cache.shape[0]
    num_heads = key_cache.shape[1]
    head_dim = value_cache.shape[2]
    block_size = value_cache.shape[3]
    # x          = key_cache.shape[4]
    total_tokens = num_blocks * block_size

    # print(f"{key_cache.shape=}{key_cache.stride()=}")
    # print(f"{value_cache.shape=}{value_cache.stride()=}")

    key_cache_permute = (
        key_cache.permute(0, 1, 3, 2, 4)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )
    value_cache_permute = (
        value_cache.permute(0, 1, 3, 2)
        .reshape(num_blocks, num_heads, block_size, -1)
        .contiguous()
    )

    k_quant, k_scale = pertoken_quant(key_cache_permute, quant_dtype=quant_dtype)
    v_quant, v_scale = pertoken_quant(value_cache_permute, quant_dtype=quant_dtype)

    # NOTE: quant_x and original x could be different
    quant_x = 16 // quant_dtype.itemsize

    k_quant = (
        k_quant.view(num_blocks, num_heads, block_size, head_dim // quant_x, quant_x)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    k_scale = k_scale.permute(1, 0, 2, 3).reshape(num_heads, total_tokens).contiguous()
    v_quant = (
        v_quant.view(num_blocks, num_heads, block_size, head_dim)
        .permute(0, 1, 3, 2)
        .contiguous()
    )
    v_scale = v_scale.permute(1, 0, 2, 3).reshape(num_heads, total_tokens).contiguous()

    # print(f"{k_quant.shape=}{k_quant.stride()=}")
    # print(f"{k_scale.shape=}{k_scale.stride()=}")
    # print(f"{v_quant.shape=}{v_quant.stride()=}")
    # print(f"{v_scale.shape=}{v_scale.stride()=}")
    # print(f"key_cache_permute:{key_cache_permute[0, :, :, :]}, k_quant:{k_quant[0, :, :, :, :]}, k_scale:{k_scale[:, 0]}")

    return k_quant, k_scale, v_quant, v_scale


# @perftest()  # Disabled: too slow for large batch sizes
def run_torch(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    kv_cache_dtype,
    num_kv_heads,
    scale,
    alibi_slopes,
    logits_soft_cap,
    k_scale,
    v_scale,
    num_queries_per_kv,
    sliding_window,
):
    output = torch.zeros_like(query)
    num_query_heads = query.shape[1]
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[2]
    head_size = key_cache.shape[3]
    num_seqs = query.shape[0]

    block_tables_lst = block_tables.cpu().tolist()
    seq_lens_lst = seq_lens.cpu().tolist()
    for i in range(num_seqs):
        q = query[i].unsqueeze(0)
        block_table = block_tables_lst[i]
        seq_len = int(seq_lens_lst[i])

        keys_lst: List[torch.Tensor] = []
        values_lst: List[torch.Tensor] = []
        for j in range(seq_len):
            block_number = int(block_table[j // block_size])
            block_offset = j % block_size

            k = key_cache[block_number, :, block_offset, :]
            k = k.reshape(num_kv_heads, head_size)
            keys_lst.append(k)

            v = value_cache[block_number, :, block_offset, :]
            values_lst.append(v)
        keys = torch.stack(keys_lst, dim=0)
        values = torch.stack(values_lst, dim=0)
        if num_queries_per_kv > 1:
            # Handle MQA and GQA
            keys = torch.repeat_interleave(keys, num_queries_per_kv, dim=1)
            values = torch.repeat_interleave(values, num_queries_per_kv, dim=1)

        alibi_bias = None
        if alibi_slopes is not None:
            # Create the ALiBi bias used in the paged attention kernel.
            position_ids = torch.arange(seq_len).int()
            alibi_bias = (position_ids - seq_len + 1).float()
            alibi_bias = alibi_slopes.view(-1, 1, 1) * alibi_bias.view(1, 1, -1)

        out = ref_masked_attention(
            q,
            keys,
            values,
            scale,
            alibi_bias,
            logits_soft_cap,
            sliding_window=sliding_window,
        )
        out = out.view(num_query_heads, head_size)
        output[i].copy_(out, non_blocking=True)
    return output, 0  # Return dummy time (not used)


@perftest()
def run_aiter(
    query,
    key_cache,
    value_cache,
    block_tables,
    cu_query_lens,
    seq_lens,
    max_seq_len,
    kv_cache_dtype,
    kv_cache_layout,
    scale,
    alibi_slopes,
    logits_soft_cap,
    k_scale,
    v_scale,
    mtp=1,
    sliding_window=0,
    k_scale_stride_h=0,
    v_scale_stride_h=0,
):
    # copied from ops.PagedAttention.forward_decode()
    _PARTITION_SIZE_ROCM = 256
    fp8_out_scale = None

    num_seqs, num_heads, head_size = query.shape
    block_size = key_cache.shape[2 if kv_cache_layout == "HND" else 1]

    output = torch.empty_like(query)
    max_num_partitions = (
        max_seq_len + _PARTITION_SIZE_ROCM - 1
    ) // _PARTITION_SIZE_ROCM
    assert _PARTITION_SIZE_ROCM % block_size == 0

    # will use single workspace buffer to accommodate following 3 intermediate tensors:
    #   1. tmp_output (shape=(num_seqs, num_heads, max_num_partitions, head_size), dtype=output.dtype)
    #   2. exp_sums (shape=(num_seqs, num_heads, max_num_partitions), dtype=float32)
    #   3. max_logits (shape=(num_seqs, num_heads, max_num_partitions), dtype=float32)
    nbyes_per_qo_elem = torch.finfo(output.dtype).bits // 8
    workspace_buffer = torch.empty(
        (num_seqs * mtp * num_heads * max_num_partitions * head_size)
        * nbyes_per_qo_elem
        + 2 * (num_seqs * mtp * num_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=output.device,
    )

    cpa_fp8_out = False
    if fp8_out_scale is not None:
        output = torch.empty_like(output, dtype=dtypes.fp8)
        cpa_fp8_out = True
    torch.ops.aiter.paged_attention_v1(
        output,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        scale,
        block_tables,
        cu_query_lens,
        seq_lens,
        max_seq_len,
        alibi_slopes,
        kv_cache_dtype,
        kv_cache_layout,
        logits_soft_cap,
        k_scale,
        v_scale,
        fp8_out_scale if cpa_fp8_out else None,
        _PARTITION_SIZE_ROCM,
        sliding_window=sliding_window,
        k_scale_stride_h=k_scale_stride_h,
        v_scale_stride_h=v_scale_stride_h,
    )
    if cpa_fp8_out:
        return output.view(num_seqs, num_heads * head_size)
    else:
        return output


def dump_input(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    kv_cache_dtype: str,
    num_kv_heads: int,
    scale: float,
    alibi_slopes: Optional[torch.Tensor],
    k_scale: float,
    v_scale: float,
):
    tensor_dump(query, "Q")
    # qbk = tensor_load('Q.bin')
    # checkAllclose(query, qbk)
    tensor_dump(key_cache, "K_cache")
    tensor_dump(value_cache, "V_cache")
    tensor_dump(block_tables, "block_tables")
    tensor_dump(seq_lens, "seq_lens")


def load_input():
    # return (tensor_load('Q.bin'),
    #         tensor_load('K_cache.bin'),
    #         tensor_load('V_cache.bin'),
    #         tensor_load('block_tables.bin'),
    #         tensor_load('seq_lens.bin'),
    #         tensor_load('out_aiter.bin'))
    # return (tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/Q_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/K_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/V_16.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/block_tables.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/seq_lens.bin'),
    #         tensor_load('/mnt/raid0/ljin1/pa_data/x8_Kzero/OUT_16.bin'),
    #         )
    return (
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/Q_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/K_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/V_BF16.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/block_tables.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/seq_lens.bin"),
        tensor_load("/mnt/raid0/ljin1/pa_data/bf16in/OUT_BF16.bin"),
    )


def asm_V_shuffle(VC):
    # [num_blocks, num_kv_heads, head_size, block_size]
    x = 16 // VC.element_size()
    num_blocks, num_kv_heads, head_size, block_size = VC.shape
    VC = VC.view(num_blocks, num_kv_heads, head_size, block_size // x, x)
    # [num_blocks, num_kv_heads, block_size/X, head_size, X]
    VC = VC.permute(0, 1, 3, 2, 4).contiguous()
    return VC


class InputSource(Enum):
    PreGen = 1
    Random = 2


class PAVariant(Enum):
    Shomy = 1
    Asm = 2
    Naive = 3


INPUT_SOURCE = InputSource.Random
DUMP_INPUTS = False  # whether to dump inputs
DUMP_OUTPUT = False  # whether to dump output


@pytest.mark.parametrize("ctx_lens", [8192, 16384, 22000])
@pytest.mark.parametrize("num_seqs", [512])
@pytest.mark.parametrize("num_heads", [(128, 8)])
@pytest.mark.parametrize("head_size", [128])
@pytest.mark.parametrize("use_alibi", [False])
@pytest.mark.parametrize("block_size", [16])
@pytest.mark.parametrize("dtype", [dtypes.bf16])
@pytest.mark.parametrize("kv_cache_dtype", ["fp8", "fp4"])
@pytest.mark.parametrize("kv_cache_layout", ["NHD", "HND"])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 30.0])
@pytest.mark.parametrize("pa_variant", [PAVariant.Shomy])
@pytest.mark.parametrize("quant_cache_dtype", [None, dtypes.fp8, dtypes.i8])
@pytest.mark.parametrize("seed", [0])
@pytest.mark.parametrize("device", ["cuda:0"])
def test_paged_attention(
    ctx_lens: int,
    num_seqs: int,
    num_heads: Tuple[int, int],
    head_size: int,
    use_alibi: bool,
    block_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str,
    kv_cache_layout: str,
    logits_soft_cap: float,
    pa_variant: PAVariant,
    quant_cache_dtype: torch.dtype,
    seed: int,
    device: str,
    sliding_window: int = 0,
) -> None:
    if pa_variant == PAVariant.Shomy:
        if quant_cache_dtype is not None:
            pytest.skip()
    elif pa_variant == PAVariant.Asm:
        if (
            use_alibi
            or head_size != 128
            or block_size != 16
            or dtype is not dtypes.bf16
            or quant_cache_dtype not in [None, dtypes.i8]
            or sliding_window != 0
        ):
            pytest.skip()
    elif pa_variant == PAVariant.Naive:
        if use_alibi:
            pytest.skip()

    torch.manual_seed(seed)
    random.seed(seed)
    torch.set_default_device(device)

    # Using default kv_scale
    k_scale = v_scale = torch.tensor([1.0], dtype=dtypes.fp32)
    scale = float(1.0 / (head_size**0.5))
    num_query_heads, num_kv_heads = num_heads
    alibi_slopes = None
    if use_alibi:
        alibi_slopes = torch.randn(num_query_heads, dtype=dtypes.fp32)
    assert num_query_heads % num_kv_heads == 0
    num_queries_per_kv = num_query_heads // num_kv_heads
    max_seq_len = ctx_lens
    max_num_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
    num_blocks = max_num_blocks_per_seq * num_seqs
    print(f"{INPUT_SOURCE=}")

    # prepare inputs & golden output
    if INPUT_SOURCE == InputSource.PreGen:
        query, key_cache, value_cache, block_tables, seq_lens, out_golden = load_input()
    else:
        query = torch.empty(num_seqs, num_query_heads, head_size, dtype=dtype)
        query.uniform_(*uniform_range)

        cache_create_dtype = "auto" if kv_cache_dtype in ["fp8", "fp4"] else kv_cache_dtype
        key_caches, value_caches = kv_cache_factory(
            num_blocks,
            block_size,
            1,
            num_kv_heads,
            head_size,
            cache_create_dtype,
            dtype,
            seed,
            device,
        )

        key_cache, value_cache = key_caches[0], value_caches[0]
        
        key_cache_quant, value_cache_quant = None, None
        k_scale_quant, v_scale_quant = None, None
        if kv_cache_dtype == "fp8":
            key_cache_quant, k_scale_quant, value_cache_quant, v_scale_quant = pertoken_quant_kvcache_symm(
                key_cache, value_cache, quant_dtype=dtypes.fp8
            )
            print(f"  [FP8] K quant shape: {key_cache_quant.shape}, V quant shape: {value_cache_quant.shape}")
            print(f"  [FP8] K scale shape: {k_scale_quant.shape}, V scale shape: {v_scale_quant.shape}")
        elif kv_cache_dtype == "fp4":
            from aiter.ops.triton.fused_kv_cache import fused_qk_rope_reshape_and_cache
            nb = key_cache.shape[0]
            nh = key_cache.shape[1]
            hs = value_cache.shape[2]
            bs_cache = value_cache.shape[3]
            total_tok = nb * bs_cache
            head_bytes = hs // 2
            x_fp4 = 16

            key_flat = rearrange(key_cache, "b h d1 s d2 -> (b s) h (d1 d2)").contiguous()
            value_flat = rearrange(value_cache, "b h d s -> (b s) h d").contiguous()

            key_cache_quant = torch.zeros(
                nb, nh, head_bytes // x_fp4, bs_cache, x_fp4,
                dtype=torch.uint8, device=device
            ).view(dtype=torch.float4_e2m1fn_x2)
            value_cache_quant = torch.zeros(
                nb, nh, head_bytes, bs_cache,
                dtype=torch.uint8, device=device
            ).view(dtype=torch.float4_e2m1fn_x2)

            k_scale_quant = torch.zeros(nh, total_tok, dtype=torch.float32, device=device)
            v_scale_quant = torch.zeros(nh, total_tok, dtype=torch.float32, device=device)

            slot_mapping = torch.arange(total_tok, dtype=torch.int64, device=device)

            cos_freqs = torch.ones(1, hs // 2, dtype=dtype, device=device)
            sin_freqs = torch.zeros(1, hs // 2, dtype=dtype, device=device)
            positions = torch.zeros(total_tok, dtype=torch.int64, device=device)

            chunk_size = 4096
            for start in range(0, total_tok, chunk_size):
                end = min(start + chunk_size, total_tok)
                sz = end - start
                q_chunk = torch.zeros(sz, nh, hs, dtype=dtype, device=device)
                fused_qk_rope_reshape_and_cache(
                    q_chunk, key_flat[start:end], value_flat[start:end],
                    key_cache_quant, value_cache_quant,
                    slot_mapping[start:end], positions[start:end],
                    cos_freqs, sin_freqs,
                    k_scale_quant, v_scale_quant,
                    is_neox=True, flash_layout=False,
                    apply_scale=False, output_zeros=False,
                )
            print(f"  [FP4-Triton] K quant shape: {key_cache_quant.shape}, V quant shape: {value_cache_quant.shape}")
            print(f"  [FP4-Triton] K scale shape: {k_scale_quant.shape}, V scale shape: {v_scale_quant.shape}")

        # Create the block tables.
        block_tables = rearrange(
            torch.randperm(num_blocks, dtype=dtypes.i32, device=device),
            "(b nblocks) -> b nblocks",
            b=num_seqs,
        )

        # seq_lens = [random.randint(1, MAX_SEQ_LEN) for _ in range(num_seqs)]
        seq_lens = torch.full(size=(num_seqs,), fill_value=ctx_lens, dtype=torch.int)

        key_cache_new = rearrange(key_cache, "b h d1 s d2 -> b h s (d1 d2)")
        value_cache_new = rearrange(value_cache, "b h d s -> b h s d")
        out_golden, _ = run_torch(
            query,
            key_cache_new,
            value_cache_new,
            block_tables,
            seq_lens,
            max_seq_len,
            kv_cache_dtype,
            num_kv_heads,
            scale,
            alibi_slopes,
            logits_soft_cap,
            k_scale,
            v_scale,
            num_queries_per_kv,
            sliding_window,
        )
        cu_query_lens = torch.arange(0, num_seqs + 1, dtype=torch.int)

    time_aiter = None
    acc_str = "N/A"
    if quant_cache_dtype is None:
        if kv_cache_dtype == "fp8":
            key_cache_run = rearrange(key_cache_quant, "b h d1 s d2 -> b h s (d1 d2)")
            value_cache_run = rearrange(value_cache_quant, "b h d s -> b h s d")
            
            if kv_cache_layout == "NHD":
                key_cache_run = rearrange(key_cache_run, "b h s d -> b s h d")
                value_cache_run = rearrange(value_cache_run, "b h s d -> b s h d")
            
            num_kv_heads = k_scale_quant.shape[0]
            current_total_tokens = k_scale_quant.shape[1]

            k_scale_run = k_scale_quant.contiguous()
            v_scale_run = v_scale_quant.contiguous()
            kv_dtype_str = "fp8"
        elif kv_cache_dtype == "fp4":
            key_cache_u8 = key_cache_quant.view(dtype=torch.uint8)
            value_cache_u8 = value_cache_quant.view(dtype=torch.uint8)

            key_cache_run = rearrange(key_cache_u8, "b h d1 s d2 -> b h s (d1 d2)")
            value_cache_run = rearrange(value_cache_u8, "b h d s -> b h s d")

            if kv_cache_layout == "NHD":
                key_cache_run = rearrange(key_cache_run, "b h s d -> b s h d")
                value_cache_run = rearrange(value_cache_run, "b h s d -> b s h d")

            num_kv_heads = k_scale_quant.shape[0]
            current_total_tokens = k_scale_quant.shape[1]

            k_scale_run = k_scale_quant.contiguous()
            v_scale_run = v_scale_quant.contiguous()
            kv_dtype_str = "fp4"
        else:
            # Use original BF16/FP16 cache
            if kv_cache_layout == "NHD":
                key_cache_run = rearrange(key_cache_new, "b h s d -> b s h d")
                value_cache_run = rearrange(value_cache_new, "b h s d -> b s h d")
            else:
                key_cache_run = key_cache_new
                value_cache_run = value_cache_new
            k_scale_run = k_scale
            v_scale_run = v_scale
            kv_dtype_str = kv_cache_dtype

        k_scale_stride_h = current_total_tokens if kv_cache_dtype in ("fp8", "fp4") else 0
        v_scale_stride_h = current_total_tokens if kv_cache_dtype in ("fp8", "fp4") else 0
        out_aiter, time_aiter = run_aiter(
            query,
            key_cache_run.contiguous(),
            value_cache_run.contiguous(),
            block_tables,
            cu_query_lens,
            seq_lens,
            max_seq_len,
            kv_dtype_str,
            kv_cache_layout,
            scale,
            alibi_slopes,
            logits_soft_cap,
            k_scale_run,
            v_scale_run,
            sliding_window=sliding_window,
            k_scale_stride_h=k_scale_stride_h,
            v_scale_stride_h=v_scale_stride_h,
        )
        # Set atol/rtol based on kv_cache_dtype
        if kv_cache_dtype == "fp8":
            atol, rtol = 0.01, 0.01
        elif kv_cache_dtype == "fp4":
            atol, rtol = 0.05, 0.05
        else:
            atol, rtol = 0.1, 0.1  # BF16/FP16
        acc = checkAllclose(out_golden, out_aiter, atol=atol, rtol=rtol, msg=f"golden vs aiter:{time_aiter:.2f} us")
        acc_str = "PASS" if acc < 0.01 else "FAIL"

    if DUMP_INPUTS:
        dump_input(
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            max_seq_len,
            kv_cache_dtype,
            num_kv_heads,
            scale,
            alibi_slopes,
            k_scale,
            v_scale,
        )

    # Calculate bandwidth (GB/s)
    dtype_size = dtype.itemsize
    if kv_cache_dtype == "fp8":
        cache_dtype_size = 1  # 1 byte per element
    elif kv_cache_dtype == "fp4":
        cache_dtype_size = 0.5  # 0.5 bytes per element (packed)
    else:
        cache_dtype_size = dtype_size  # BF16/FP16
    
    q_bytes = num_seqs * num_query_heads * head_size * dtype_size
    kv_bytes = 2 * num_seqs * ctx_lens * num_kv_heads * head_size * cache_dtype_size
    block_table_bytes = num_seqs * max_num_blocks_per_seq * 4
    output_bytes = num_seqs * num_query_heads * head_size * dtype_size
    
    total_bytes = q_bytes + kv_bytes + block_table_bytes + output_bytes
    total_gb = total_bytes / (1024 ** 3)
    
    bw = total_gb / (time_aiter * 1e-6) if time_aiter and time_aiter > 0 else 0
    
    print(f"  [BW] Data: {total_bytes/1e6:.2f} MB, BW: {bw:.2f} GB/s, Acc: {acc_str}")
    
    return {
        "time_us": round(time_aiter, 2) if time_aiter else None,
        "BW_GBs": round(bw, 2),
        "Acc": acc_str,
    }


@pytest.mark.parametrize("ctx_lens", [1, 26, 128, 4097])
@pytest.mark.parametrize("num_seqs", [1, 3, 31, 128])
@pytest.mark.parametrize("num_heads", [(8, 1), (32, 4)])
@pytest.mark.parametrize("use_alibi", [False, True])
@pytest.mark.parametrize("sliding_window", [0, 10])
def test_paged_attention_sliding_window(
    ctx_lens: int,
    num_seqs: int,
    num_heads: Tuple[int, int],
    use_alibi: bool,
    sliding_window: int,
) -> None:
    test_paged_attention(
        ctx_lens,
        num_seqs,
        num_heads,
        128,
        use_alibi,
        block_size=16,
        dtype=dtypes.fp16,
        kv_cache_dtype="auto",
        kv_cache_layout="NHD",
        logits_soft_cap=0.0,
        pa_variant=PAVariant.Shomy,
        quant_cache_dtype=None,
        seed=0,
        device="cuda:3",
        sliding_window=sliding_window,
    )


if __name__ == "__main__":
    # ============================================================================
    # LLaMA 3.1 405B Baseline Test Configurations
    # Full model: num_heads=128, num_kv_heads=8, head_dim=128
    # ============================================================================
    
    LLAMA_NUM_HEADS = (128, 8)  # (num_query_heads, num_kv_heads)
    LLAMA_HEAD_SIZE = 128
    LLAMA_BLOCK_SIZE = 16
    
    # Test cases: num_seqs=512, ctx_lens=[8192, 16384, 22000]
    CASES = [
        {"ctx_len": 8192,  "bs": 512},
        {"ctx_len": 16384, "bs": 512},
        {"ctx_len": 22000, "bs": 512},
    ]
    
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="LLaMA 3.1 405B PagedAttention Baseline Test (Reference)",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=["bf16", "fp16"],
        default="bf16",
        help="""Data type (default: bf16).
    e.g.: -d bf16""",
    )
    parser.add_argument(
        "--kv_cache_dtype",
        type=str,
        choices=["auto", "fp8", "fp4"],
        default="fp8",
        help="""KV cache dtype (default: fp8 to match model trace).
    e.g.: --kv_cache_dtype fp8
          --kv_cache_dtype fp4  (experimental: MXFP4 E2M1)""",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="""CUDA device (default: cuda:0).
    e.g.: --device cuda:1""",
    )

    torch.set_printoptions(sci_mode=False)
    args = parser.parse_args()
    
    # Setup dtype
    if args.dtype == "bf16":
        dtype = dtypes.bf16
    else:
        dtype = dtypes.fp16
    
    print("=" * 80)
    print("LLaMA 3.1 405B PagedAttention Baseline Test (Reference Implementation)")
    print(f"Model config: num_heads={LLAMA_NUM_HEADS}, head_size={LLAMA_HEAD_SIZE}, block_size={LLAMA_BLOCK_SIZE}")
    print(f"dtype: {args.dtype}, kv_cache_dtype: {args.kv_cache_dtype}")
    print("=" * 80)
    
    df = []
    
    for case in CASES:
        ctx_len = case["ctx_len"]
        bs = case["bs"]
        
        print(f"\n{'='*60}")
        print(f"ctx_len={ctx_len}, bs={bs}")
        print(f"{'='*60}")
        
        ret = test_paged_attention(
            ctx_len,
            bs,
            LLAMA_NUM_HEADS,
            LLAMA_HEAD_SIZE,
            False,  # use_alibi
            LLAMA_BLOCK_SIZE,
            dtype,
            args.kv_cache_dtype,
            "NHD",
            0.0, 
            PAVariant.Shomy,
            None,
            0,
            args.device,
            0,
        )
        ret["ctx_len"] = ctx_len
        ret["bs"] = bs
        ret["dtype"] = args.dtype
        ret["kv_dtype"] = args.kv_cache_dtype
        df.append(ret)
    
    df = pd.DataFrame(df)
    # Reorder columns for better readability
    col_order = ["ctx_len", "bs", "dtype", "kv_dtype", "time_us", "BW_GBs", "Acc"]
    df = df[[c for c in col_order if c in df.columns]]
    
    # Rename columns for clarity
    df = df.rename(columns={
        "time_us": "time(us)",
        "BW_GBs": "BW(GB/s)",
    })
    
    print("\n" + "=" * 80)
    aiter.logger.info(f"LLaMA 3.1 405B Baseline Summary (Reference):\n{df.to_string(index=False)}")
    print("=" * 80)

