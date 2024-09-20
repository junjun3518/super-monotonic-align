import torch

import triton
import triton.language as tl

@triton.jit
def maximum_path(
    path, value, text_lengths, audio_lengths,
    stride_B, stride_T, stride_S,
    max_neg_val,
    BLOCK_SIZE_T: tl.constexpr
    ):
    batch = tl.program_id(axis=0)
    path += batch * stride_B
    value += batch * stride_B
    text_length = tl.load(text_lengths + batch)
    audio_length = tl.load(audio_lengths + batch)
    offs_prev = tl.arange(0, BLOCK_SIZE_T)
    init = tl.where(offs_prev ==0, tl.load(value), max_neg_val)
    # for j in range(0,1,1):  # set the first column to max_neg_val without init point
    tl.store(value + offs_prev * stride_T, init, mask=offs_prev < text_length)
    for j in range(1, audio_length, 1):
        v_cur= tl.load(value + (offs_prev) * stride_T + (j-1) * stride_S, 
                       mask=(offs_prev < text_length), other=max_neg_val)
        v_prev =tl.load(value + (offs_prev-1) * stride_T + (j-1) * stride_S,
                        mask=(0 < offs_prev) & (offs_prev < text_length), other=max_neg_val)
        # compare v_cur and v_prev, and update v with larger value
        v = (tl.maximum(v_cur, v_prev) + tl.load(value + (offs_prev) * stride_T + j * stride_S,
                                                 mask=(offs_prev < text_length)))
        tl.store(value + (offs_prev) * stride_T + j * stride_S, v,
                 mask=(offs_prev < text_length))

    index = text_length-1
    for j in range(audio_length-1,-1,-1):
        tl.store(path + (index) * stride_T + j * stride_S, 1)
        if (index > 0): # (index == j) is not checked due to max_neg_val init
            v_left = tl.load(value+ (index) * stride_T + (j-1) * stride_S)#.to(tl.float32)
            v_leftdown =  tl.load(value+(index-1) * stride_T + (j-1) * stride_S)#.to(tl.float32)
            if (v_left < v_leftdown):
                index += - 1
            
                        
@torch.no_grad()
def maximum_path_triton(path, value, x_lengths, y_lengths, max_neg_val=-1e32, audio_last=True):
    if audio_last:
        B,T,S = path.shape
        stride_B = T*S
        stride_T = S
        stride_S = 1
        text_lengths, audio_lengths = x_lengths, y_lengths
    else:
        B,S,T = path.shape
        stride_B = T*S
        stride_S = T
        stride_T = 1
        audio_lengths, text_lengths = x_lengths, y_lengths

    BLOCK_SIZE_T = max(triton.next_power_of_2(T), 16)
    num_warps = 1 # Need to be 1 to prevent wrong output by slicing the operation
    with torch.cuda.device(value.device.index):
        maximum_path[(B, )](
            path, value, text_lengths, audio_lengths, 
            stride_B, stride_T, stride_S,
            max_neg_val = max_neg_val,
            num_warps = num_warps,
            BLOCK_SIZE_T = BLOCK_SIZE_T)
    return path

@triton.jit
def maximum_path_old(
    path, value, t_x, t_y,
    B, T, S,
    max_neg_val,
    BLOCK_SIZE_X: tl.constexpr
    ):
    batch = tl.program_id(axis=0)
    path += batch * T * S
    value += batch * T * S
    x_length = tl.load(t_x + batch)
    y_length = tl.load(t_y + batch)
    offs_prev = tl.arange(0, BLOCK_SIZE_X)
    init = tl.where(offs_prev ==0, tl.load(value), max_neg_val)
    # for j in range(0,1,1):  # set the first column to max_neg_val without init point
    tl.store(value + offs_prev * S, init, mask=offs_prev < x_length)
    for j in range(1, y_length, 1):
        v_cur= tl.load(value + (offs_prev) * S + (j-1), mask=(offs_prev < x_length), other=max_neg_val)
        v_prev =tl.load(value + (offs_prev-1) * S + (j-1), mask=(0 < offs_prev) & (offs_prev < x_length), other=max_neg_val)
        # compare v_cur and v_prev, and update v with larger value
        v = (tl.maximum(v_cur, v_prev) + tl.load(value + (offs_prev) * S + j, mask=(offs_prev < x_length)))
        tl.store(value + (offs_prev) * S + j, v, mask=(offs_prev < x_length))

    index = x_length-1
    for j in range(y_length-1,-1,-1):
        tl.store(path + (index) * S + j, 1)
        if (index > 0): # (index == j) is not checked due to max_neg_val init
            v_left = tl.load(value+ (index) * S+ j-1)#.to(tl.float32)
            v_leftdown =  tl.load(value+(index-1) * S + j-1)#.to(tl.float32)
            if (v_left < v_leftdown):
                index += - 1
            
                        
@torch.no_grad()
def maximum_path_triton_old(path, value, t_x, t_y, max_neg_val=-1e32):
    B,T,S = path.shape
    BLOCK_SIZE_X = max(triton.next_power_of_2(T), 16)
    num_warps = 1 # Need to be 1 to prevent wrong output by slicing the operation
    with torch.cuda.device(value.device.index):
        maximum_path_old[(B, )](
            path, value, t_x, t_y, 
            B, T, S,
            max_neg_val = max_neg_val,
            num_warps = num_warps,
            BLOCK_SIZE_X = BLOCK_SIZE_X)
    return path
