"""Exact unsharded causal metadata without examining S squared mask entries."""
import functools
import numpy as np
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as masks
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask_info as native


@functools.lru_cache(maxsize=12)
def process_mask(mask,block_shape,is_dkv=False,*,downcast_smem_data=True,
                 partial_mask_blocks_dtype=np.int8,q_seq_shards=1,kv_seq_shards=1,
                 return_dynamic_grid=True):
  q,k=mask.shape;bq,bk=block_shape
  if (type(mask) is not masks.CausalMask or mask.offset!=0 or q!=k
      or q_seq_shards!=1 or kv_seq_shards!=1 or q%bq or k%bk):
    return native._process_mask(mask,block_shape,is_dkv,
        downcast_smem_data=downcast_smem_data,partial_mask_blocks_dtype=partial_mask_blocks_dtype,
        q_seq_shards=q_seq_shards,kv_seq_shards=kv_seq_shards,return_dynamic_grid=return_dynamic_grid)
  first_q=np.arange(q//bq,dtype=np.int64)[:,None]*bq
  first_k=np.arange(k//bk,dtype=np.int64)[None,:]*bk
  state=np.where(first_q+bq-1<first_k,0,np.where(first_q>=first_k+bk-1,2,1)).astype(np.int32)
  if np.all(state==2):
    return native._process_mask(mask,block_shape,is_dkv,downcast_smem_data=downcast_smem_data,
        partial_mask_blocks_dtype=partial_mask_blocks_dtype,return_dynamic_grid=return_dynamic_grid)
  # A computable causal mask never reads partial block payloads or mask_next.
  # Reuse native schedule generation; only the unused partial IDs differ.
  partial=np.where(state==1,0,-1).astype(np.int32)
  rows,cols,_,blocks,count=native._generate_shard_metadata(state,partial,is_dkv,return_dynamic_grid)
  if return_dynamic_grid:
    if downcast_smem_data:
      rows=native._downcast_to_small_type(rows);cols=native._downcast_to_small_type(cols)
    count=np.array([count],dtype=np.int32)
  else:
    rows=cols=count=None
  if downcast_smem_data:blocks=native._downcast_to_small_type(blocks)
  return native.MaskInfo(mask_next=None,active_rows=rows,active_cols=cols,block_mask=blocks,
      num_active_blocks=count,partial_mask_blocks=None,q_sequence=mask.q_sequence),mask.mask_function
