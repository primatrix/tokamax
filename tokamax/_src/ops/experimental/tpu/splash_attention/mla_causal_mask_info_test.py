import numpy as np
import pytest
from tokamax._src.ops.experimental.tpu.splash_attention import mla_causal_mask_info as fast
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask as masks
from tokamax._src.ops.experimental.tpu.splash_attention import splash_attention_mask_info as native


@pytest.mark.parametrize("sequence,bq,bk",[(512,128,256),(1024,512,128),(1024,128,512),(2048,256,256),(8192,4096,8192)])
@pytest.mark.parametrize("dkv",[False,True])
@pytest.mark.parametrize("dynamic",[False,True])
@pytest.mark.parametrize("downcast",[False,True])
def test_matches_every_native_field_and_dtype(sequence,bq,bk,dkv,dynamic,downcast):
  mask=masks.CausalMask((sequence,sequence))
  kwargs=dict(downcast_smem_data=downcast,return_dynamic_grid=dynamic)
  expected,fn=native._process_mask(mask,(bq,bk),dkv,**kwargs)
  actual,other=fast.process_mask(mask,(bq,bk),dkv,**kwargs)
  assert fn is other
  for a,b in zip(expected,actual):
    if a is None:assert b is None
    else:
      assert a.dtype==b.dtype
      np.testing.assert_array_equal(a,b)


def test_offset_falls_back_to_native():
  mask=masks.CausalMask((512,512),offset=4)
  a,fn=native._process_mask(mask,(128,256),False)
  b,other=fast.process_mask(mask,(128,256))
  for x,y in zip(a,b):
    if x is None:assert y is None
    else:np.testing.assert_array_equal(x,y)
