"""Baseline adapter: execute frozen MaxText's Ulysses source, not a new collective.

Only the training model shell (projection/Flax/config resolution) is omitted.
The seq/head sharding-constraint statements are extracted verbatim. Splash is
the frozen public Tokamax kernel with save_residuals=True, batch-vmapped inside
the same head-sharded shard_map used by AttentionOp.
"""
import ast
import hashlib
from pathlib import Path
import textwrap
from types import SimpleNamespace


def build(config, mesh, source_path):
    import jax
    import jax.numpy as jnp
    from attention_core import build as build_core
    cp, axis = config['cp_size'], config['mesh_axis']
    n, heads = config['tokens'], config['global_heads']
    if cp < 1 or heads % cp or n % cp:
        raise ValueError('cp must divide global heads and sequence length')
    if mesh.shape[axis] != cp:
        raise ValueError('cp_size must equal mesh axis size')
    for key in ('block_q','block_kv'):
        if (n // cp) % config['splash'][key]:
            raise ValueError(f'local sequence must be divisible by {key}')
    assert config['heads'] == heads // cp
    P = jax.sharding.PartitionSpec
    source = Path(source_path).read_text()
    tree = ast.parse(source)
    candidates = [x for x in ast.walk(tree) if isinstance(x, ast.If)
                  and ast.unparse(x.test) == 'using_a2a_cp']
    pre = next(x for x in candidates if 'qk_d' in ast.get_source_segment(source,x))
    post = next(x for x in candidates if 'seq_pspec =' in ast.get_source_segment(source,x))
    # Context axis name in the training source is a mesh spelling, not an
    # algorithm constant. Resolve it through the user parameter in the AST.
    class AxisName(ast.NodeTransformer):
        def visit_Constant(self,node):
            return ast.copy_location(ast.Constant(axis),node) if node.value == 'context' else node
    def extract(node, signature, prefix, suffix):
        body = textwrap.dedent(''.join(source.splitlines(True)[node.lineno:node.end_lineno]))
        code = 'def extracted('+signature+'):\n' + textwrap.indent(prefix+body+suffix,'  ')
        parsed = ast.fix_missing_locations(AxisName().visit(ast.parse(code)))
        ns = dict(jax=jax,jnp=jnp,BATCH='batch',
                  logical_to_mesh_axes=lambda *args: (None,))
        exec(compile(parsed,str(source_path),'exec'),ns)
        return ns['extracted'],dict(start_line=node.lineno,end_line=node.end_lineno,
             source=ast.get_source_segment(source,node),sha256=hashlib.sha256(body.encode()).hexdigest())
    pre_fn, pre_evidence = extract(pre,'query,key,value,decoder_segment_ids,self',
        'a2a_segment_ids=decoder_segment_ids\na2a_bidirectional_mask=None\n',
        'return query,key,value,a2a_segment_ids\n')
    post_fn, post_evidence = extract(post,'out,self','batch_axes_q=None\n','return out\n')
    adapter = SimpleNamespace(mesh=mesh,config=SimpleNamespace(force_q_layout=False,logical_axis_rules=()))
    core, _, metadata = build_core(config)
    seq = jax.sharding.NamedSharding(mesh,P(None,axis,None,None))
    # User's measurement boundary includes Q/K/V and O exchanges only. IDs are
    # supplied already replicated, so the training ID constraint is a no-op.
    seg = jax.sharding.NamedSharding(mesh,P(None,None))
    hspec = P(None,axis,None,None)
    statspec = P(None,axis,None)
    @jax.shard_map(mesh=mesh,in_specs=(hspec,hspec,hspec,P(None,None)),
                   out_specs=(hspec,{'logsumexp':statspec,'max_logits':statspec}),check_vma=False)
    def mapped(q,k,v,ids):
        return jax.vmap(core)(q,k,v,ids,ids)
    def whole(q,k,v,ids):
        with jax.named_scope('MLA_PRE_A2A'):
            q,k,v,ids = pre_fn(q,k,v,ids,adapter)
        with jax.named_scope('MLA_SPLASH'):
            q,k,v = [jnp.transpose(x,(0,2,1,3)) for x in (q,k,v)]
            out,stats=mapped(q,k,v,ids)
            out=jnp.transpose(out,(0,2,1,3))
        with jax.named_scope('MLA_POST_A2A'):
            out=post_fn(out,adapter)
        return out,stats
    shapes=[jax.ShapeDtypeStruct((1,n,heads,d),jnp.bfloat16) for d in
            (config['qk_head_dim'],config['qk_head_dim'],config['v_head_dim'])]
    shapes.append(jax.ShapeDtypeStruct((1,n),jnp.int32))
    output_sharding=(seq,{'logsumexp':jax.sharding.NamedSharding(mesh,statspec),
                         'max_logits':jax.sharding.NamedSharding(mesh,statspec)})
    metadata['training_ulysses'] = dict(source_path=str(source_path),
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),pre=pre_evidence,post=post_evidence,
        force_q_layout=False,axis_name_substitution=axis,
        attention_op_lines='attention_op.py:1370-1374,1411-1483,1749-1777,1916-1932,2049-2052',
        residual_a2a=False,residual_note='Public stats select lane 0; internal Pallas ABI remains FP32 [H_local,N,128]. Stats remain head sharded.',
        batch_axis_resolution='B=1, data/fsdp axes of size 1 removed; context is the only nontrivial mesh axis')
    return whole,shapes,(seq,seq,seq,seg),output_sharding,metadata
