"""Build private JobSets from the verified B template; never commit outputs."""
import argparse, ast, copy, hashlib, json
from pathlib import Path

FACTORS = {
    'c-base2': (True, False, False),
    'd-base2-scale': (True, True, False),
    'e-pv': (False, False, True),
    'f-base2-pv': (True, False, True),
}
BASE_TOKAMAX = '34992492c7f366da1286966d64fc69046f347773'
BASE_JOB = 'plus-vl-fa4-ab-1011-b-fa4'


def independent_patcher(source):
    tree = ast.parse(source)
    changed = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
            continue
        s = node.value
        if 'sa_config = sa_config.with_gpu_fa4_forward_numerics' not in s:
            continue
        s = s.replace(
            '          sa_config = sa_config.with_gpu_fa4_forward_numerics(softmax_scale=self.qk_logits_scale)',
            "          import dataclasses\n"
            "          mode = os.environ['PLUS_VL_FA4_FACTOR_MODE']\n"
            "          factors = " + repr(FACTORS) + "[mode]\n"
            "          sa_config = dataclasses.replace(sa_config,\n"
            "              gpu_fa4_forward_numerics=False, use_base2_exp=factors[0],\n"
            "              combine_log2_scale=factors[1], pv_bf16_rounding=factors[2],\n"
            "              softmax_scale=self.qk_logits_scale, max_logit_const=None,\n"
            "              attn_logits_soft_cap=None)")
        s = s.replace("'gpu_fa4_forward_numerics', 'use_base2_exp'", "'gpu_fa4_forward_numerics', 'pv_bf16_rounding', 'use_base2_exp'")
        s = s.replace("assert actual['gpu_fa4_forward_numerics'] == use_vit_fa4", "assert actual['gpu_fa4_forward_numerics'] is False")
        s = s.replace("assert actual['use_base2_exp'] and actual['combine_log2_scale']", "assert (actual['use_base2_exp'], actual['combine_log2_scale'], actual['pv_bf16_rounding']) == factors")
        s = s.replace("actual['guarded_fixed_shift'] = self.vit_guarded_fixed_shift", "actual['guarded_fixed_shift'] = self.vit_guarded_fixed_shift\n          actual['factor_mode'] = mode")
        node.value = s
        changed += 1
    assert changed == 1
    out = ast.unparse(tree) + '\n'
    ast.parse(out)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--template', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tokamax-commit', required=True)
    args = parser.parse_args()
    raw = args.template.read_text(encoding='utf-8')
    assert BASE_TOKAMAX in raw and BASE_JOB in raw
    template = json.loads(raw)
    patcher = independent_patcher(template['items'][0]['data']['apply_fa4.py'])
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = dict(steps=1000, gpu_baseline=344670243,
        source_commit='cfd3e6369cc922ede65608a7c5b6dc898dd481e6',
        pallas_commit='8ef7b3b6d33b881a771af6312d2f794e8a848e9d',
        tokamax_parent=BASE_TOKAMAX, tokamax_commit=args.tokamax_commit,
        cp=16, sequence_length=262144, dtype='bfloat16', ns='eager',
        formula='100*(TPU-GPU)/GPU; signed step mean; keep raw spike windows',
        reused_controls={'a-control': (False,False,False), 'b-fa4': (True,True,True)},
        windows=[[1,3],[120,126],[654,659]], arms=[])
    maps=[]
    for mode, factors in FACTORS.items():
        name='plus-vl-fa4-factors-1012-'+mode
        bundle=json.loads(raw.replace(BASE_JOB,name).replace(BASE_TOKAMAX,args.tokamax_commit))
        cm,job=bundle['items']
        cm['data']['apply_fa4.py']=patcher
        job['metadata']['annotations']['ablation.factor_mode']=mode
        job['metadata']['annotations']['ablation.factors']=json.dumps(factors)
        job['metadata']['annotations']['ablation.fa4']='independent'
        job['metadata']['annotations']['ablation.arm']=mode
        container=job['spec']['replicatedJobs'][0]['template']['spec']['template']['spec']['containers'][0]
        assert next(e['value'] for e in container['env'] if e['name']=='PLUS_VL_FA4_ABLATION')=='1'
        container['env'].append({'name':'PLUS_VL_FA4_FACTOR_MODE','value':mode})
        for key,value in cm['data'].items():
            if key.endswith('.py'): ast.parse(value,filename=key)
        assert 'steps=1000' in cm['data']['train.sh']
        assert manifest['pallas_commit'] in cm['data']['train.sh']
        assert args.tokamax_commit in cm['data']['train.sh']
        maps.append(cm['data'])
        path=args.output/(name+'-private.json')
        path.write_text(json.dumps(bundle,indent=2),encoding='utf-8')
        manifest['arms'].append(dict(jobname=name, mode=mode, factors=factors,
            chips=256, status='prepared', bundle_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert all(m == maps[0] for m in maps)
    (args.output/'apply_fa4_factors.py').write_text(patcher,encoding='utf-8')
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))

if __name__ == '__main__': main()
