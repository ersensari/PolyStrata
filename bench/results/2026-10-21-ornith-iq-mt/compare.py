"""One otherwise-idle GPU; baseline and opt-in alternate from identical expert profiles."""
import argparse,hashlib,json,os,re,shutil,subprocess,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'tools'))
from strata_tokenizer import Tokenizer
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--model',required=True)
ap.add_argument('--baseline',required=True)
ap.add_argument('--engine',required=True)
ap.add_argument('--tokenizer',required=True)
ap.add_argument('--profile',required=True)
ap.add_argument('--out',required=True)
a=ap.parse_args()
out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
tok=Tokenizer.from_dir(a.tokenizer)
model=a.model;base=a.baseline;new=a.engine
prompts=[
 'Write a complete Python LRU cache class with get and put methods, type hints, docstrings and five unit tests. Explain its time complexity.',
 'Explain Python asyncio tasks, cancellation and timeouts in Turkish. Include runnable examples and common mistakes.',
 'Implement a Python parser for a simple arithmetic language with + - * / parentheses, with a tokenizer, recursive descent parser, error handling and tests.']
rows=[]
def run(tag,exe,on=False,text=prompts[0],draft=True,freeze=False,max_new=256):
 profile=out/'active-profile.txt';shutil.copyfile(a.profile,profile)
 ids=tok.encode('<|im_start|>user\n'+text+'<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n',parse_special=True)
 env=dict(os.environ,STRATA_FATTN='1',STRATA_FDEC='1')
 for k in ['STRATA_POLY_HOST_KV','STRATA_POLY_IQ_MT','STRATA_FREEZE_EXPERTS']:env.pop(k,None)
 if on:env['STRATA_POLY_IQ_MT']='1'
 if freeze:env['STRATA_FREEZE_EXPERTS']='1'
 cmd=[exe,'--model',model,'--tokens',','.join(map(str,ids)),'--max-new',str(max_new),'--max-context','262144','--kv','q8_0','--threads','12','--vram-reserve-mib','1024','--expert-profile',str(profile),'--expert-swaps','0','--tail-skip','0','--temperature','0','--window','8','--lookup','3']
 if not draft:cmd+=['--no-draft','--window','1','--lookup','0']
 telemetry=open(out/(tag+'.gpu.csv'),'w')
 mon=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,utilization.gpu,power.draw,memory.used','--format=csv,noheader,nounits','-lms','200'],stdout=telemetry,stderr=subprocess.DEVNULL)
 try:p=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=600)
 finally:mon.terminate();mon.wait();telemetry.close()
 log=p.stderr.replace(model,'<MODEL>').replace(str(profile),'<PROFILE>')
 (out/(tag+'.txt')).write_text(log);(out/(tag+'.ids')).write_text(p.stdout)
 m=re.search(r'prompt (\d+) tokens ([\d.]+) tok/s, generated (\d+) tokens ([\d.]+) tok/s.*?experts from VRAM ([\d.]+)%, CPU experts ([\d.]+) ms/token, router wait ([\d.]+) ms/token.*?(\d+) of (\d+) guessed tokens right',log)
 row=dict(tag=tag,on=on,exit=p.returncode,input_tokens=len(ids),output_sha256=hashlib.sha256(p.stdout.encode()).hexdigest())
 if m:row.update(prompt_n=int(m[1]),prompt_tps=float(m[2]),generated_n=int(m[3]),decode_tps=float(m[4]),expert_hit_pct=float(m[5]),cpu_ms_per_token=float(m[6]),router_ms_per_token=float(m[7]),accepted=int(m[8]),offered=int(m[9]))
 rows.append(row);(out/'results.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
 if p.returncode or not m:raise RuntimeError('engine failed '+tag)
 return p.stdout
# Warm the OS file cache before recording the default-path check.
run('warmup',base,max_new=32)
a=run('default-reference',base,draft=False,freeze=True)
b=run('default-patched',new,draft=False,freeze=True)
assert a==b,'unset path changed'
for r in range(2):
 for i,prompt in enumerate(prompts):
  for on in ([False,True] if (r+i)%2==0 else [True,False]):
   run(f'p{i}-r{r}-'+('on' if on else 'off'),new,on=on,text=prompt)
long='Reference words:\n'+'river stone market lantern copper winter garden signal harbor needle orbit velvet '*600+'\n\n'+prompts[0]
for on in [False,True]:run('long-'+('on' if on else 'off'),new,on=on,text=long)
