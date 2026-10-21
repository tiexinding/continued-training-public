"""本地重判分（不推理）：读取 humaneval_*.json 中保存的 outs，用两种拼接规则重新执行。
规则 v2（原）：含 def entry_point → header+body；否则 prompt+body（不缩进）。
规则 v3（自动缩进）：同 v2，但当 body 不含 def entry_point 且首个非空行无缩进时，整体缩进 4 空格。
输出每臂两种规则的 passed 与错误类型。
注意：本脚本以子进程直接执行模型生成的程序，10 秒超时不是隔离措施；请在隔离环境（容器或虚拟机）中运行。"""
import json,re,sys,subprocess,tempfile,os,numpy as np,math,collections
from pathlib import Path
rows=[json.loads(l) for l in open(sys.argv[1])]; H=json.load(open(sys.argv[2])); out=sys.argv[3]
def build(code,r,auto):
    m=re.search(r"```(?:python)?\n(.*?)```",code,re.S); body=m.group(1) if m else code
    header="\n".join(l for l in r['prompt'].split("\n") if l.startswith("import ") or l.startswith("from "))+"\nfrom typing import *\nimport math\n"
    if ("def "+r['entry_point']) in body: prog=header+body
    else:
        if auto:
            first=next((l for l in body.split("\n") if l.strip()),"")
            if first and not first.startswith((" ","\t")): body="\n".join(("    "+l if l.strip() else l) for l in body.split("\n"))
        prog=r['prompt']+body
    return prog+"\n\n"+r['test']+f"\n\ncheck({r['entry_point']})\n"
def run(prog):
    with tempfile.NamedTemporaryFile("w",suffix=".py",delete=False) as f: f.write(prog); fn=f.name
    try: pr=subprocess.run([sys.executable,fn],capture_output=True,timeout=10); return pr.returncode==0, pr.stderr.decode(errors="ignore")[-200:]
    except subprocess.TimeoutExpired: return False,"timeout"
    finally: Path(fn).unlink(missing_ok=True)
res={}
for arm in H:
    res[arm]={}
    for rule,auto in (("v2",False),("v3_autoindent",True)):
        rr=[run(build(o,r,auto)) for o,r in zip(H[arm]['outs'],rows)]
        res[arm][rule]=dict(passed=[x[0] for x in rr],errors=[x[1] for x in rr],pass_rate=float(np.mean([x[0] for x in rr])),
            syntax=float(np.mean(['SyntaxError' in e for _,e in rr])),assertion=float(np.mean(['AssertionError' in e for _,e in rr])))
    print(arm,{k:(round(v['pass_rate'],3),'syn',round(v['syntax'],2)) for k,v in res[arm].items()},'原 json',round(H[arm]['pass_rate'],3))
json.dump(res,open(out,'w'))
