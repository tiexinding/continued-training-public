#!/usr/bin/env python3
"""混合状态评估（卡上运行）：配置 = 底状态 + 从另一状态换入的参数组。C1–C4 共用。

配置文件（JSON 列表），每项 {"name": ..., "base": "A"|"B"|"W0", "swap_from": "A"|"B"|null, "groups": ["L23", "down", ...]}
或插值对照 {"name": ..., "interp": α}：W = A + α(B − A)
组名与 state_swap.groups_of 相同（L00–L23 / q k v o gate up down / vocab / final_norm）。

--nll：三任务探针（各 100 条）与留出集（GSM8K 测试集 200 条、Dolly closed_qa 未用样本 200 条；MBPP 已全部用于训练/探针，无留出）
       逐样本 (Σ token NLL, token 数)，以及按 token 类别拆分：
         GSM8K：answer（最后的 "#### 数" 行）/ calc（<<…>> 计算标注）/ reasoning（其余）
         MBPP ：ws（纯空白 token）/ sig（首行函数签名）/ body（其余）
         Dolly：all
--gen ：贪心生成。GSM8K 留出 200 条：strict（含 "#### 正确数"）与 relaxed（最后一个数正确）；
        MBPP 探针 100 条：生成代码 + 测试用例子进程执行（10 s 超时），报 pass@1。
输出 <out>/<name>.json
"""
import argparse, json, random, re, subprocess, sys, tempfile
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import forklib as fl  # noqa: E402
from fork_continue import load_model  # noqa: E402
from state_swap import groups_of  # noqa: E402

NUM = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")
HASH = re.compile(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)")


def norm_num(s):
    s = s.replace(",", "").strip().rstrip(".")
    try:
        from decimal import Decimal, InvalidOperation
        x = Decimal(s); return str(int(x)) if x == x.to_integral_value() else str(x.normalize())
    except Exception:
        return s


def write_rows(rows, path):
    with open(path, "w") as f:
        for r in rows: f.write(json.dumps(r, ensure_ascii=False) + "\n")


def heldout_rows(data_dir: Path, n: int, seed=20261004):
    from datasets import load_dataset
    gs = load_dataset("openai/gsm8k", "main", split="test")
    idx = list(range(len(gs))); random.Random(seed).shuffle(idx)
    g = [{"example_id": f"gsm8k_test_{i}", "messages": [{"role": "user", "content": gs[i]["question"]}, {"role": "assistant", "content": gs[i]["answer"]}],
          "metadata": {"source": "openai/gsm8k/main:test", "source_index": i}} for i in idx[:n]]
    used = set()
    for part in ("train", "probe"):
        for line in open(data_dir / "dolly_qa" / f"{part}.jsonl"):
            used.add(json.loads(line)["metadata"]["source_index"])
    dl = load_dataset("databricks/databricks-dolly-15k", split="train")
    cand = [i for i in range(len(dl)) if dl[i]["category"] == "closed_qa" and i not in used]
    random.Random(seed).shuffle(cand); d = []
    for i in cand[:n]:
        parts = [str(dl[i]["instruction"])]; ctx = str(dl[i].get("context") or "").strip()
        if ctx: parts.extend(["", ctx])
        d.append({"example_id": f"dolly_heldout_{i}", "messages": [{"role": "user", "content": "\n".join(parts)}, {"role": "assistant", "content": str(dl[i]["response"])}],
                  "metadata": {"source": "dolly closed_qa unused", "source_index": i}})
    return g, d


def token_categories(task, tok, comp_ids, text):
    """完成部分每个 token 的类别。用逐 token 解码累计长度定位字符位置。"""
    cats, pos = [], 0
    spans_calc = [(m.start(), m.end()) for m in re.finditer(r"<<.*?>>", text)] if task == "gsm8k" else []
    ans_start = text.rfind("####") if task == "gsm8k" else -1
    first_nl = text.find("\n") if task == "mbpp" else -1
    prev = ""
    for i in range(len(comp_ids)):
        cur = tok.decode(comp_ids[: i + 1]); piece = cur[len(prev):]; start = len(prev); prev = cur
        if task == "gsm8k":
            if ans_start >= 0 and start >= ans_start: c = "answer"
            elif any(a <= start < b for a, b in spans_calc): c = "calc"
            else: c = "reasoning"
        elif task == "mbpp":
            if piece.strip() == "": c = "ws"
            elif first_nl < 0 or start < first_nl: c = "sig"
            else: c = "body"
        else:
            c = "all"
        cats.append(c)
    return cats


@torch.no_grad()
def nll_eval(model, tok, task, exs, rows):
    dev = next(model.parameters()).device; out = {"s": [], "c": [], "cat": {}}
    for e, r in zip(exs, rows):
        ids = torch.tensor([e.input_ids], device=dev); m = torch.tensor(e.response_mask, dtype=torch.bool)
        pos = m.nonzero().flatten(); pos = pos[pos > 0]
        logits = model(input_ids=ids).logits[0, (pos - 1).to(dev)].float()
        nl = F.cross_entropy(logits, ids[0, pos.to(dev)], reduction="none").double().cpu()
        out["s"].append(float(nl.sum())); out["c"].append(int(nl.numel()))
        comp_ids = [e.input_ids[p] for p in pos.tolist()]
        cats = token_categories(task, tok, comp_ids, r["messages"][-1]["content"])
        per = {}
        for c, v in zip(cats, nl.tolist()):
            a = per.setdefault(c, [0.0, 0]); a[0] += v; a[1] += 1
        for c in set(list(per) + list(out["cat"])):
            out["cat"].setdefault(c, {"s": [], "c": []})
        for c in out["cat"]:
            a = per.get(c, [0.0, 0]); out["cat"][c]["s"].append(a[0]); out["cat"][c]["c"].append(a[1])
    # 补齐此前样本中未出现的类别（长度对齐）
    n = len(out["s"])
    for c, v in out["cat"].items():
        if len(v["s"]) < n:
            pad = n - len(v["s"]); v["s"] = [0.0] * pad + v["s"]; v["c"] = [0] * pad + v["c"]
    return out


@torch.no_grad()
def generate(model, tok, exs, max_new, bs=16):
    dev = next(model.parameters()).device; outs = []
    im_end = tok.convert_tokens_to_ids("<|im_end|>"); eos = [t for t in {tok.eos_token_id, im_end} if t is not None]
    for i in range(0, len(exs), bs):
        batch = exs[i:i + bs]
        prompts = [e.input_ids[: e.response_mask.index(1)] if 1 in e.response_mask else e.input_ids for e in batch]
        L = max(len(p) for p in prompts); pad = tok.pad_token_id if tok.pad_token_id is not None else eos[0]
        ids = torch.tensor([[pad] * (L - len(p)) + p for p in prompts], device=dev)
        att = torch.tensor([[0] * (L - len(p)) + [1] * len(p) for p in prompts], device=dev)
        g = model.generate(input_ids=ids, attention_mask=att, max_new_tokens=max_new, do_sample=False, eos_token_id=eos, pad_token_id=pad)
        for row in g[:, L:].tolist():
            outs.append(tok.decode([t for t in row if t not in eos and t != pad], skip_special_tokens=True))
    return outs


def run_mbpp(code, row):
    m = re.search(r"```(?:python)?\n(.*?)```", code, re.S); code = m.group(1) if m else code
    user = row["messages"][0]["content"]; tests = [l for l in user.split("\n") if l.strip().startswith("assert")]
    setup = user.split("Test setup:")[1].strip() if "Test setup:" in user else ""
    prog = code + "\n\n" + setup + "\n" + "\n".join(tests) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(prog); fn = f.name
    try:
        r = subprocess.run([sys.executable, fn], capture_output=True, timeout=10); return r.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    finally:
        Path(fn).unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct"); ap.add_argument("--tokenizer", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--stateA", required=True); ap.add_argument("--stateB", required=True); ap.add_argument("--data", required=True)
    ap.add_argument("--configs", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--nll", action="store_true"); ap.add_argument("--gen", action="store_true")
    ap.add_argument("--n_heldout", type=int, default=200); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--max_len", type=int, default=512)
    args = ap.parse_args(); torch.set_num_threads(4)
    model, tok = load_model(args); model.eval(); named = fl.named_params(model)
    S = {"W0": {n: p.detach().cpu().clone() for n, p in named},
         "A": torch.load(Path(args.stateA) / "weights.pt", map_location="cpu"), "B": torch.load(Path(args.stateB) / "weights.pt", map_location="cpu")}
    G = groups_of([n for n, _ in named]); data = Path(args.data)
    sets = {}
    for t in ("gsm8k", "mbpp", "dolly_qa"):
        rows = [json.loads(l) for l in open(data / t / "probe.jsonl")]
        if args.limit: rows = rows[: args.limit]
        sets[f"probe_{t}"] = (t, rows)
    g, d = heldout_rows(data, args.limit or args.n_heldout); sets["heldout_gsm8k"] = ("gsm8k", g); sets["heldout_dolly_qa"] = ("dolly_qa", d)
    enc = {}
    tmp = Path(tempfile.mkdtemp())
    for k, (t, rows) in sets.items():
        write_rows(rows, tmp / f"{k}.jsonl")
        pcs = fl.prompt_completion_rows(str(tmp / f"{k}.jsonl"), tok, args.max_len, filter_ctx=False)
        assert len(pcs) == len(rows), (k, len(pcs), len(rows))
        keep_e, keep_r = [], []
        for pc, r in zip(pcs, rows):          # 逐条编码，保持与原始行对齐；提示超过 max_len 的样本被丢弃（与其他脚本一致）
            e = fl.encode_probe([pc], tok, args.max_len)
            if e: keep_e.append(e[0]); keep_r.append(r)
        enc[k] = (t, keep_e, keep_r); print(k, "样本", len(keep_e), "/", len(rows), flush=True)
    Path(args.out).mkdir(parents=True, exist_ok=True)
    for cfg in json.load(open(args.configs)):
        if (Path(args.out) / f"{cfg['name']}.json").exists():
            print("skip", cfg["name"], flush=True); continue
        names = set(n for gname in cfg.get("groups", []) for n in G[gname])
        with torch.no_grad():
            for n, p in named:
                if "interp" in cfg:     # 插值对照：W = A + α(B − A)（全部参数）
                    a = float(cfg["interp"]); p.copy_((S["A"][n] + a * (S["B"][n] - S["A"][n])).to(p.device)); continue
                src = S[cfg["swap_from"]] if (cfg.get("swap_from") and n in names) else S[cfg["base"]]
                p.copy_(src[n].to(p.device))
        res = {"config": cfg}
        if args.nll:
            res["nll"] = {k: nll_eval(model, tok, t, exs, rows) for k, (t, exs, rows) in enc.items()}
        if args.gen:
            t, exs, rows = enc["heldout_gsm8k"]; outs = generate(model, tok, exs, 400)
            gold = [norm_num(HASH.search(r["messages"][-1]["content"]).group(1)) for r in rows]
            strict = [bool((m := HASH.search(o)) and norm_num(m.group(1)) == y) for o, y in zip(outs, gold)]
            relaxed = [bool((ns := NUM.findall(o)) and norm_num(ns[-1]) == y) for o, y in zip(outs, gold)]
            t, exs, rows = enc["probe_mbpp"]; mo = generate(model, tok, exs, 512)
            passed = [run_mbpp(o, r) for o, r in zip(mo, rows)]
            res["gen"] = {"gsm8k_strict": strict, "gsm8k_relaxed": relaxed, "mbpp_pass": passed,
                          "gsm8k_out": outs[:20], "mbpp_out": mo[:20]}
            print(cfg["name"], "GSM8K strict %.3f relaxed %.3f | MBPP pass %.3f" % (sum(strict) / len(strict), sum(relaxed) / len(relaxed), sum(passed) / len(passed)), flush=True)
        if args.nll:
            print(cfg["name"], " ".join(f"{k} {sum(v['s']) / sum(v['c']):.4f}" for k, v in res["nll"].items()), flush=True)
        json.dump(res, open(Path(args.out) / f"{cfg['name']}.json", "w"), ensure_ascii=False)
    print("DONE")


if __name__ == "__main__":
    main()
