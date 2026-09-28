"""Batched adaptation and serving benchmark, with optional sampled GPU power.

No checkpoint writes outside this rebuttal directory; no API or network calls.
Dictionary/model loading is excluded from warm per-task adaptation and reported
separately. Prompt-state replacement and full-ICL routing are not timed here.
"""
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
from analyze_evidence import dictionary
from cass.extract import extract_fewshot_z
from cass.dictionary import build_multilayer_dictionary
from cass.models import HookedLM
from cass.pipeline import code_for, ops_for, z_list_from_Z
from cass.tasks import load_task, icl_prompt, zs_prompt, build_fewshot_pair_prompts


class PowerSampler:
    def __init__(self):
        self.samples = []
        self.error = None
        try:
            self.proc = subprocess.Popen([
                "nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits",
                "--id=" + os.environ.get("CASS_POWER_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]),
                "--loop-ms=100"], stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True)
            def read():
                for line in self.proc.stdout:
                    try:
                        self.samples.append((time.perf_counter(), float(line.strip())))
                    except ValueError:
                        self.error = line.strip()
            self.thread = threading.Thread(target=read, daemon=True)
            self.thread.start()
        except OSError as exc:
            self.error = str(exc)
            self.proc = None

    def joules(self, start, end):
        data = np.array(self.samples)
        if len(data) < 2:
            return None
        # Need bracketing samples, rather than silently extrapolating.
        if data[0, 0] > start or data[-1, 0] < end:
            return None
        inner = data[(data[:, 0] > start) & (data[:, 0] < end), 0]
        ts = np.r_[start, inner, end]
        watts = np.interp(ts, data[:, 0], data[:, 1])
        integrate = getattr(np, "trapezoid", None) or np.trapz
        return float(integrate(watts, ts))

    def close(self):
        if self.proc:
            self.proc.terminate()
            self.proc.wait(timeout=5)


def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    G, _ = dictionary()
    dictionary_load_s = time.perf_counter() - started
    started = time.perf_counter()
    hlm = HookedLM("llama31-8b")
    model_load_s = time.perf_counter() - started
    sampler = PowerSampler()
    out = dict(
        model="llama31-8b", gpu=torch.cuda.get_device_name(0),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        torch=torch.__version__, dtype="bfloat16", attention="sdpa",
        repeats=5, tasks=["antonym", "country-capital", "english-french"],
        model_load_seconds=model_load_s, dictionary_rebuild_seconds=dictionary_load_s,
        energy_scope="Sampled GPU board power, 100ms polling, trapezoidal integration; includes idle/background board draw, excludes CPU/system energy",
        scope="Warm model/dictionary; zero-shot query path and hybrid steering; natural greedy outputs, max_new_tokens=8; no shared-prefix cache",
        dictionary_policy="Each benchmark target is held out of its 31-skill dictionary; rebuilding those dictionaries is reported separately from adaptation",
        held_out_dictionary_rebuild_seconds={},
        measurements=[])

    def measure(task, name, fn, **metadata):
        fn()  # configuration-specific warmup
        records = []
        for _ in range(out["repeats"]):
            torch.cuda.synchronize()
            start = time.perf_counter()
            value = fn()
            torch.cuda.synchronize()
            end = time.perf_counter()
            time.sleep(0.15)  # outside latency interval, to bracket power samples
            records.append(dict(seconds=end-start, gpu_board_joules=sampler.joules(start, end)))
        row = dict(task=task, measurement=name, **metadata, repeats=records,
                   median_seconds=float(np.median([r["seconds"] for r in records])))
        js = [r["gpu_board_joules"] for r in records if r["gpu_board_joules"] is not None]
        row["median_gpu_board_joules"] = float(np.median(js)) if js else None
        if isinstance(value,list) and all(isinstance(v,str) for v in value):
            row['reencoded_output_tokens_excluding_specials']=[len(hlm.tok.encode(v,add_special_tokens=False)) for v in value]
        out["measurements"].append(row)
        (HERE / "latency_results.json").write_text(json.dumps(out, indent=2)+"\n")
        print(task, name, metadata, row["median_seconds"], row["median_gpu_board_joules"], flush=True)
        return value

    try:
        for name in out["tasks"]:
            task = load_task(name)
            # Same seed-0 demonstrations as the original experiments.
            rng = np.random.default_rng(4)
            ex = [task.fewshot_pool[i] for i in rng.choice(len(task.fewshot_pool), 4, replace=False)]
            started=time.perf_counter()
            D=build_multilayer_dictionary({l:{t:g for t,g in by_task.items() if t!=name}
                                          for l,by_task in G.items()},r0=1)
            out['held_out_dictionary_rebuild_seconds'][name]=time.perf_counter()-started
            for bs in [8, 24]:
                Z = measure(name, "extraction", lambda: extract_fewshot_z(
                    hlm, ex, seed=0, batch_size=bs, n_reps=6),
                    sequences=48, batch_size=bs, model_calls=2*int(np.ceil(24/bs)))

            clean, corr = build_fewshot_pair_prompts(ex, random.Random(9000), n_reps=6)
            def combined():
                H = hlm.last_token_hiddens(clean+corr, batch_size=48)
                return (H[:24]-H[24:]).reshape(4, 6, *H.shape[1:]).mean(1)
            measure(name, "extraction_combined", combined, sequences=48, batch_size=48, model_calls=1)

            def assemble():
                zl = z_list_from_Z(D, Z)
                code = code_for(D, zl)
                return ops_for(D, code, 1.0, 2.0, 1.0, delta_vec=np.mean(zl, axis=0))
            ops, layers = measure(name, "coding_and_operator", assemble)
            def end_to_end():
                Z0 = extract_fewshot_z(hlm, ex, seed=0, batch_size=8, n_reps=6)
                zl = z_list_from_Z(D, Z0)
                return ops_for(D, code_for(D, zl), 1.0, 2.0, 1.0, delta_vec=np.mean(zl, axis=0))
            ops,layers=measure(name, "adaptation_end_to_end", end_to_end, extraction_batch_size=8)
            for bs in [1, 8, 25]:
                queries = task.eval_queries[:bs]
                q0 = [zs_prompt(x) for x, _ in queries]
                q4 = [icl_prompt(ex, x) for x, _ in queries]
                # Warm per-batch natural generation, including tokenization and hooks.
                measure(name, "serve_cass", lambda: hlm.generate(q0, batch_size=bs, op=ops, layer=layers),
                        queries=len(queries), batch_size=bs)
                measure(name, "serve_icl4", lambda: hlm.generate(q4, batch_size=bs),
                        queries=len(queries), batch_size=bs)
        out["power_sampling_error"] = sampler.error
        (HERE / "latency_results.json").write_text(json.dumps(out, indent=2)+"\n")
    finally:
        sampler.close()


if __name__ == "__main__":
    main()
