"""Per-step DUST.step wall-clock and peak memory at the paper config, pop 256, real batches."""
import sys, time, json, torch
sys.path.insert(0, '.')
from dust import DUST, GPTConfig, recipe, make_model, make_optimizer, load_sequences
variant = sys.argv[1]; steps = int(sys.argv[2])
kw = dict(A=dict(), B=dict(value_credit='analytic'), C=dict(value_credit='analytic', value_gamma=1.0))[variant]
dev = torch.device('cuda'); torch.set_float32_matmul_precision('high')
settings = recipe('1M', 256, **kw)
model = make_model(GPTConfig(), dev, 42); opt = make_optimizer(model, settings)
est = DUST(model, settings, 42)
rows = load_sequences('data/train_1M.pt', 8 * (steps + 1))
times = []
for s in range(steps + 1):  # step 0 is warm-up, discarded
    b = rows[s * 8:(s + 1) * 8].to(dev)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t = time.perf_counter()
    loss = est.step(b[:, :-1].contiguous(), b[:, 1:].contiguous(), opt)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t)
print(json.dumps(dict(variant=variant, warmup_s=round(times[0], 2), step_s=[round(x, 2) for x in times[1:]],
                      peak_gib=round(torch.cuda.max_memory_allocated() / 2**30, 2), loss=loss)), flush=True)
