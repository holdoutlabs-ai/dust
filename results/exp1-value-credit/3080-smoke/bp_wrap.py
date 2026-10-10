import runpy, sys, time, torch
sys.path.insert(0, '.')
sys.argv = ['baselines/backprop.py'] + sys.argv[1:]
t = time.monotonic()
runpy.run_path('baselines/backprop.py', run_name='__main__')
print(f'WALL {time.monotonic() - t:.1f}s PEAK {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB', flush=True)
