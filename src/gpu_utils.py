"""GPU cleanup helper used at the end of every probe so memory is released and
the release is clearly logged (a process also frees all its GPU memory on exit,
but doing it explicitly removes ambiguity)."""
import gc


def release(model=None, tag=""):
    try:
        import torch
    except Exception:
        return
    if model is not None:
        try:
            del model
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            with torch.cuda.device(i):
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        resv = sum(torch.cuda.memory_reserved(i)
                   for i in range(torch.cuda.device_count())) / 1e9
        lbl = f" [{tag}]" if tag else ""
        print(f"[gpu_release]{lbl} torch still reserves {resv:.2f} GB; "
              f"this process now exits → the OS releases ALL GPU memory it held.")
