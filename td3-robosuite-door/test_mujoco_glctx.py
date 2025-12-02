# test_mujoco_glctx.py
import os, gc, traceback
os.environ.setdefault("MUJOCO_GL", "egl")   # or "osmesa"

import mujoco
print("mujoco:", getattr(mujoco, "__version__", "unknown"))
print("Has GLContext:", hasattr(mujoco, "GLContext"))

try:
    # Prefer the context-manager style if available
    try:
        with mujoco.GLContext(64, 64) as ctx:
            ctx.make_current()
            print("GLContext made current inside 'with' block.")
    except TypeError:
        # if 'with' is not supported, fall back to manual create -> make_current -> delete -> gc
        ctx = mujoco.GLContext(64, 64)
        ctx.make_current()
        print("GLContext created and made current (manual).")
        # delete while Python still has OpenGL modules loaded
        del ctx
        gc.collect()
        print("Deleted GLContext and forced garbage collection.")
except Exception:
    traceback.print_exc()
finally:
    # Force a brief pause and GC so any destructors run now (not at interpreter shutdown)
    gc.collect()
    print("End of test script — GC called.")
