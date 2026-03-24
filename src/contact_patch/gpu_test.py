import cupy as cp

import numpy as np

for i in range(10):
    a_np = np.arange(10)
    s = cp.cuda.Stream()
    with s:
        a_cp = cp.asarray(a_np)  # H2D transfer on stream s
        b_cp = cp.sum(a_cp)      # kernel launched on stream s
        assert s == cp.cuda.get_current_stream()

    s = cp.cuda.Stream()
    s.use()  # any subsequent operations are done on steam s  
    b_np = cp.asnumpy(b_cp)
    assert s == cp.cuda.get_current_stream()
    cp.cuda.Stream.null.use()  # fall back to the default (null) stream
    assert cp.cuda.Stream.null == cp.cuda.get_current_stream()

    e1 = cp.cuda.Event()
    e1.record()
    a_cp = b_cp * a_cp + 8
    e2 = cp.cuda.get_current_stream().record()

    # set up a stream order
    s2 = cp.cuda.Stream()
    s2.wait_event(e2)
    with s2:
        # the a_cp is guaranteed updated when this copy (on s2) starts
        a_np = cp.asnumpy(a_cp)

    # timing
    e2.synchronize()
    t = cp.cuda.get_elapsed_time(e1, e2)
    print(t)