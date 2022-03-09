import numpy as np

from enot_lite.benchmark import Benchmark
from enot_lite import backend

_BATCH_SIZE = 8


if __name__ == '__main__':
    # Benchmark baseline
    benchmark_baseline = Benchmark(
        batch_size=_BATCH_SIZE,
        onnx_model='baseline.onnx',
        onnx_input={'input': np.ones((_BATCH_SIZE, 3, 480, 480), dtype=np.float32)},
        backends=[backend.OrtTensorrtFloatBackend, backend.OrtTensorrtFloatOptimBackend],
    )
    benchmark_baseline.run()
    benchmark_baseline.print_results()

    # Benchmark tuned
    benchmark_tuned = Benchmark(
        batch_size=_BATCH_SIZE,
        onnx_model='tune.onnx',
        onnx_input={'input': np.ones((_BATCH_SIZE, 3, 536, 536), dtype=np.float32)},
        backends=[backend.OrtTensorrtFloatBackend, backend.OrtTensorrtFloatOptimBackend],
    )
    benchmark_tuned.run()
    benchmark_tuned.print_results()