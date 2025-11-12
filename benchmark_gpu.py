import pandas as pd
from dyspyosis import Dyspyosis
import time

GPU_EPOCHS = 100


def run_gpu(df):
    dyspyosis_gpu = Dyspyosis(
        df.values,
        labels=df.index.tolist(),
        rarefication_depth=5000,
        rarefication_count=1000,
    )

    dyspyosis_gpu.run_training(epochs=GPU_EPOCHS)

    return dyspyosis_gpu


if __name__ == "__main__":
    df = pd.read_table("./data/test.tsv", index_col=0)

    print(f"Starting GPU Benchmark with {GPU_EPOCHS} epochs.")

    start = time.perf_counter()

    _ = run_gpu(df)

    stop = time.perf_counter()

    gpu_time = stop - start

    print("==================\n\n")
    print(f"Processed {GPU_EPOCHS} epochs on GPU in {gpu_time:0.4f} seconds.")
