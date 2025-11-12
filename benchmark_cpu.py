import pandas as pd
import time

CPU_EPOCHS = 100


def run_cpu(df):
    from dyspyosis import Dyspyosis

    dyspyosis_cpu = Dyspyosis(
        df.values,
        labels=df.index.tolist(),
        rarefication_depth=5000,
        rarefication_count=1000,
    )

    dyspyosis_cpu.run_training(epochs=CPU_EPOCHS)

    return dyspyosis_cpu


if __name__ == "__main__":
    # if not os.environ["CUDA_VISIBLE_DEVICES"] == "-1":
    #     print(
    #         "ERROR: CUDA_VISIBLE_DEVICES should be set to -1 before starting this script!"
    #     )
    # if not os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID":
    #     print(
    #         "ERROR: CUDA_DEVICE_ORDER should be set to PCI_BUS_ID before starting this script!"
    #     )

    df = pd.read_table("./data/test.tsv", index_col=0)

    print(f"Starting CPU Benchmark with {CPU_EPOCHS} epochs.")

    start = time.perf_counter()

    _ = run_cpu(df)

    stop = time.perf_counter()

    cpu_time = stop - start

    print("==================\n\n")
    print(f"Processed {CPU_EPOCHS} epochs on CPU in {cpu_time:0.4f} seconds.")
