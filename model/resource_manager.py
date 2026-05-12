import os


def set_gpu_resource(gpus, gpu_memory_mb):
    """
    Configure visible GPUs and optional per-GPU memory cap.

    Args:
        gpus: list of GPU indices to use, e.g. [0], [1], [2], [0,1]
        gpu_memory_mb: int or None, max memory per visible GPU in MB
    """

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    # CPU mode
    if not gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("No GPU selected. Running on CPU.")
        return

    # Make sure ids are integers
    try:
        requested_gpus = [int(g) for g in gpus]
    except Exception:
        print("Warning! Invalid GPU list:", gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        print("Falling back to CPU.")
        return

    # Set requested visible devices first
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in requested_gpus)

    try:
        import tensorflow as tf
    except Exception as e:
        print("Warning! TensorFlow import failed:", e)
        print("CUDA_VISIBLE_DEVICES was set to:", os.environ["CUDA_VISIBLE_DEVICES"])
        return

    try:
        physical_gpus = tf.config.experimental.list_physical_devices("GPU")
    except Exception as e:
        print("Warning! Could not list physical GPUs:", e)
        physical_gpus = []

    if not physical_gpus:
        print("Warning! No CUDA-capable GPU detected by TensorFlow.")
        print("Requested GPUs:", requested_gpus)
        print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
        return

    print("Visible GPUs for this process:", os.environ.get("CUDA_VISIBLE_DEVICES", ""))
    print("TensorFlow detected {} GPU(s).".format(len(physical_gpus)))

    for i, gpu in enumerate(physical_gpus):
        try:
            if gpu_memory_mb is not None and int(gpu_memory_mb) > 0:
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [
                        tf.config.experimental.VirtualDeviceConfiguration(
                            memory_limit=int(gpu_memory_mb)
                        )
                    ],
                )
                print(f"GPU logical index {i}: memory limited to {gpu_memory_mb} MB")
            else:
                try:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print(f"GPU logical index {i}: memory growth enabled")
                except Exception as growth_err:
                    print(f"Warning! Could not enable memory growth for GPU {i}: {growth_err}")
        except RuntimeError as e:
            print(f"RuntimeError while configuring GPU {i}: {e}")
        except Exception as e:
            print(f"Warning! Failed to configure GPU {i}: {e}")

    try:
        logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        print("Logical GPUs available to TensorFlow:", len(logical_gpus))
        for i, lgpu in enumerate(logical_gpus):
            print(f"  Logical GPU {i}: {lgpu.name}")
    except Exception as e:
        print("Warning! Could not list logical GPUs:", e)
