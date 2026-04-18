# src/runtime.py
import multiprocessing
import tensorflow as tf


def setup_tensorflow_runtime(verbose: bool = True) -> int:
    """
    TensorFlow runtime inicializálás:
    - GPU-k listázása
    - memory growth bekapcsolása
    - CPU magszám lekérdezése

    Returns
    -------
    int
        Elérhető CPU magok száma.
    """
    gpus = tf.config.list_physical_devices("GPU")
    cpu_count = multiprocessing.cpu_count()

    if verbose:
        print("Num GPUs Available:", len(gpus))

    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        print(f"[WARN] GPU memory growth beállítási hiba: {e}")

    if verbose:
        print(f"Number of CPU cores: {cpu_count}")

    return cpu_count
