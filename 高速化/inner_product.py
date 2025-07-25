import time
import numpy as np
from tqdm import tqdm

from qulacs import QuantumStateGpu
from qulacs import ParametricQuantumCircuit
from qulacs.gate import ParametricRY
from qulacs.state import inner_product  # ← ここが重要！

def main():
    n = 10  # qubit数
    n_iter = 10000

    state_ref = QuantumStateGpu(n)
    state = QuantumStateGpu(n)
    circuit = ParametricQuantumCircuit(n)

    # パラメトリックゲートを追加
    for i in range(n):
        circuit.add_parametric_gate(ParametricRY(i, 0.1))

    # 参照状態を1回作る
    circuit.update_quantum_state(state_ref)

    print("Starting inner_product benchmark...")
    start = time.time()
    for _ in tqdm(range(n_iter)):
        state.set_zero_state()
        circuit.update_quantum_state(state)
        _ = inner_product(state_ref, state)  # ✅ Python APIから呼べる！
    end = time.time()

    print(f"Elapsed (inner_product x {n_iter}): {end - start:.3f} sec")
    time.sleep(5)

if __name__ == "__main__":
    main()
