import time
from qulacs import QuantumStateGpu, QuantumCircuit
from qulacs.gate import X
from tqdm import tqdm

def main():
    state = QuantumStateGpu(25)
    print("Using GPU:", state.get_device_name())

    circuit = QuantumCircuit(25)
    for i in range(25):
        circuit.add_gate(X(i))

    print("Starting heavy circuit execution...")
    start = time.time()
    for _ in tqdm(range(100000)):
        circuit.update_quantum_state(state)  # ✅ 正しい呼び出し
    end = time.time()

    print(f"Elapsed: {end - start:.3f} sec")

    # モニタリングのためsleep
    time.sleep(10)

if __name__ == "__main__":
    main()
