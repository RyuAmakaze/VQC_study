import numpy as np
from qulacs import QuantumState
from qulacs_core import ParametricQuantumCircuit
from qulacs.gate import ParametricRY
from qulacs.state import inner_product

def main():
    n_qubit = 10
    n_iter = 100
    lr = 0.1

    circuit = ParametricQuantumCircuit(n_qubit)
    circuit.add_parametric_gate(ParametricRY(0, 0.1))
    circuit.add_parametric_gate(ParametricRY(0, 0.2))
    circuit.add_parametric_gate(ParametricRY(0, 0.3))
    print(circuit)

    target_state = QuantumState(n_qubit)
    target_state.set_computational_basis(1)  # 教師状態 |1⟩

    input_state = QuantumState(n_qubit)
    input_state.set_zero_state()  # 入力状態 |0⟩

    for step in range(n_iter):
        # フォワード
        state = input_state.copy()
        circuit.update_quantum_state(state)

        # 予測振幅と確率
        amp = inner_product(target_state, state)
        prob = abs(amp) ** 2
        loss = -np.log(prob + 1e-10)

        # 教師状態を引数にbackprop_inner_productを使う
        grads_complex = circuit.backprop_inner_product(target_state)
        grads = []

        for dψ_dθ in grads_complex:
            grad = -2 * np.real(np.conj(amp) * dψ_dθ) / (prob + 1e-10)
            grads.append(grad)

        # パラメータ更新
        for i, grad in enumerate(grads):
            theta = circuit.get_parameter(i)
            circuit.set_parameter(i, theta - lr * grad)

        if step % 10 == 0 or step == n_iter - 1:
            print(f"[{step:03d}] loss={loss:.6f}  prob={prob:.6f}  θ={circuit.get_parameter(0):.4f}")

if __name__ == "__main__":
    main()
