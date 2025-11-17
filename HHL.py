from qiskit import QuantumCircuit, transpile, ClassicalRegister #transpile was used in testing once
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_state_city
from qiskit.visualization import plot_histogram
import numpy as np

"""
FUNCTION to generate a Quantum Fourier Transform circuit of size n
"""
def gen_QFT_circuit(n):
    qc = QuantumCircuit(n, name="QFT")

    # Apply the QFT
    for i in range(n):
        # Hadamard on qubit i
        qc.h(i)

        # Controlled phase rotations
        for j in range(i+1, n):
            qc.cp(np.pi / (2 ** (j - i)), j, i)

    # Reverse the qubit order (QFT includes a bit-reversal)
    for i in range(n//2):
        qc.swap(i, n - i - 1)

    return qc


"""
FUNCTION to generate Quantum Phase Estimation circuit (estimates the eigenvalue of a unitary U given an eigenvector psi)

              |q_0> --------------------[H]-------------------------*----//////--
|controlReg>  ...                       [H]                         |    |IQFT|
              |q_controlRegSize> -------[H]-----------__*__ ------__|__--|____|--
              |q_controlRegSize + 1> ----------------|     |     |     |---------
    |psiReg>  ...                                    |U^2^0| ... |U^2^i|            **Believe it or not, Tanner made this diagram, not GPT <(^o^)>
              |q_{psiRegSize + 1 + controlRegSize}>--|_____|     |_____|---------

INPUT:
    U : 2D array = the unitary matrix we are trying to estimate the eigenvalues of
    control_reg_size : int = size (#qubits) of the control register (controls the number of U operations applied to psi reg). 
        NOTE: this is also the register that is measured when only doing QPE (but in our case, for HHL, we don't measure)
LOCAL VAR:
    psi_reg_size : int = size (#qubits) of the register that psi (U's eigen vector) goes in (the register that U is applied to)
        NOTE: U should be square of size 2^psi_reg_size
"""
#TODO: maybe remove the psi_reg_size parameter and just set psi_reg_size = rows using the U.shape method
def gen_QPE_circuit(U, control_reg_size):
    psi_reg_size = infer_qubits_from_unitary(U)
    qc = QuantumCircuit(control_reg_size + psi_reg_size, name="QPE")
    
    #apply hadimar gates to all qubits in the control register
    for i in range(control_reg_size):
        qc.h(i)
    
    #apply U^{2^i} gates to psi_reg based on control_reg
    for i in range(control_reg_size):
        # compute U^{2^i}
        Upow = np.linalg.matrix_power(U, 2**i)

        # turn matrix to gate
        UpowGate = UnitaryGate(Upow)

        # turn to controlled-U^{2^i}
        CUpowGate = UpowGate.control()

        # qubits that U acts on
        targetQubits = list(range(control_reg_size, control_reg_size + psi_reg_size))

        # append controlled gate
        qc.append(CUpowGate, [control_reg_size - i - 1] + targetQubits)  #reason for [control_reg_size-i-1]: we want the U^2^0 to be applied to the last qubit in the control register and U^2^i to be controlled by the first
    
    #append the inverse quantum fourier transform to the end
    inverseQFT = gen_QFT_circuit(control_reg_size).inverse()
    qc.append(inverseQFT, range(control_reg_size))
    return qc

"""
HELPER FUNCTION to determine the number of qubits a unitary gate U acts on
Use the following property:
    U = 2^(registerSize) ----> registerSize = log(U)
"""
def infer_qubits_from_unitary(U):
    rows, cols = U.shape

    if rows != cols:
        raise ValueError("U must be a square matrix.")

    # Check if dimension is a power of 2
    dim = rows
    if (dim & (dim - 1)) != 0:
        raise ValueError("Matrix dimension must be a power of 2 (2^n × 2^n).")

    # Compute n = log2(dim)
    n = int(np.log2(dim))

    return n
    

# Function to generate HHL circuit that will solve an equation of the form Ax = b
# Where A is a square hermetian matrix 
"""
FUNCTION to generate HHL circuit that will solve an equation of the form Ax = b where
A is a square hermetian matrix, and x and b are vectors
INPUT: 
    A: 2D np array  (dimensions must be a power of 2)
    b: 1D np vector (dimension must be a power of 2)
    clockReg_size: an integer representing the number of qubits in the clock register.
        NOTE: These are the control qubits of QPE so I think they control the accuracy of the Phase Estimation
    
    |ancilla> ------------_-_-_--[Rotation]--[M]--_-_-_----- |1>
      |clock> -----------| QPE |-----------------|IQPE |----
        |b_0> -[prep_b]--|_____|-----------------|_____|-[M]-|x>
    
"""
def gen_HHL_circuit(A, b, clockReg_size):
    
    bReg_size = np.log2(b.size())  #b register is log(bSize) since there are 2^n entries in b and n qubits have 2^n basis states
    total_num_qubits = 1 + clockReg_size + bReg_size  # 1 <-- ancilla qubit
    
    qc = QuantumCircuit(total_num_qubits, name="QPE")
    
    #---------------------------------------
    # STEP 1: Prepare b (amplitude encoding)
    #---------------------------------------
    
    #TODO: do we have to normalize b?  normalized_b = b / np.linalg.norm(b)     # normalize
    targetQubits = list(range(total_num_qubits - bReg_size, total_num_qubits)) #get the indicies of the b register
    qc.initialize(b, targetQubits) #initializes the b register with b using amplitude encoding
    
    #---------------------------------------
    # STEP 2: Phase Estimation
    #---------------------------------------
    
    #add QPE circuit to span the clock register and the b register
    qc.append(gen_QPE_circuit(A, bReg_size + clockReg_size), range(1,total_num_qubits))
    
    #---------------------------------------
    # STEP 3: controlled rotation
    #---------------------------------------
    
    #TODO
    
    #---------------------------------------
    # STEP 4: Measurement to get 1 on ancilla???
    #---------------------------------------
    
    #TODO
    
    #---------------------------------------
    # STEP 5: Inverse Phase Estimation
    #---------------------------------------
    
    #add Inverse QPE circuit to span the clock register and the b register
    inverseQPE = gen_QPE_circuit(A, bReg_size + clockReg_size).inverse()
    qc.append(inverseQPE, range(1,total_num_qubits))
    
    #---------------------------------------
    # STEP 6: Measurement
    #---------------------------------------
    
    #TODO

    return qc
    