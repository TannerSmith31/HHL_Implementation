from qiskit import QuantumCircuit, transpile, ClassicalRegister #transpile was used in testing once
from qiskit.circuit.library import UnitaryGate
from qiskit_aer import AerSimulator, Aer
from qiskit.visualization import plot_state_city
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import UCRYGate
import numpy as np
from scipy.linalg import expm


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

              |q_0> --------------------[H]-------------------------*----//////-- MSB
|controlReg>  ...                       [H]                         |    |IQFT|
              |q_controlRegSize> -------[H]-----------__*__ ------__|__--|____|-- LSB
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
    rows = len(U)
    cols = len(U[0])

    if rows != cols:
        raise ValueError("U must be a square matrix.")

    # Check if dimension is a power of 2
    dim = rows
    if (dim & (dim - 1)) != 0:
        raise ValueError("Matrix dimension must be a power of 2 (2^n × 2^n).")

    # Compute n = log2(dim)
    n = int(np.log2(dim))

    return n
    

"""
FUNCTION to generate the ancilla rotation circuit in HHL.
""" 
def gen_ancilla_rotation_circuit(clockReg_size, C):
    rotationCircuit_size = 1 + clockReg_size  #the size of the circuit is 1 ancilla qubit and the clock register
    qc = QuantumCircuit(rotationCircuit_size)
    
    #Compute angles for controlled rotation
    N = 2 ** clockReg_size
    angles = []
    for c in range(N):
        if c == 0:
            angles.append(0)   # undefined -> 0 rotation
        else:
            lam = c / N
            theta = 2 * np.arcsin(C / lam)
            angles.append(theta)
    
    #Put angles into a controlled y rotation gate
    rotation_gate = UCRYGate(angles)
    
    #add the rotation gate to the quantum circuit to return it
    qc.append(rotation_gate, range(rotationCircuit_size))
    print(qc.draw())
    return qc

def gen_ancilla_rotation_circuit_from_eigenvalues(eigenvalues, C):
    num_clock_qubits = int(np.ceil(np.log2(len(eigenvalues))))
    qc = QuantumCircuit(1 + num_clock_qubits)  # ancilla + clock

    N = 2 ** num_clock_qubits

    # Compute angles
    angles = []
    for k in range(N):
        if k < len(eigenvalues):
            lam = eigenvalues[k]
            ratio = np.clip(C / lam, 0, 1)   # clamp to avoid NaN
            theta = 2 * np.arcsin(ratio)
        else:
            theta = 0
        angles.append(theta)

    # Apply UCRY gate with all angles to ancilla + clock
    qc.append(UCRYGate(angles), range(1 + num_clock_qubits))

    return qc
    

"""
FUNCTION to generate HHL circuit that will solve an equation of the form Ax = b where
A is a square hermetian matrix, and x and b are vectors
INPUT: 
    A: 2D np array  (dimensions must be a power of 2)
    b: 1D np vector (dimension must be a power of 2)
    C: The coefficient for the ancilla qubit rotation (must be smaller than all eigen values of A)
    clockReg_size: an integer representing the number of qubits in the clock register.
        NOTE: These are the control qubits of QPE so I think they control the accuracy of the Phase Estimation
    AeigenVals: Optional parameter for if you know the eigen values of the matrix A so that you can generate the ancilla rotation
        to be exact based on the eigen values that would be seen. If no eigenvals are passed in it will default to generating 1/2^i for 
        i in range(0,clockReg_size).
    
    |ancilla> ------------_-_-_--[Rotation]---_-_-_--[M]- |1>    <-- NOTE only keep runs where we measure this to be 1
      |clock> -----------| QPE |-------------|IQPE |----
        |b_0> -[prep_b]--|_____|-------------|_____|-[M]-|x>
    
"""
def gen_HHL_circuit(A, b, C, clockReg_size):
    bReg_size = np.log2(len(b))  #b register is log(bSize) since there are 2^n entries in b and n qubits have 2^n basis states
    AeigenVals = np.linalg.eigvals(A)
    
    #ensure b is of a correct size
    if bReg_size % 1 != 0:
        print("len(b) must be a power of 2 (len(b)=2^i). Returning None")
        return None
    bReg_size = int(bReg_size)
    
    total_num_qubits = 1 + clockReg_size + bReg_size  # 1 <-- ancilla qubit
    
    qc = QuantumCircuit(total_num_qubits, name="HHL Circuit")
    
    #---------------------------------------
    # STEP 1: Prepare b (amplitude encoding)
    #---------------------------------------
    
    normalized_b = b / np.linalg.norm(b)     # we have to normalize b
    targetQubits = list(range(total_num_qubits - bReg_size, total_num_qubits)) #get the indicies of the b register
    qc.initialize(normalized_b, targetQubits) #initializes the b register with b using amplitude encoding
    
    #---------------------------------------
    # STEP 2: Phase Estimation
    #---------------------------------------
    
    #Create unitary U=e^{iAt} matrix out of (hermetian) A matrix where t = 1/max_eigenval
    t = 1 / max(AeigenVals)
    U = expm(1j * t * np.array(A))
    
    #add QPE circuit to span the clock register and the b register
    qc.append(gen_QPE_circuit(U, clockReg_size), range(1,total_num_qubits))
    
    #---------------------------------------
    # STEP 3: controlled rotation
    #---------------------------------------    
    
#     qc.append(gen_ancilla_rotation_circuit(clockReg_size, C), range(clockReg_size+1))
    qc.append(gen_ancilla_rotation_circuit_from_eigenvalues(AeigenVals, C), range(clockReg_size+1))
    
    #---------------------------------------
    # STEP 4: Inverse Phase Estimation
    #---------------------------------------
    
    #add Inverse QPE circuit to span the clock register and the b register
    inverseQPE = gen_QPE_circuit(U, clockReg_size).inverse()
    qc.append(inverseQPE, range(1,total_num_qubits))

    
    #---------------------------------------
    # STEP 5: Measurement
    #---------------------------------------
    
    #make classical registers for the ancilla qubit and the b qubits
    ancilla_c = ClassicalRegister(1, 'anc')              # ancilla measurement
    b_c = ClassicalRegister(bReg_size, 'breg')          # b-register measurement
    qc.add_register(ancilla_c)
    qc.add_register(b_c)
    
    #measure the quantum circuit and put results in the classical registers
    qc.measure(0, ancilla_c[0])

    b_start = clockReg_size + 1 #have to add 1 for the ancilla qubit at beginning 
    for i in range(bReg_size):
        qc.measure(b_start + i, b_c[i])
        
    return qc

    

"""
Function to run the HHL algorithm on a given A matrix and b vector. It creates the circuit and runs it, keeping
only relevant runs for final answer (i.e. we only keep runs where the ancilla qubit measures 1)
"""
def runHHL(A,b,C,backend,clockRegSize=2,shots=10000):
    
    qc = gen_HHL_circuit(A,b,C,clockRegSize)
#     print(qc.draw())  TODO: REMOVE
    transpiledQC = transpile(qc, backend, optimization_level=3)
#     print(transpiledQC.draw())  TODO: REMOVE
    result = backend.run(transpiledQC, shots=shots).result()
    counts = result.get_counts()

    #Only keep shots where the ancilla qubit is 1
    filteredCounts = {}
    for bitstring, count in counts.items():
        # split format where 1st chunk is b register and second is ancilla register: '01 1' -> ('01', '1')
        b_state, ancilla_state = bitstring.split()

        if ancilla_state == '1':
            # accumulate the b-register counts for ancilla = 1
            filteredCounts[b_state] = count
    
    return filteredCounts
    