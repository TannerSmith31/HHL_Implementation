#THIS FILE CONTAINS THE TEST CODE I HAD GPT GENERATE (AFTER FIGHTING WITH IT FOR LIKE AN HOUR) TO TEST QPE AND QFT
#WE WILL WANT MORE TESTING TO ENSURE EVERYTHING IS WORKING PROPOERLY, BUT THIS IS A STARTING POINT
#NOTE: I HAD ALL THIS AT THE END OF THE HHL.PY FILE BUT MOVED IT HERE TO NOT CONFUSE THE MAIN IMPLEMENTATION CODE

#######################TESTING QPE###############################
# ----------------------------
# Parameters
# ----------------------------
psi_size = 1              # number of qubits in the psi register
num_control_qubits = 3    # number of qubits in the control register

# ----------------------------
# Define a simple U and eigenstate
# ----------------------------
# Example: Z gate as unitary, eigenstate |1>
U = np.array([[1, 0],
              [0, -1]], dtype=complex)
eigenstate_index = 1  # we want |1> as eigenstate

# ----------------------------
# Initialize psi register in eigenstate
# ----------------------------
total_qubits = psi_size + num_control_qubits
init_circuit = QuantumCircuit(total_qubits)

# Set psi qubit to eigenstate |1>
init_circuit.x(num_control_qubits)  # psi qubit is the last qubit

# ----------------------------
# Call your QPE circuit function
# ----------------------------
# Assuming you have a function: gen_QPE_circuit(U, num_control_qubits)
qpe_circuit = gen_QPE_circuit(U, num_control_qubits)

# Decompose all controlled-unitaries to basic gates (Aer can simulate)
qpe_circuit = qpe_circuit.decompose(reps=10)

# ----------------------------
# Combine init state and QPE
# ----------------------------
full_circuit = init_circuit.compose(qpe_circuit)

# ----------------------------
# Add classical register and measure control qubits
# ----------------------------
creg = ClassicalRegister(num_control_qubits)
full_circuit.add_register(creg)
full_circuit.measure(range(num_control_qubits), creg)

# ----------------------------
# Draw the circuit
# ----------------------------
print(full_circuit.draw('mpl'))

# ----------------------------
# Simulate
# ----------------------------
sim = AerSimulator()
result = sim.run(full_circuit, shots=2048).result()
counts = result.get_counts()

# ----------------------------
# Print and plot results
# ----------------------------
print("Measurement counts:", counts)
plot_histogram(counts)

###################END TESTING QPE#########################

#####################TESTING QFT###########################
# n=3
# # Example input state |3> = |011> for 3 qubits
# qc = QuantumCircuit(n)
# qc.x(1)  # second qubit is 1
# qc.x(0)  # first qubit is 1 (little-endian order)
# qc.append(gen_QFT_circuit(n), range(n))

# # Simulate
# backend = Aer.get_backend('statevector_simulator')
# job = backend.run(transpile(qc, backend))
# result = job.result()
# statevector = result.get_statevector()

# # Visualize
# for k, amp in enumerate(statevector):
#     print(f"|{k}>: amplitude = {amp}, magnitude = {np.abs(amp)}, phase = {np.angle(amp)}")
# plot_state_city(statevector)

########################### END TESTING QFT #########################