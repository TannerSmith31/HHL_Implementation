"""
File to run HHL algorithm for different test cases
NOTE: this code can be pasted into the end of HHL.py to run directly from there
Otherwise you will have to do imports and stuff here.
"""

##################### TESTING HHL ############################
backend = AerSimulator()

# A = [[0.25, 0],[0,0.75]]
# b = [1,0]
# C = 0.25 #Ensure C <= all eigenvalues
# clockReg_size = 2
# AeigenVals = np.linalg.eigvals(A)

A = [[0.125,0,0,0],[0,0.375,0,0],[0,0,0.625,0], [0,0,0,0.875]]
A = np.diag([0.125, 0.375, 0.625, 0.875])
b = [0.5,0.5,0.5,0.5]
C = 0.125
clockReg_size = 2
AeigenVals = np.linalg.eigvals(A)

# A = [[1, -0.333333],[-0.333333,1]]
# b = [0,1]
# C = 0.25
# clockReg_size = 1


HHLresults = runHHL(A,b,C,backend,clockRegSize=clockReg_size)
print(HHLresults)

print("DONE")