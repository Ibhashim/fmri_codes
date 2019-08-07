def single_trial_fit(X, hrf): # X is timecourse of voxel, hrf is hrf timecourse over same time
    X = np.array(X).reshape(len(X),1) # add voxel time course as list - converst to np.array column vector
    hrf = np.array(hrf).reshape(len(hrf),1) # add hrf as list - converts to np.array column vector
    DM = np.concatenate((hrf, np.ones(np.shape(hrf))), axis=1) # creates 2d array - DESIGN MATRIX
    
    Betas = lstsq(DM,X, rcond=None)[0] # solves the equation a x = b by computing a vector x that minimizes the Euclidean 2-norm
    B = Betas.item(0) # generates B values
    CC = inv(DM.transpose().dot(DM))
    VarErr = np.var(X - DM.dot(Betas), ddof=1) # ddof is degrees of freedome - divisor N - ddof
    C = np.zeros((1, (int(np.size(hrf, axis=1)) + 1)))
    C[0, 0] = 1
    T = C.dot(np.divide(Betas, np.sqrt(VarErr * (C.dot(CC.dot(C.transpose())))).item(0))).item(0) # Calculates T values
    return B
