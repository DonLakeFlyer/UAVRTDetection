from generatePulsePositionsMatrix import *
import matlab as ml

import numpy as np
import scipy

# function Wq = assembleWq(PRI_means_q,PRI_jitters_q, K)
def assembleWq(PRI_means_q: np.ndarray, PRI_jitters_q: np.ndarray, K: float):
    if PRI_means_q.ndim != 1:
        raise
    if PRI_jitters_q.ndim != 1:
        raise

    # PPM Contains the column positions (time windows) in the S matrix to search
    # for the K pulses assuming the first pulse arrives at window 1. 
    # pulsePositionMatrix = generate_pulse_positions_func(K, PRI_means_q, PRI_jitters_q );
    pulsePositionMatrix = generatePulsePositionsMatrix(K, PRI_means_q, PRI_jitters_q)

    #  if K>1
    #      # n_shifts = min(pulsePositionMatrix(:,2))-1;   # The number of shifts necessary. 
    #  
    #  else
    #      n_shifts = max(PRI_means_q)+max(PRI_jitters_q);
    #  end

    # The max element of the ppm for K>1 is (K-1)*(N+M)+J+1 which is the latest 
    # possible index of the Kth pulse if the first arrived at n = 1. 
    # 
    # The max element of the ppm for K=1 is 1 which is the first possible index
    # of the 1st pulse. 
    # 
    # The size of the STFT matrix is the latest possible location of the Kth
    # pulse which is equal to K*(N+M)+J+1
    # 
    # We want to take the PPM matrix that contains the the time locations for
    # the search if the first pulse is at n=1 and shift them to do the search if
    # the first pulse arrives and n = 2,..., N+M+J+1 which is the latest
    # possible location of the pulse pulse. The difference is N+M+J+1-1 = N+M+J
    # This is the max number we need to add the the PPM matrix. 
    # 
    # For the K = 1 case, the PPM is simply 1 and we add N+M+J which generates a
    # Wq matrix with a dimension of 1+N+M+J. This is equivalent to the STFT
    # size definition for K = 1, which is K(N+M)+J+1 = N+M+J+1, so the sizes
    # will be compatible. 
    # 
    # For the K > 1 case however the maximum element of the PPM is
    # (K-1)(N+M)+J+1. Adding N+M+J to this will result in a Wq with a dimension
    # of (K-1)(N+M)+J+1+N+M+J = K(N+M)+2J+1. This will be J elements larger than
    # the STFT definitions of K(N+M)+J+1. This extra J comes from the fact that
    # the max element of the PPM include the jitter of the final pulse and we
    # also include the jitter in the shifting. Because jitter is stochastic and
    # zero mean, we don't sum jitter over multiple pulses. We need to remove the
    # extra jitter when shifting for K>1. 

    # if K>1
    #     n_shifts = max(PRI_means_q);# Equivalent to N+M+J-J
    # else
    #     n_shifts = max(PRI_means_q)+max(PRI_jitters_q);# Equivalent to N+M+J
    # end
    if K > 1:
        n_shifts = PRI_means_q.max()                        # Equivalent to N+M+J-J
    else:
        n_shifts = PRI_means_q.max() + PRI_jitters_q.max()  # Equivalent to N+M+J

    # ppmRows     = size(pulsePositionMatrix, 1);       # Number of rows in pulse_position_matrix
    # ppmCols     = size(pulsePositionMatrix, 2);       # Number of cols in pulse_position_matrix --- should always be K
    # ppmBig      = repmat(pulsePositionMatrix, n_shifts+1, 1); # Create multiple copies of the PPM matrix. Each copy will eventually be shifted. The second copy will get a 1 added to it, the third a 2 added, and so on up to n_shifts. Use n_shifts+1 to account for the first one that receives no shifting.
    # ppmBigRows  = size(ppmBig,1);                      # Get the number of rows in the big PPM matrix
    # shiftermat  = repmat(0:n_shifts, ppmRows, 1);      # Create a matrix of the shifts needed for each copy of the PPM matrix
    # WqRowInds   = repmat(shiftermat(:), 1, ppmCols) + ppmBig; 
    # WqColInds   = repmat(1:ppmBigRows,1,ppmCols);
    # WqElements  = ones(size(WqRowInds));
    # Wq          = sparse(WqRowInds(:),WqColInds(:),WqElements(:));
    ppmRows     = pulsePositionMatrix.shape[0]                              # Number of rows in pulse_position_matrix
    ppmCols     = pulsePositionMatrix.shape[1]                              # Number of cols in pulse_position_matrix --- should always be K
    ppmBig      = np.tile(pulsePositionMatrix,      (n_shifts + 1, 1))      # Create multiple copies of the PPM matrix. Each copy will eventually be shifted. The second copy will get a 1 added to it, the third a 2 added, and so on up to n_shifts. Use n_shifts+1 to account for the first one that receives no shifting.
    ppmBigRows  = ppmBig.shape[0]                                           # Get the number of rows in the big PPM matrix
    shiftermat  = np.tile(ml.siVector(0, n_shifts), (ppmRows, 1))           # Create a matrix of the shifts needed for each copy of the PPM matrix
    WqRowInds   = np.tile(shiftermat.reshape(-1,1), (1, ppmCols)) + ppmBig 
    WqColInds   = np.tile(np.arange(ppmBigRows),    (1, ppmCols))
    WqRowInds   = WqRowInds.ravel(order='F').astype(np.intc)
    WqColInds   = WqColInds.ravel(order='F').astype(np.intc)
    WqElements  = np.ones(len(WqRowInds))
    Wq          = scipy.sparse.csr_array((WqElements, (WqRowInds, WqColInds)), dtype=np.float)
    Wq.eliminate_zeros()

    return Wq