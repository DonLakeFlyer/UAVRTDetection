import matlab as ml

import numpy as np

def cartesianProduct(A: np.ndarray):
    #   returns the Cartesian product C of the (values in the) rows 
    #   of the input matrix A.
    #   Each row of A is taken as a set to draw from
    #   Assumes all sets have the same number of members

    if A.ndim != 2:
        raise

    #num_sets = size(A,1);
    #num_members_per_set = size(A,2);
    num_sets            = A.shape[0]
    num_members_per_set = A.shape[1]

    #num_rows_C = num_members_per_set^num_sets;
    #num_cols_C = num_sets; 
    num_rows_C = num_members_per_set ** num_sets
    num_cols_C = num_sets;

    #C = zeros( num_rows_C, num_cols_C );
    C = np.zeros((num_rows_C, num_cols_C))

    # loop through columns of C

    # for i_col = 1:num_sets

    #     # fill the column (aka fill the elements in the column)
    #     # stride is number of elements to repeat in a column
    #     # num_strides is number of strides per column
    #     stride = num_members_per_set^( num_sets - i_col );
    #     num_strides = num_rows_C/stride;
    #     i_col_A = 0; # the column in A that we use
    #     for i_stride = 1:num_strides
    #         i_col_A = i_col_A + 1;
    #         if (i_col_A > num_members_per_set)
    #             i_col_A = 1;
    #         end
    #         star = (i_stride - 1)*stride + 1;
    #         # column of C gets elements from row of A
    #         C( star:star+stride-1, i_col ) = A( i_col, i_col_A );
    #     end
    # end

    for i_col in range(1, num_sets+1):
        # fill the column (aka fill the elements in the column)
        # stride is number of elements to repeat in a column
        # num_strides is number of strides per column
        stride      = num_members_per_set ** (num_sets - i_col)
        num_strides = num_rows_C // stride
        i_col_A     = 0 # the column in A that we use
        for i_stride in range(1, num_strides+1):
            i_col_A = i_col_A + 1
            if i_col_A > num_members_per_set:
                i_col_A = 1
            star = (i_stride - 1) * stride + 1;
            # column of C gets elements from row of A
            C[ml.siVector(star - 1, star + stride - 2), i_col - 1] = A[i_col-1, i_col_A-1]

    return C

