import matlab as ml

# function [outmat] = vecfind(refvec,comparison_op_string,searchvec)
def vecFind(refvec, comparison_op_string, searchvec):
    # VECFIND creates a logical matrix based on the input operator comparing a
    # reference vector and a vector to be compared.
    #    This function can be used to do comparison operations of vectors of
    #    difference sizes. Comparing a single to a vector work natively
    #    in Matlab and create a resulting logic vector, i.e. [1 3 5]>4 outputs
    #    [0 0 1]. Many cases though require comparison of on vector to all the
    #    elements of another. i.e. [1 3 5]>[2 4] would ask which elements of 
    #    [1 3 5] are greater than 2 and which are greater than 4? This function
    #    produces an output matrix whose columns are the logicial comparion of
    #    the reference vector to each element of the search vector. Logicals fpr
    #    comparison are input to this function as strings.
    # 
    #    Example:
    #              outmat = vecfind([1 3 5],'>',[2 4])
    #  
    #              outmat =
    #  
    #              3×2 logical array
    #  
    #              0   0
    #              1   0
    #              1   1
    # 
    # 
    # INPUTS:
    #    refvec              n-element vector of numerics to be compared.
    #    comparison_op_string   A string of the comparison operator to be used
    #                        Valid options are '<' '<=' '>' '>=' '==' '~='
    #    searchvec           m-element vector of numerics used for comparison
    # OUTPUTS:
    #    outmat              nxm logical matrix comparing the ref and search
    #                        vectors
    # # 
    # Author: Michael Shafer
    # Date:   2021-05-25
    # 
    # 2022-01-24 - Updated to use limited comparison operators and removed use
    #              of the eval.m function.
    # # 

    # refvec    = refvec(:);      # Ensure column vector
    # searchvec = searchvec(:);   # Ensure column vector
    refvec    = refvec.reshape(-1, 1)       # Ensure column vector
    searchvec = searchvec.reshape(-1, 1)    # Ensure column vector

    # refmat    = repmat(refvec,    1,              numel(searchvec));
    # searchmat = repmat(searchvec',numel(refvec),  1);
    refmat    = ml.repmat(refvec,                       1,                  ml.numel(searchvec));
    searchmat = ml.repmat(searchvec.conj().transpose(), ml.numel(refvec),   1);

    # if strcmp(comparison_op_string,'>')
    #     outmat = refmat > searchmat;
    # elseif strcmp(comparison_op_string,'>=')
    #     outmat = refmat >= searchmat;
    # elseif strcmp(comparison_op_string,'<')
    #     outmat = refmat < searchmat;
    # elseif strcmp(comparison_op_string,'<=')
    #     outmat = refmat <= searchmat;
    # elseif strcmp(comparison_op_string,'==')
    #     outmat = refmat == searchmat;
    # elseif strcmp(comparison_op_string,'~=')
    #     outmat = refmat == searchmat;
    # else
    #     error('UAV-RT: Operator not supported')
    # end
    if comparison_op_string == '>':
        outmat = refmat > searchmat;
    elif comparison_op_string == '>=':
        outmat = refmat >= searchmat;
    elif comparison_op_string == '<':
        outmat = refmat < searchmat;
    elif comparison_op_string == '<=':
        outmat = refmat <= searchmat;
    elif comparison_op_string == '==':
        outmat = refmat == searchmat;
    elif comparison_op_string == '~=':
        outmat = refmat != searchmat;
    else:
        raise Exception('UAV-RT: Operator not supported')

    return outmat