    function batch = select_batch(A, givenIndices, num_dim)
    
    if nargin < 3
        num_dim = ndims(A);
    end
    
    % for input array A returns A(:,:,...,:,lastIndices)
    % By Philipp Fischer, 2013 
    
        args = cell(1,ndims(A));
        args(:) = {':'};
        args(num_dim) = {givenIndices};
        batch = A(args{:});
        
    end
