function data = genDataFromPartitions(columns_to_views, rows_in_views_to_cats, cell_std, cell_spacing)
% genDataFromPartitions generate data from partitions
% inputs:
%   columns_to_views: a vector of length n_cols, where each entry assigns
%       the column i to a view
%   rows_in_views_to_cats: a n_views by n_rows matrix where entry (i,j)
%       assigns row j in view i to a category.
%   cell_std: (Optional) the standard deviation of the data in each cell.
%       default is 0.5.
%   cell_spacing: (Optional) the space between cell means. Default is 2.
% outputs:
%   data: a n_rows by n_cols matrix of data

    n_views = max(columns_to_views);
    
    % make sure the partitions agree
    assert(n_views == size(rows_in_views_to_cats,1))
    
    % get number of rows and columns
    n_cols = numel(columns_to_views);
    n_rows = size(rows_in_views_to_cats,2);
    
    data = zeros(n_rows,n_cols);
    
    if nargin < 4, cell_spacing = 2; end
    if nargin < 3, cell_std = .5; end
    
    for col = 1:n_cols
        view = columns_to_views(col);
        n_cells = max(rows_in_views_to_cats(view,:));
        for cell = 1:n_cells
            data_mean = (cell-1)*cell_spacing;
            row_indices = find(rows_in_views_to_cats(view,:)==cell);
            data(row_indices,col) = normrnd(data_mean, cell_std, numel(row_indices),1);
        end
    end
    
    
end