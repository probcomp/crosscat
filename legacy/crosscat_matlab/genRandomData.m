function [data, columns_to_views, rows_in_views_to_cats] = genRandomData(n_cells, n_cols, n_rows)

    
    n_views = numel(n_cells);

    assert(n_cols >= n_views)
    
    columns_to_views = shuffle([1:n_views, randi(n_views,1,n_cols-n_views)]);
    
    rows_in_views_to_cats = zeros(n_views,n_rows);
    
    for view = 1:n_views
        nc = n_cells(view);
        rows_in_views_to_cats(view,:) = shuffle([1:nc,randi(nc,1,n_rows-nc)]);
    end

    % generate a figure of the partitioning (sort the values, generate data)
    sorted_views = sort(columns_to_views);
    sorted_cells = [];
    for view = 1:n_views
        sorted_cells = [sorted_cells; sort(rows_in_views_to_cats(view,:))];
    end
    
    pcolor(genDataFromPartitions(sorted_views, sorted_cells,.1))
    
    data = genDataFromPartitions(columns_to_views, rows_in_views_to_cats);
end