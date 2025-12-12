function Ispline(x, spl_order, knots)
    k = spl_order + 1  
    num_knots = length(knots)
    num_basis = num_knots - 2 + k    

    extended_knots = vcat(
        fill(knots[1], k), 
        knots[2:(num_knots-1)], 
        fill(knots[num_knots], k)
    )
    
    basis_functions = zeros(num_basis + k - 1, length(x)) 
    for i = k:num_basis 
        basis_functions[i, :] = (extended_knots[i] .<= x .< extended_knots[i+1]) ./ 
                                (extended_knots[i+1] - extended_knots[i])
    end
    
    rec_basis = basis_functions
    for recursion_level = 1:spl_order
        next_basis = zeros(num_basis + k - 1 - recursion_level, length(x))
        for i = (k - recursion_level):num_basis
            factor = (recursion_level + 1) / recursion_level
            knot_span = extended_knots[i + recursion_level + 1] - extended_knots[i]
            
            left_term = (x .- extended_knots[i]) .* rec_basis[i, :]
            right_term = (extended_knots[i + recursion_level + 1] .- x) .* rec_basis[i + 1, :]
            
            next_basis[i, :] = factor * (left_term + right_term) ./ knot_span
        end
        rec_basis = next_basis
    end
    
    # Calculate knot indices for each evaluation point
    knot_indices = zeros(length(x))
    for i = 1:size(x)[1]
        knot_indices[i] = sum(extended_knots .<= x[i])
    end
    
    ispline_basis = zeros(num_basis - 1, length(x))

    if spl_order == 1 
        for i = 2:num_basis
            ispline_basis[i - 1, :] = (i .< knot_indices .- spl_order .+ 1) .+ 
                            (i .== knot_indices) .* 
                            (extended_knots[i + spl_order + 1] - extended_knots[i]) .* 
                            rec_basis[i, :] ./ (spl_order + 1)
        end
    else
        for point_idx = 1:length(x)
            for basis_idx = 2:num_basis
                if basis_idx < (knot_indices[point_idx] - spl_order + 1)
                    ispline_basis[basis_idx - 1, point_idx] = 1
                elseif basis_idx >= (knot_indices[point_idx] - spl_order + 1) && 
                       basis_idx <= knot_indices[point_idx]
                    start_idx = basis_idx
                    end_idx = Int(knot_indices[point_idx])
                    
                    left_knots = extended_knots[start_idx:end_idx]
                    right_knots = extended_knots[(basis_idx + spl_order + 1):Int(knot_indices[point_idx] + spl_order + 1)]
                    
                    ispline_basis[basis_idx - 1, point_idx] = sum((right_knots - left_knots) .* 
                            rec_basis[start_idx:end_idx, point_idx]) / (spl_order + 1)
                else
                    ispline_basis[basis_idx - 1, point_idx] = 0
                end
            end
        end
    end
    
    return ispline_basis
end

function Mspline(x, spl_order, knots)
    k = spl_order 
    num_knots = length(knots)  
    num_basis = num_knots - 2 + k    
    
    extended_knots = vcat(
        fill(knots[1], k), 
        knots[2:(num_knots-1)], 
        fill(knots[num_knots], k)
    )
    
    basis_functions = zeros(num_basis + k - 1, length(x)) 
    for i = k:num_basis 
        basis_functions[i, :] = (extended_knots[i] .<= x .< extended_knots[i+1]) ./ 
                              (extended_knots[i+1] - extended_knots[i])
    end
    
    if spl_order == 1
        return basis_functions
    end
    
    rec_basis = basis_functions
    for recursion_level = 1:(spl_order-1)
        next_basis = zeros(num_basis + k - 1 - recursion_level, length(x))
        for i = (k - recursion_level):num_basis
            factor = (recursion_level + 1) / recursion_level
            knot_span = extended_knots[i + recursion_level + 1] - extended_knots[i]
            
            left_term = (x .- extended_knots[i]) .* rec_basis[i, :]
            right_term = (extended_knots[i + recursion_level + 1] .- x) .* rec_basis[i + 1, :]
            
            next_basis[i, :] = factor * (left_term + right_term) ./ knot_span
        end
        rec_basis = next_basis
    end
    
    return rec_basis
end