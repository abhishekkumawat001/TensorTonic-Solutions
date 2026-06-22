def mean_rating_imputation(ratings_matrix, mode):
    """
    Fill missing ratings (zeros) with user or item means.
    """
    result = [row[:] for row in ratings_matrix]  # copy matrix
    
    if mode == "user":
        for i in range(len(result)):
            non_zero = [x for x in result[i] if x != 0]
            mean_val = sum(non_zero) / len(non_zero) if non_zero else 0
    
            for j in range(len(result[i])):
                if result[i][j] == 0:
                    result[i][j] = mean_val
    
    elif mode == "item":
        rows = len(result)
        cols = len(result[0]) if rows > 0 else 0
    
        for j in range(cols):
            non_zero = [result[i][j] for i in range(rows) if result[i][j] != 0]
            mean_val = sum(non_zero) / len(non_zero) if non_zero else 0
    
            for i in range(rows):
                if result[i][j] == 0:
                    result[i][j] = mean_val
    
    return result