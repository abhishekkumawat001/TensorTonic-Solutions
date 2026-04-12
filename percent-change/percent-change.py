def percent_change(series):
    """
    Compute the fractional change between consecutive values.
    """
    series = np.asarray(series)
    pct = []
    for i in range(1, len(series)):

        prev = series[i-1]
        if prev ==0:
            pct.append(0.0)
        else:    
            
            change = (series[i] - prev)/prev
            pct.append(change)
            
    return pct