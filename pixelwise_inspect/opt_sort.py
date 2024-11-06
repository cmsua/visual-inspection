import os

def opt_sort(optimal_threshold, values):

    opt_good = []
    opt_flagged = []

    # Flag segments according to optimal threshold
    for i, value in enumerate(values):
          
          if value[i][0] < optimal_threshold:
                opt_flagged.append(values[i])
                continue
          else:
                opt_good.append(values[i])
                continue
          
    return opt_good, opt_flagged
