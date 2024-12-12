import numpy as np

def find_neighbors_in_two_lists(keys_with_single_value, list_to_find_neighbors_in):
    '''Iterate over each item in first list to find a pair in the second list'''
    neighbors = []
    for i in keys_with_single_value:
        for j in [x for x in list_to_find_neighbors_in if x != i]:
            if i+1 == j:
                neighbors.append(i)
                neighbors.append(j)
            if i-1 == j:
                neighbors.append(i)
                neighbors.append(j)
    return neighbors

def find_keys_without_neighbor(neighbors):
    '''Find integers in list without pair (consecutive increment of + or - 1 value) in the same list'''
    no_pair = []
    for i in neighbors:
        if i + 1 in neighbors:
            continue
        elif i - 1 in neighbors:
            continue
        else:
            no_pair.append(i)
    return no_pair

def not_need_further_execution(y_binned_count):
    '''Check if there are bins with single value counts'''
    return 1 not in y_binned_count.values()


def record_match(raw_graph, output):
    total = np.sum(raw_graph['solutions'])
    matched = 0
    unmatched = 0

    mapping = {}
    for i in range(raw_graph['solutions'].shape[0]):
        if raw_graph['solutions'][i] == True and output[i] == 1:
            start_vessel_class = raw_graph['vertex_labels'][i][0]
            target_vessel_class = raw_graph['vertex_labels'][i][1]
            matched += 1
            mapping[start_vessel_class] = target_vessel_class
        elif raw_graph['solutions'][i] == False and output[i] == 1:
            # NOT MATCHED
            unmatched += 1
            start_vessel_class = raw_graph['vertex_labels'][i][0]
            target_vessel_class = raw_graph['vertex_labels'][i][1]
            mapping[start_vessel_class] = target_vessel_class

    return mapping, total, matched, unmatched
