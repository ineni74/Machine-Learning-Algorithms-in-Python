from collections import Counter

def mode_func(lables):
    return Counter(lables).most_common(1)[0][0]


def calc_minkowski_distance(a, b, p):
    return sum(abs(m1-m2)**p for m1, m2 in zip(a,b))**(1/p)


def knn(data, query, k, p):
    neighbor_distance_and_lables = []
    for index, example in enumerate(data):
        distance = calc_minkowski_distance(example[:-1],query, p)
        neighbor_distance_and_lables.append(distance, example[-1])
    
    sorted_neighbor_distance_and_lables = sorted(neighbor_distance_and_lables)
    k_nearest_distance_ans_lables = sorted_neighbor_distance_and_lables[:k]
    k_nearest_lables = [lable for distance, lable in k_nearest_distance_ans_lables]
    
    return k_nearest_distance_ans_lables, k_nearest_lables, mode_func(k_nearest_lables)

