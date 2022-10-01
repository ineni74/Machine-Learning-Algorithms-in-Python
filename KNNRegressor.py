from collections import Counter

def mean(weights):
    return sum(weights) / len(weights)


def calc_minkowski_distance(a, b, p):
    return sum(abs(m1-m2)**p for m1, m2 in zip(a,b))**(1/p)


def knn(data, query, k, p):
    neighbor_distance_and_weights = []
    for index, example in enumerate(data):
        distance = calc_minkowski_distance(example[:-1],query, p)
        neighbor_distance_and_weights.append(distance, example[-1])
    
    sorted_neighbor_distance_and_weights = sorted(neighbor_distance_and_weights)
    k_nearest_distance_ans_weights = sorted_neighbor_distance_and_weights[:k]
    k_nearest_weights = [lable for distance, lable in k_nearest_distance_ans_weights]
    
    return k_nearest_distance_ans_weights, k_nearest_weights, mean(k_nearest_weights)

