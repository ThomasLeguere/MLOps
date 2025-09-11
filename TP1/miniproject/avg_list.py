def avg_list(l: list[int]) -> int:
    l_sum = 0

    for k in range(len(l)):
        l_sum += l[k]

    return l_sum / len(l)
