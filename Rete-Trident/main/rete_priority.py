"""RETE's joint patch score and lazy variable-binding order.

Equation (1) in Parasaram, Barr and Mechtaev, ICSE 2023. Lower is better.
"""

import heapq
from itertools import count


def joint_score(template_distance, probabilities, theta=0.073):
    """Score a concrete binding of every hole in one template."""
    if template_distance < 0 or theta <= 0 or not probabilities:
        raise ValueError("distance must be nonnegative; theta and hole count positive")
    if any(not 0 < probability <= 1 for probability in probabilities):
        raise ValueError("variable probabilities must lie in (0, 1]")
    return template_distance + theta * sum(1 / p for p in probabilities) / len(probabilities)


def ranked_bindings(options, template_distance, theta=0.073):
    """Yield (score, values) without materialising the Cartesian product.

    Each hole's options are (value, probability) pairs. The highest-probability
    option for each hole starts the search; neighbouring index tuples are added
    as their predecessors are visited.
    """
    if not options or any(not choices for choices in options):
        return
    ordered = [sorted(choices, key=lambda item: (-item[1], str(item[0])))
               for choices in options]
    initial = (0,) * len(ordered)
    seen = {initial}
    sequence = count()

    def entry(indices):
        probabilities = [ordered[i][index][1] for i, index in enumerate(indices)]
        return (joint_score(template_distance, probabilities, theta),
                next(sequence), indices)

    queue = [entry(initial)]
    while queue:
        score, _, indices = heapq.heappop(queue)
        yield score, tuple(ordered[i][index][0] for i, index in enumerate(indices))
        for hole, index in enumerate(indices):
            if index + 1 >= len(ordered[hole]):
                continue
            successor = indices[:hole] + (index + 1,) + indices[hole + 1:]
            if successor not in seen:
                seen.add(successor)
                heapq.heappush(queue, entry(successor))
