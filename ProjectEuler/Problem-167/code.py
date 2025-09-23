def indicator(x):
    return 1 if x == 0 else 0

def ulam_term(v, t):
    base = 2
    step = 2 * v + 2

    # initialise sequence and membership flags
    seq = [base] + list(range(v, step, 2)) + [step]
    flags = [0] + [1 if k in seq else 0 for k in range(3, step, 2)]

    idx = (step - 1) // 2 + 1
    first_repeat = None

    while True:
        candidate = 2 * idx + 1
        mark = indicator(flags[idx - 1] - 1) + indicator(flags[idx - v - 1] - 1)

        if mark == 1:
            seq.append(candidate)

            if seq[-1] - seq[-2] == step:
                if first_repeat is None:
                    first_repeat = len(seq) - 2
                else:
                    second_repeat = len(seq) - 2
                    break

        flags.append(mark)
        idx += 1

    if t < second_repeat:
        return seq[t]

    period_len = second_repeat - first_repeat
    period_gap = seq[second_repeat] - seq[first_repeat]
    pos = (t - first_repeat) // period_len
    offset = (t - first_repeat) % period_len
    return pos * period_gap + seq[offset + first_repeat - 1]

total = 0
for n in range(2, 11):
    total += ulam_term(2 * n + 1, 10**11)

print(total)
