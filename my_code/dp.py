def copy(v, gamma, epsilon=1e-4):
    new_V = v.copy()

    cnt = 0

    while True:
        new_V["L1"] = 0.5 * (-1 + gamma * v["L1"]) + 0.5 * (1 + gamma * v["L2"])
        new_V["L2"] = 0.5 * (0 + gamma * v["L2"]) + 0.5 * (1 + gamma * v["L1"])

        delta = max(abs(new_V["L1"] - v["L1"]), abs(new_V["L2"] - v["L2"]))
        v = new_V.copy()
        cnt += 1

        if delta < epsilon:
            print(v)
            print(cnt)
            break


def overwrite(v, gamma, epsilon=1e-4):
    cnt = 0

    while True:
        t = 0.5 * (-1 + gamma * v["L1"]) + 0.5 * (1 + gamma * v["L2"])
        delta = abs(t - v["L1"])
        v["L1"] = t

        t = 0.5 * (0 + gamma * v["L2"]) + 0.5 * (1 + gamma * v["L1"])

        cnt += 1

        if delta < epsilon:
            print(v)
            print(cnt)
            break
