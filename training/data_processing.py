
def read_data(gt_path, noise_path):
    data = []
    with open(gt_path) as f:
        gt = f.readlines()
    with open(noise_path) as f:
        noise = f.readlines()
    for i, j in zip(noise, gt):
        data.append(tuple([i[:-1], j[:-1]]))
    return data


def read_data_char_based(gt_path, noise_path):
    data = []
    with open(gt_path) as f:
        gt = f.readlines()
    with open(noise_path) as f:
        noise = f.readlines()
    for i, j in zip(noise, gt):
        data.append(tuple([i[:-1], j[:-1]]))
    for ind, i in enumerate(data):
        data[ind] = (i[0].replace(' ', '_'), i[1].replace(' ', '_'))
    return data
