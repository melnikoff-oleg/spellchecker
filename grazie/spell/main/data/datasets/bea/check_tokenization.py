if __name__ == '__main__':
    with open('bea500.gt') as f:
        x = f.readlines()
    with open('bea500.noise') as f:
        y = f.readlines()

    col = 0
    for i, j in zip(x, y):
        if len(i.split(' ')) != len(j.split(' ')):
            col += 1
            print(':(', col)
            print(i)
            print(j)
