if __name__ == '__main__':
    with open('1blm.test.gt') as f:
        x = f.readlines()
    with open('1blm.test.noise') as f:
        y = f.readlines()

    col = 0
    for i, j in zip(x, y):
        if len(i.split(' ')) != len(j.split(' ')):
            col += 1
            print(':(', col)
            print(i)
            print(j)
