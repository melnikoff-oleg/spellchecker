if __name__ == '__main__':
    col = 0
    with open('1blm.train.noise') as f:
        for line in f:
            col += 1
    print(col)
    with open('1blm.test.noise') as f:
        for line in f:
            col += 1
    print(col, 'gtgh')

# 4097007
# 4916407