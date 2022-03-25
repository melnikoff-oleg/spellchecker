import random

if __name__ == '__main__':

    with open('1blm/1blm.test.noise', 'w') as f:
        with open('test.1blm.noise.prob') as g:
            for line in g:
                f.write(line)
        with open('test.1blm.noise.word') as g:
            for line in g:
                f.write(line)
        with open('test.1blm.noise.random') as g:
            for line in g:
                f.write(line)

    with open('1blm/1blm.test.gt', 'w') as f:
        with open('test.1blm') as g:
            for line in g:
                f.write(line)
        with open('test.1blm') as g:
            for line in g:
                f.write(line)
        with open('test.1blm') as g:
            for line in g:
                f.write(line)

    with open('1blm/1blm.test.gt') as f:
        x = f.readlines()
    with open('1blm/1blm.test.noise') as g:
        y = g.readlines()


    with open('1blm.test.gt.real', 'w') as f:
        with open('1blm.test.noise.real', 'w') as g:
            z = list(zip(x, y))
            random.shuffle(z)
            for i, j in z:
                f.write(i)
                g.write(j)
