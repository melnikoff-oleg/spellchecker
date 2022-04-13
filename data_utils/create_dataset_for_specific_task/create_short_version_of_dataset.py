

if __name__ == '__main__':

    length = 50

    with open('bea60k.gt') as f:
        with open(f'bea{length}.gt', 'w') as g:
            ind = 0
            for line in f:
                g.write(line)
                ind += 1
                if ind == length:
                    break

    with open('bea60k.noise') as f:
        with open(f'bea{length}.noise', 'w') as g:
            ind = 0
            for line in f:
                g.write(line)
                ind += 1
                if ind == length:
                    break
