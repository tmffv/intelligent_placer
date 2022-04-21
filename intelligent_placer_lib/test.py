from lib import is_inside

def main():
    for i in range(1, 10):
        is_inside("inputImages/%i.jpeg" % i)
    return


if __name__ == '__main__':
    main()
