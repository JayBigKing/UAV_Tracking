def test1Help(*args, **kwargs):
    print("args : %r, \n"
          "kwargs : %r" % (args, kwargs))
    print(args == ())
    print(kwargs == {})
def test1():
    test1Help()

def main():
    test1()

if __name__ == "__main__":
    main()