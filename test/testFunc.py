def test1Help(*args, **kwargs):
    print("args : %r, \n"
          "kwargs : %r" % (args, kwargs))
    print(args == ())
    print(kwargs == {})
def test1():
    test1Help()

def test2():
    dict0 = {"cmpFunc" : lambda a, b : a - b,}
    print(dict0["cmpFunc"](10, 5))

def test3Help(a, b, c):
    print("1 : %r, 2 : %r, 3 : %r" %(a, b, c) )

def test3():
    from operator import methodcaller
    test3Tester = methodcaller("test3Help", 2., 30)
    test3Tester(0.1)
def main():
    test3()

if __name__ == "__main__":
    main()