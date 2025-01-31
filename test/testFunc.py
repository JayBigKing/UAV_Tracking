from Jay_Tool.EfficiencyTestTool.EfficiencyTestTool import clockTester

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

def test4Help(x):
    return x**2, x**x

def test4():
    retVal = test4Help(1)
    print(retVal[0])


def test5():
    test5Sum()
    test5Reduce()

@clockTester
def test5Sum():
    print(sum([(i - 5) for i in range(100000)]))
@clockTester
def test5Reduce():
    from functools import reduce
    print(reduce(lambda a,b: (a-5) + (b - 5), [i for i in range(1, 100000)]))

def main():
    test5()

if __name__ == "__main__":
    main()