from enum import Enum


class EC_CodingType2(Enum):
    BINARY_CODING = 0
    GRAY_CODING = 1
    FLOAT_CODING = 2
    SYMBOL_CODING = 4
    OTHER_CODING = 5

class TestShareFunc:
    def __init__(self):
        self.hh = 0
    def shareFunc(self):
        self.hh += 1
    def seeAttribute(self):
        print(self.__getattribute__('hh'))
    def __getattr__(self, item):
        print('__getattr__')

def test1Helper(func):
    for i in range(10):
        func()

def test1():
    t0, t1 = TestShareFunc(), TestShareFunc()
    test1Helper(t0.shareFunc)
    print("t0' hh is %d, while t1's hh is %d" % (t0.hh, t1.hh))

def test2():
    t1 = TestShareFunc()
    t1.seeAttribute()
    a = t1.x
    print(hasattr(TestShareFunc, 'hh'))


def main():
    test2()

if __name__ == '__main__':
    main()


