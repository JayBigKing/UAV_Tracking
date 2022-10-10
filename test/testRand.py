import numpy as np
import random
def test1():
    for i in range(5):
        print(np.random.choice([-1, 1]))

def test2():
    t1 = {"niu":1,}
    t2 = {"h":2,}
    t3 = t2["niu"] if t2.get("niu") else t1["niu"]
    print(t3)

def test3():
    r0 = random.sample(range(10), 2)
    r1, r2 = random.sample(range(10), 2)
    r3 = np.random.choice(range(10), replace=False, size=10)
    print(r0)
    print(r1)
    print(r2)
    print(r3)

def test4():
    print(np.random.rand(2))
    print(np.random.randn(200))

def main():
    test4()

if __name__ == '__main__':
    main()