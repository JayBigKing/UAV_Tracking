import collections
def test1():
    DEFAULT_EC_ARGS2 = {
        "floatMutationOperateArg": 0.8,
        "floatCrossoverAlpha": 0.5
    }
    dea = (collections.defaultdict(list))
    dea.update(DEFAULT_EC_ARGS2)
    print(dea)
    print(dea["hh"] == [])
    DEFAULT_EC_ARGS2["floatCrossoverAlpha"] = 0.09
    print(dea["floatCrossoverAlpha"])


def test2():
    selectIndexSet = set()
    selectIndexSet.add(1)
    selectIndexSet.update({2,3,4,5,6,999})
    print(selectIndexSet)
    for i,index in enumerate(selectIndexSet):
        print("{}:{}".format(i, index))

def test3():
    d0 = {"h":1, "u":2}
    print(len(d0))
    print(isinstance(d0,list))

def main():
    test3()

if __name__ == "__main__":
    main()