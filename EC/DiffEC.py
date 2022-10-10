import random
from abc import ABC, abstractmethod
import EC_Base
# from copy import deepcopy


class DiffEC(EC_Base, ABC):
    '''
    differential evolution compution
    '''

    def __init__(self, n, dimNum, maxConstraint, minConstraint, needEpochTimes, useCuda=True):
        self.baseInit(n, dimNum, maxConstraint, minConstraint, needEpochTimes, useCuda)

    def mutation(self, Fg):
        if self.useCuda is False:
            for i in range(0, self.Np):
                r1, r2, r3 = random.sample(range(self.Np), 3)
                for j in range(0, self.dimNum):
                    self.middleChromosomes[j, i] = min(max(self.chromosomes[j, r1] + Fg * (
                            self.chromosomes[j, r2] - self.chromosomes[j, r3]), self.minConstraint[j]),
                                                       self.maxConstraint[j])

    def crossover(self, CRgList):
        if self.useCuda is False:
            for i in range(0, self.Np):
                for j in range(0, self.dimNum):
                    if random.random() > CRgList[i]:
                        self.middleChromosomes[j, i] = self.chromosomes[j, i]

    @abstractmethod
    def cmpFitting(self,val1, val2):
        pass

    @abstractmethod
    def fitting(self, index):
        pass

    def choose(self):
        if self.useCuda is False:
            pass
        else:
            self.chromosomes = self.chromosomes_device.copy_to_host()
            self.middleChromosomes = self.middleChromosomes_device.copy_to_host()
        for i in range(0, self.Np):
            middleChromosome = self.middleChromosomes[:, i]
            middleChromosomeValue = self.fitting(i)
            if self.cmpFitting(middleChromosomeValue, self.chromosomesVaule[i]) > 0:
                self.chromosomesVaule[i] = middleChromosomeValue
                for j in range(self.dimNum):
                    self.chromosomes[j, i] = middleChromosome[j]
                if self.cmpFitting(middleChromosomeValue, self.chromosomesVaule[self.bestChromosomeIndex]) > 0:
                    self.bestChromosomeIndex = i

    # @abstractmethod
    # def computing(self):
    #     pass

