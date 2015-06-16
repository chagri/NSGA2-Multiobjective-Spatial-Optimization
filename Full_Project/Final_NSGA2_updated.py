from random import Random
from time import time
import inspyred
import copy
from inspyred import ec
from inspyred.ec import emo
from inspyred.ec import selectors
from inspyred.ec import variators
from inspyred import swarm
import itertools
import math
import random

class Benchmark(object):
        
        def __init__(self, dimensions, objectives=1):
                self.dimensions = dimensions
                self.objectives = objectives
                self.bounder = None
                self.maximize = True
                
        def __str__(self):
                if self.objectives > 1:
                        return '{0} ({1} dimensions, {2} objectives)'.format(self.__class__.__name__, self.dimensions, self.objectives)
                else:
                        return '{0} ({1} dimensions)'.format(self.__class__.__name__, self.dimensions)
                
        def __repr__(self):
                return self.__class__.__name__
        
        def generator(self, random, args):
                """The generator function for the benchmark problem."""
                raise NotImplementedError
                
        def evaluator(self, candidates, args):
                """The evaluator function for the benchmark problem."""
                raise NotImplementedError
                
        def __call__(self, *args, **kwargs):
                candidate = [a for a in args]
                fit = self.evaluator([candidate], kwargs)
                return fit[0]

class NSGA2MOSO(Benchmark):
        
        def __init__(self, dimensions=43278):
                A_LU=3
                B_LU=4
                C_LU=5
                D_LU=6
                E_LU=7
                cSUIT=11
                cNoChange=9
                #id2idx = []
                #idx2id = []
                id2idx = {}
                idx2id = {}
                id2idx2 = {}
                idx2id2 ={}
                minA = {}
                maxA = {}
                suit=[]
                olt = {}
                for i in range(50000):
                        suit.append(0)
                        #olt.append(0)
                '''
                        0: AIR: Airport, 1: CIV: Civic, 2:POS: Preserved Open Space

                        3:WF, Working Farm
                        3:RL, Rural Living
                        3:RCR, Rural Crossroads

                        4:ER, Estate Residential
                        4:LLRN, Large Lot Residential Neighborhood

                        5:SLRN, Small Lot Residential Neighborhood
                        5:MFRN, Multifamily Residential Neighborhood
                        5:MRN, Mixed Residential Neighborhood
                        5:MHP

                        6:HRR, Healthcare

                        7:NCC, Neighborhood Commercial Center
                        7:SCC, Suburban Commercial Corridor
                        7:SOC, Suburban Office Center

                        8:LIC, Light Industrial
                        8:HIC, Heavy Industrial
                        8:HCC, Highway Commercial Corridor

                        9:MUN, Mixed-Use Neighborhood
                        9:TC, Town Center
                        9:UN, Urban Neighborhood
                        9:TOD, Transit Oriented Development
                        9:MUC, Mixed Use Center
                        9:MC, Metropolitan Center
                        9:UC, Urban Center
                '''
                olt2nlt = {}
                olt2nlt["AIR"] = 1
                olt2nlt["CIV"] = 2
                olt2nlt["HIC"] = 3
                olt2nlt["LIC"] = olt2nlt["HCC"] = 4
                olt2nlt["LLRN"] = olt2nlt["ER"] = 5
                olt2nlt["MFRN"] = 6
                olt2nlt["MHP"] = 7
                olt2nlt["MRN"] = 8
                olt2nlt["MUN"] =  9
                olt2nlt["NCC"] = 10
                olt2nlt["POS"]= 11
                olt2nlt["RCR"]= 12
                olt2nlt["RL"] = 13
                olt2nlt["SCC"]= 14
                olt2nlt["SLRN"] = 15
                olt2nlt["SOC"] = 16
                olt2nlt["TC"] = 17
                olt2nlt["UC"] = 18 
                olt2nlt["UN"] = 19
                olt2nlt["VC"] = olt2nlt["HRR"] = 20
                olt2nlt["WF"] = 21
                olt2nlt["MUC"] = 22
                olt2nlt["MC"] = 23
                olt2nlt["TOD"] = 24


                #Loading adj list for each cell
                f= open("Adj_List_Final.gal","r")
                data = f.read().splitlines()[1:]
                adj_List = {}
                for line in data:
                        l = line.split(' ')
                        if not l[0] is '':
                                k = int(l[0])
                                for i in range(len(l)):
                                        #print i, k
                                        j = int(l[i])
                                        if (i==0):
                                                if not k in adj_List:
                                                        adj_List[k]=[]
                                        else:
                                                adj_List[k].append(j)
                                        
                                        if not j in adj_List:
                                                adj_List[j] =[]
                                        if not k in adj_List[j]:
                                                adj_List[j].append(k)  
                #print adj_List[20][1]
                

                #Loading areas, landtype for each cell
                g=open("Refined_Swas_E_Updated2.csv","r")
                dim=0
                area = {}
                sumIndArea = []
                for i in range(0,25):
                        sumIndArea.append(0)
                #print sumIndArea[21]
                datag = g.readlines()[1:]
                indxMap = {}
                areaMap = {}
                landType = []
                for line in datag:
                        line = line.split('\t')
                        pt = int(line[0]) 
                        ar = float(line[2]) #area
                        tee = line[1] # landtype
                        temp_lt = olt2nlt[tee] 
                        landType.append(temp_lt)
                        # indxMap : contains decision (change/nochange), and four preferred land types
                        indxMap[pt] = (int(line[cNoChange]),olt2nlt[line[B_LU]] ,olt2nlt[line[C_LU]],olt2nlt[line[D_LU]],olt2nlt[line[E_LU]])
                        areaMap[pt] = ar
                        #print temp_lt
                        sumIndArea[temp_lt] += ar 
                #print indxMap[20]
                
                copyLandType = landType
                copySumIndArea = sumIndArea



                area_sum=0.00
                for i in range(43278):
                        if not i in areaMap:
                                areaMap[i]=0
                for pt in areaMap.keys():
                        area_sum = area_sum + areaMap[pt]


                        
                # Loading area targets
                gh = open("SWAS_E_LU_PERCENT.csv","r")
                datagh = gh.readlines()
                for line in datagh:
                        line = line.split('\t')
                        #print line[0], line[1], line[2]
                        minA[int(line[0])] = float(line[1]) * area_sum
                        maxA[int(line[0])] = float(line[2]) * area_sum
                        #print line[0], minA[int(line[0])], maxA[int(line[0])], area_sum



                #43151
                self.dimensions = dim
                self.objectives = 25
                Benchmark.__init__(self, dimensions, 25) #numobj=3, dimensions=numcells=6
                #self.bounder = ec.Bounder([0] * self.dimensions, [3.0] * self.dimensions)
                self.bounder = ec.DiscreteBounder([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
                self.maximize = True

                #self.candidatesArea = {0:10,1:10,2:10,3:10,4:10,5:10}
                #self.total_area = 60
                #self.adjList = {0:[1,2],1:[0,2],2:[0,1,3],3:[2,4,5],4:[3,5],5:[3,4]}
                self.adjList = adj_List
                self.candidatesArea = area
                self.landType = landType
                self.copyLandType = copyLandType
                self.areaMap = areaMap
                self.indxMap = indxMap
                self.sumIndArea = sumIndArea
                self.copySumIndArea = copySumIndArea
                self.total_area = area_sum
                self.minA = minA
                self.maxA = maxA
                self.suit = suit
                
                # http://people.hofstra.edu/geotrans/eng/ch6en/conc6en/ch6c2en.html
                
                

        def generator(self, random, args):
                

                X = copy.copy(self.landType)
                for i in range(len(self.landType)):
                        if self.indxMap[i][0] == 0:
                                xchoice = random.choice([1, 2, 3, 4])
                                #self.sumIndArea[self.landType[i]] = self.sumIndArea[self.landType[i]] - self.areaMap[i]
                                X[i] = self.indxMap[i][xchoice]
                                #self.sumIndArea[self.landType[i]] = self.sumIndArea[self.landType[i]] + self.areaMap[i]
                                
                return X

        def evaluatorSol(self, c):
                
                numTarget = len(self.minA)

                f=[]
                f1 = 0.00 # for contiguity
                objDir=[]
                for i in range(numTarget+1):
                        f.append(0)
                        objDir.append(False)
                        
                count = 0
                tempSumArea = []
                for i in range(0,25):
                        tempSumArea.append(0)

                sumIndArea = copy.deepcopy(self.sumIndArea)
                p = range(len(c))
                for i in p:
                        
                        #print "i",i, "idx2id:", self.idx2id[i],"c[i]", c[i]

                        if self.indxMap[i][0] == 0:
                                sumIndArea[self.copyLandType[i]] = sumIndArea[self.copyLandType[i]] - self.areaMap[i]
                                sumIndArea[c[i]] = sumIndArea[c[i]] + self.areaMap[i]
                                
                                if c[i] == self.indxMap[i][4]:

                                        count +=1

                                if count == 6570:
                                        print "these are X:", sumIndArea

                        
                        for val in self.adjList[i]:
                                if c[i] == c[val]:
                                        f1 += 0.01
                
                #print count
                        
                
                #"""
                for ltidx in range(1,numTarget+1):
                        #print ltidx
                        objDir[ltidx]=False
                        if (sumIndArea[ltidx] - self.minA[ltidx] < 0 ):
                                f[ltidx] = -(sumIndArea[ltidx] - self.minA[ltidx])
                        elif (sumIndArea[ltidx] - self.maxA[ltidx] > 0 ):
                                f[ltidx] = (sumIndArea[ltidx] - self.maxA[ltidx])
                        else: 
                                f[ltidx] = 0
#                                       f[ltidx] = (self.maxA[ltidx] + self.minA[ltidx])/2 - abs(sumIndArea[ltidx] - (self.maxA[ltidx] + self.minA[ltidx])/2 )
                

                self.landType = self.copyLandType
                self.sumIndArea = self.copySumIndArea
                #print f
                #flist = [-i for i in f]
                #flist.append(-f1)
                f[0]=f1
                objDir[0]=True
                f = [f1, sum(f[1:])]
                objDir = [True, False]
                #f[1] = f2
                #flist.append(-f)
                #print f

                return emo.Pareto(f,objDir)
                
        def evaluator(self, candidates, args):
                fitness = []
                
                
                for c in candidates:
                        par = self.evaluatorSol(c)
                        fitness.append(par)

                
                #print "fintess are:", fitness  
                return fitness

def main(prng=None, display=False):
        if prng is None:
                prng = Random()
                prng.seed(time()) 

        #problem = inspyred.benchmarks.Kursawe(3)
        problem = NSGA2MOSO(43278)
        ea = inspyred.ec.emo.NSGA2(prng)
        ea.variator = [inspyred.ec.variators.blend_crossover, 
                                   inspyred.ec.variators.gaussian_mutation]
        ea.terminator = inspyred.ec.terminators.generation_termination
        
        X = problem.landType
        X_total_area = 0
        #X_area = 
        for i in range(len(problem.landType)):
                if problem.indxMap[i][0] == 0:
                        #self.sumIndArea[self.landType[i]] = self.sumIndArea[self.landType[i]] - self.areaMap[i]
                        X[i] = problem.indxMap[i][4]
                X_total_area += problem.areaMap[i]


        print "X_area total is ",X_total_area
                        #self.sumIndArea[self.landType[i]] = self.sumIndArea[self.landType[i]] + self.areaMap[i]
                        
        
        fGOLD = problem.evaluatorSol(X)
        print "fGold is", fGOLD

        final_pop = ea.evolve(generator=problem.generator, 
                                                  evaluator=problem.evaluator, 
                                                  pop_size=60,
                                                  maximize=problem.maximize,
                                                  bounder=problem.bounder,
                                                  max_generations=60)
        
        if display:
                final_arc = ea.archive
                print "final Solutions", len(final_arc)
                print('Best Solutions: \n')
                
                for f in range(len(final_arc)):
                        #print(i)
                        Sum = 0
                        for i in final_arc[f].fitness:
                                Sum += i
                        print Sum, final_arc[f].fitness

                        sol = final_arc[f].candidate
                        s = "out/sol" + str(f) + ".txt"
                        #p = "out_best_sol/sol" + str(f) + ".txt"
                        fl = open(s,"w")
                        for j in range(len(sol)):
                                fl.write(str(j))
                                fl.write("\t")
                                fl.write(str(sol[j]))
                                fl.write("\n")

                import pylab
                x = []
                y = []
                for f in final_arc:
                        x.append(f.fitness[0])
                        temp=0
                        for i in range(1,len(f.fitness)):
                                temp += f.fitness[i]
                        
                        y.append(temp)
                        #y.append(f.fitness[1])
                pylab.scatter(x, y, color='b')
                x1 = fGOLD[0]
                x2 = 0
                for i in range(1,len(fGOLD)):
                        x2 += fGOLD[i]
                #print x1,x2
                #print "test"
                pylab.scatter(x1,x2,color='r')
                pylab.savefig('{0} Example ({1}).pdf'.format(ea.__class__.__name__, 
                                                                                                         problem.__class__.__name__), 
                                          format='pdf')
                pylab.show()
        return ea
                
if __name__ == '__main__':
        main(display=True)    



