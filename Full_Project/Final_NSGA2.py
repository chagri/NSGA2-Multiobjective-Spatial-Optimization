"""

This Code takes in three files:
    1) File 1: called as Adj_List_file
    2) File 2: called as Main_data_file
    3) File 3: called as Tarets_File


"""


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
import sys

class Benchmark(object):
        #default methods taken from the library

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

                # Column numbers of the Main_data_file: These are [the land type of various cell in 5 different scenarios: A-E
                A_LU=3
                B_LU=4
                C_LU=5
                D_LU=6
                E_LU=7
                cSUIT=11
                cNoChange=9 # It is Whether the land allocation changed w.r.t the base case: 0: implies change
                minA = {} # Minimum area for each land type (1-25)
                maxA = {} # Max area for each land type
                suit=[]
                olt = {}
                olt2nlt = {} # Original land type to number : Basically key value, key is the land type, value is id allotted to it
                adj_List = {} # Adjacency list for each cell
                dim=0
                area = {}
                sumIndArea = [] # Sum of area for each type of land type (not cell), total 25
                indxMap = {}
                areaMap = {} # area for each cell
                landType = [] 

                generator_count = 0
                
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
                f= open("Adj_List_file.gal","r")
                data = f.read().splitlines()[1:]
                for line in data:
                        l = line.split(' ')
                        if not l[0] is '':
                                k = int(l[0])
                                for i in range(len(l)):
                                        #print i, k
                                        j = int(l[i])
                                        if (i==0):
                                                # Adding the key, i.e. the cell for which adj_list is being obtained, adds for the first time, when it does not exist int he dictionary
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
                g=open("Main_data_file.csv","r")
                
                # initiating the total sums for each landtype to zero
                for i in range(0,25):
                        sumIndArea.append(0)
                datag = g.readlines()[1:]
                for line in datag:
                        line = line.split('\t')
                        pt = int(line[0]) 
                        ar = float(line[2]) #area
                        tee = line[1] # landtype
                        temp_lt = olt2nlt[tee] 
                        landType.append(temp_lt)
                        # indxMap : contains decision (change/nochange), and four other preferred land types
                        indxMap[pt] = (int(line[cNoChange]),olt2nlt[line[B_LU]] ,olt2nlt[line[C_LU]],olt2nlt[line[D_LU]],olt2nlt[line[E_LU]])
                        areaMap[pt] = ar
                        sumIndArea[temp_lt] += ar 
                
                copyLandType = landType
                copySumIndArea = sumIndArea


                area_sum=0.00
                for i in range(43278):
                        if not i in areaMap:
                                areaMap[i]=0
                for pt in areaMap.keys():
                        area_sum = area_sum + areaMap[pt]


                        
                # Loading area targets
                gh = open("Tarets_File.csv","r")
                datagh = gh.readlines()[1:]
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
                self.bounder = ec.DiscreteBounder([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
                self.maximize = True # Maximizing criteria: i.e. maximizing the objectives
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
                self.generator_count = generator_count
                
                # http://people.hofstra.edu/geotrans/eng/ch6en/conc6en/ch6c2en.html
                
                
        # Know more about this class from the Readme of the package: detailed info there
        # Generates candidates in every generation
        def generator(self, random, args):
            p = random.uniform(0,1)

            # 80% of the times allow the randomness to choose any of the 4 land types
            # 20% of the times consider the Base case E
            if (p < 0.2):
                X = copy.copy(self.landType)
                for i in range(len(self.landType)):
                        if self.indxMap[i][0] == 0:
                            xchoice = 4
                            #self.sumIndArea[self.landType[i]] = self.sumIndArea[self.landType[i]] - self.areaMap[i]
                            X[i] = self.indxMap[i][xchoice] 
                        
                self.generator_count += 1
                # printing just to know whether job is executing for all the genrations, and changes happening
                print "1"
                return X

            else:
                X = copy.copy(self.landType)
                for i in range(len(self.landType)):
                        if self.indxMap[i][0] == 0:
                            xchoice = random.choice([1, 2, 3, 4])
                            #self.sumIndArea[self.landType[i]] = self.sumIndArea[self.landType[i]] - self.areaMap[i]
                            X[i] = self.indxMap[i][xchoice]
                            #self.sumIndArea[self.landType[i]] = self.sumIndArea[self.landType[i]] + self.areaMap[i]
                                
                return X

        # Evaluates the candidates, how each solution or population in a given generation is
        def evaluatorSol(self, c):
                # TOtal number of targets equals the land types
                numTarget = len(self.minA)

                f=[]    # List of all the objective value for each solution
                f1 = 0.00 #  Objective for contiguity
                
                objDir=[] # Boolean parameter for anyobjective to maximize or minimize: FALSE=> Minimize that objective
                # initiating the parameter for all the objectives
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
                        # if there is change in landtype with respect to basecase for that cell, recalculate the area for the landtypes (1-25) involved
                        # We let the changes happen only for the cells for which there was change wrt base case 
                        if self.indxMap[i][0] == 0:
                                # Because the land types have changed, reevaluate the area for that cell
                                # it is: Get landtype of that cell, update the change in area of that landtype because of change in landtype of this cell
                                sumIndArea[self.copyLandType[i]] = sumIndArea[self.copyLandType[i]] - self.areaMap[i]
                                sumIndArea[c[i]] = sumIndArea[c[i]] + self.areaMap[i]
                                
                                if c[i] == self.indxMap[i][4]:

                                        count +=1

                                if count == 6570:
                                        print "these are X:", sumIndArea

                        # We need to maximize this object: => more similar adjacent cells => more contiguity
                        for val in self.adjList[i]:
                                if c[i] == c[val]:
                                        f1 += 0.01
                
                #print count
                        
                
                #"""
                for ltidx in range(1,numTarget+1):
                        #print ltidx
                        objDir[ltidx]=False # minimizing objective which goes out of boundary
                        if (sumIndArea[ltidx] - self.minA[ltidx] < 0 ):
                                f[ltidx] = -(sumIndArea[ltidx] - self.minA[ltidx])
                        elif (sumIndArea[ltidx] - self.maxA[ltidx] > 0 ):
                                f[ltidx] = (sumIndArea[ltidx] - self.maxA[ltidx])
                        else: 
                                f[ltidx] = 0
#                                       f[ltidx] = (self.maxA[ltidx] + self.minA[ltidx])/2 - abs(sumIndArea[ltidx] - (self.maxA[ltidx] + self.minA[ltidx])/2 )
                

                self.landType = self.copyLandType # just in case to maintain the original values of the landtypes and sumIndexArea
                self.sumIndArea = self.copySumIndArea
                #print f
                #flist = [-i for i in f]
                #flist.append(-f1)
                f[0]=f1
                objDir[0]=True # maximizing the contiguity objective f1
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
                        # append fitness obtained from evaluatorSol method for a given candidate
                        fitness.append(par)

                
                #print "fintess are:", fitness  
                # returns fitness of each candidate combined as a list
                return fitness



def main(prng=None, display=False):
        # default params
        if prng is None:
                prng = Random()
                prng.seed(time()) 


        problem = NSGA2MOSO(43278)
        # total 43278 candidates

        # look into the package to know more about this method.
        # calling evolutionary component which has NSGA2 method
        ea = inspyred.ec.emo.NSGA2(prng)

        # Declaring the type of variator which we have used, can be changed according to user's choice: mutation and crossover rate; default blend_crossover, gaussian_mutation 
        ea.variator = [inspyred.ec.variators.blend_crossover, 
                                   inspyred.ec.variators.gaussian_mutation]

        # terminator used : generation_Termination: When counts of generations are reached
        ea.terminator = inspyred.ec.terminators.generation_termination
        
        # Defining problem, here it is landtype for each cell: evaluating how landtypes change and impact the objectives
        # X is the original base case, used for plotting
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

        # Running NSGA2 for population size of pop_size and generations max_generations (also terminating cond.)
        final_pop = ea.evolve(generator=problem.generator, 
                                                  evaluator=problem.evaluator, 
                                                  pop_size=60,
                                                  maximize=problem.maximize,
                                                  bounder=problem.bounder,
                                                  max_generations=60)
        
        # Plotting the solutions and base case (called as fGold, or gold solution)
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



