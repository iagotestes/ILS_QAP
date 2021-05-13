# -*- coding: utf-8 -*-

"""**QPA**"""

import math
import secrets
import numpy as np
import os
import subprocess
from matplotlib import pyplot as plt
# %matplotlib inline

def load_matrices(path_to_file):
  #file format:
  # n = order of the matrices
  # [[Flux]]
  # [[Distance]]
  raw_numbers = open(path_to_file, "r")

  #getting order of matrices
  matrix_size = list(map(int, raw_numbers.readline().split()))[0]
  print(matrix_size)

  F = []
  D = []
  flux_size, distance_size = (0,0)
  for x in raw_numbers:
    #put the numbers in the Flux and Distance matrices
    row = list(map(int, x.split()))
    if len(row) < (matrix_size/2 + 1):
      print("length = " +  str(len(row)) + " error reading line of the matrix: jumping line")
    else:
      #print(row)
      if flux_size < matrix_size :
        F.append(row)
        flux_size += 1
      elif distance_size < matrix_size :
        D.append(row)
        distance_size += 1
  raw_numbers.close()
  nF, nD = (np.array(F), np.array(D))
  return (matrix_size, nF, nD )

def cost_function(n, solution, F, D):
    cost=0
    for i in range(n):
        for j in range(n):
            cost += D[solution[i], solution[j]] * F[i][j]
    return cost

"""**ITERATE LOCAL SEARCH**
"""
#Procedure ILS
# s0 = GeneralInitialSolution
# s* = LocalSearch(s0)
# do 
#   s' = Perturbation(s*, history)
#   s*' = LocalSearch(s')
#   s* = AcceptanceCriterion(s*, s*', history)
# while condition
# end

def swap(a,b,vet):
  c = vet[a]
  vet[a] = vet[b]
  vet[b] = c

def local_search(n, history, s_vector, D ,F):
  cost = [] # cost of neighbors
  first = 0
  #find the first, this one will be the index for maping the neighbors
  while first < n :
    if history[first] == 0 : # immutable memory value
      first += 1
      continue
    else:
      break
  #find the cost of all the neighbors
  i = first + 1
  index_min_cost = 0
  while i < n :
    if history[i] == 1 :
       neighbor = s_vector.copy()
       swap(first,i,neighbor)
       value = cost_function(n, neighbor, F, D)
       cost.append(value)
       if min(cost) == value:  #keep track of the minimum cost's position
         index_min_cost = i
    i = i + 1
  #find the neighbor list with minimum cost
  min_cost = min(cost)
  neighbor = s_vector.copy()
  swap(first, index_min_cost, neighbor)
  new_history = history.copy()
  new_history[first]=0
  new_history[index_min_cost]=0
  return (new_history, min_cost, neighbor)

def perturbation(n, history, s_vector, D, F):
  indexes = [i for i in range(n) if history[i] == 1]
  a = secrets.choice(indexes)
  indexes.remove(a) # a != b
  b = secrets.choice(indexes)
  new_s = s_vector.copy()
  swap(a,b, new_s)
  new_cost = cost_function(n, new_s, F, D)    
  new_history = history.copy()
  return (new_history, new_cost, new_s)

def better(sa, sa1):
  return sa if sa[1] < sa1[1] else sa1

def rw(sa, sa1):
  return sa1

def lsmc(sa, sa1, history):
  if sa1[1] < sa[1] :
    return sa1
  else:
    T = history.count(1) if history.count(1) != 0 else 1 
    psa1 = math.exp((sa[1] - sa1[1])/T)
    pv = [1-psa1, psa1]
    return sa if np.random.choice([0,1], 1, pv)[0] == 0 else sa1

def acceptance_criterion(sa, sa1, history, flag):
  if flag == 0:
    return better(sa,sa1)
  elif flag == 1:
    return rw(sa, sa1)
  else:
    return lsmc(sa, sa1, history)  

def ils(n, D, F, criteria_flag):
  list_plot=[]
  history = [1 for i in range(n)] # history is nothing but a bit map
  s0 = [i for i in range(n)] #s identity is the general solution
  sa = local_search(n, history, s0, D, F)
  #do
  s1 = perturbation(n, sa[0], sa[2], D, F)
  sa1 = local_search(n, s1[0], s1[2], D, F)
  sa = acceptance_criterion(sa, sa1, sa[0], criteria_flag)
  while history.count(1) > 1 :
    s1 = perturbation(n, sa[0], sa[2], D, F)
    sa1 = local_search(n, s1[0], s1[2], D, F)
    sa = acceptance_criterion(sa, sa1, sa[0], criteria_flag)
    history = sa[0]
    list_plot.append(sa[1])
#         cost  result[]  plot[]  
  return (sa[1], sa[2], list_plot)
    
def timer():
    import time
    return time.time()

# execution tests
def execute_mean_test(cicles, path, criteria_flag):
    n, F, D = load_matrices(path)
   
    start = timer()
    result = ils(n, D, F, criteria_flag)
    end = timer()

    mean = result[2]
    best_result = result[1]
    best_cost = result[0]
    mean_cost = result[0]
    mean_time = end - start
    i=1
    while i < cicles:
            n, F, D = load_matrices(path)

            start = timer()
            result = ils(n, D, F, criteria_flag)
            end = timer()

            time = end - start
            mean_time = (time + mean_time) / 2 
            if result[0] < best_cost:
                best_cost = result[0]
                best_result = result[1]
            mean = [x + y for x, y in zip(mean, result[2])]
            mean = [x / 2 for x in mean]
            mean_cost = (mean_cost + result[0]) / 2

            print("cicle: " + str(i) +" m_size:" +  str(n) + " criteria: " + str(criteria_flag))
            i += 1
    return (best_cost, best_result, mean, mean_time, mean_cost)                
 

########################################### EXECUTION ##################################################

local_path = os.getcwd()
file_names = os.listdir("./qpa")
results_dir = local_path + os.path.sep + "results_e_imgs"
result_file = results_dir + os.path.sep + "results.txt"
try: 
    os.mkdir(results_dir)
except OSError:
    print("could not create images folder, it may already exists")
else:
    for problem_file in file_names:
        path = local_path + os.path.sep + 'qpa' + os.path.sep + problem_file
        a = 0
        #FOR EACH ACCEPTANCE CRITERIA 0=BETTER 1=RW 2=LSMC
        while a < 3:
            #MEAN RESULTS OF 30 TESTS
            number_of_tests = 30
            res = execute_mean_test(number_of_tests, path, a) 
           
            t_color = ('blue' if a == 0 else ('red' if a == 1 else 'green')) 

            plt.title(os.path.splitext(problem_file)[0])
            plt.plot(res[2], color=t_color)
            plt.ylabel('cost')
            plt.xlabel('iteractions')

            text =  "best cost: " + str(res[0])
            plt.text(0.02, (1+a*0.05), text, fontsize=14, transform=plt.gcf().transFigure, c=t_color )
            
            text =  "mean cost: " + str(int((res[4])))
            plt.text(0.4, (1+a*0.05), text, fontsize=14, transform=plt.gcf().transFigure, c=t_color )

            text =  "mean time: " + str('%.5f'%(res[3]))
            plt.text(0.8, (1+a*0.05), text, fontsize=14, transform=plt.gcf().transFigure, c=t_color )
        
            text =  ('better' if a ==0 else ('rw' if a == 1 else 'lsmc'))
            plt.text(1.2, (1+a*0.05), text, fontsize=14, transform=plt.gcf().transFigure, c=t_color )

            #put the results in a file
            # (best_cost, best_result, mean, mean_time, mean_cost)                
            bash_cmd = "echo \"" + problem_file +  (' better' if a ==0 else (' rw' if a == 1 else ' lsmc')) + "\" >> " + result_file
            bash_cmd += " && echo \"" + str(res[0]) + "\" >> " + result_file
            bash_cmd += " && echo \"" + str(res[1]) + "\" >> " + result_file
            bash_cmd += " && echo \"" + str('%.5f'%(res[3])) + "\" >> " + result_file
            bash_cmd += " && echo \"" + str(int(res[4])) + "\" >> " + result_file
            bash_cmd += " && echo \"\"  >> " + result_file
            
            error = subprocess.run(bash_cmd, shell=True, check=True, text=True)        
            a += 1 

        plt.savefig(results_dir + os.path.sep + "plot_" + os.path.splitext(problem_file)[0] + ".png" , bbox_inches='tight')
        plt.clf()
    print("end")
