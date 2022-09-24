import numpy as np
from matplotlib import pyplot as plt 
from matplotlib import colors
from mpl_toolkits.mplot3d import axes3d 
import time
import random


def import_data_from_csv(csv_fname ='cost_table.csv'): 
    r = np.genfromtxt(csv_fname, delimiter=',') 
    return(r)

def export_data_to_csv(data, csv_fname ='cost_table.csv'):
    np.savetxt(csv_fname, data, fmt='%10.0f', delimiter=",")

def cost_table_maker( node_number =10 , lower_number = 1 , upper_number =100 , csv_fname = 'cost_table.csv'):
    if node_number == 0 or lower_number == 0 or upper_number == 0: 
        print(' please enter nonzero number ')
        return ()
    data = np.zeros((node_number , node_number)) 
    for i in range(node_number):
        for j in range(node_number): 
            if i==j:
                data[i][j] = np.nan 
            else :
                data[i][j]= int(np.random.randint(lower_number, upper_number))
    export_data_to_csv(data , csv_fname)

def tasktime_table_maker( node_number =10 , lower_number = 1 , upper_number =100 , csv_fname = 'tasktime_table.csv'):
    if node_number == 0 or lower_number == 0 or upper_number == 0: 
        print ( ' please enter nonzero number ' )
        return ()

    data = np.zeros((node_number , node_number)) 
    for i in range(node_number):
        for j in range(node_number): 
            if i==0 :
                continue 
            elif j==0:
                data[i][j] = np.nan 
            else :
                data[i][j]= int(np.random.randint(lower_number, upper_number))
        #p r i n t ( d a t a [ i ] [ j ] )
    export_data_to_csv(data , csv_fname)

def nodes_table_by_coordinates(node_number =10 , lower_number=10, upper_number =100 , csv_fname = 'nodes_table_by_coordinates.csv'):
    data = lower_number + upper_number*np.random.rand((node_number*2) ,2)
    data = data.astype(int)
    
    a = np.ascontiguousarray(data)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    data = unique_a.view(a.dtype).reshape((unique_a.shape[0] , a.shape[1]))
    data = data [:node_number]
    export_data_to_csv(data , csv_fname)

def cost_table_by_coordinates( csv_fname = 'nodes_table_by_coordinates.csv' , csv_fname_out = ' cost_table_by_coordinates.csv'):
    data = import_data_from_csv(csv_fname).astype(int)
    x = np.asarray([x[0] for x in data] , dtype=np.int) 
    y = np.asarray([x[1] for x in data] , dtype=np.int)
    # take advantage of broadcasting , to make a 2dim array of diffs
    dx = np.sqrt((x[... , np.newaxis] - x[np.newaxis, ...])**2 + (y [... , np.newaxis] - y[np.newaxis , ...]) **2).astype(int)
    # dx= dx.astype(float)
    export_data_to_csv(dx, csv_fname_out)
    data = import_data_from_csv(csv_fname_out)
    for i in range(len(data)):
        for j in range(len(data)):
            if i == j :
                data[i][j] =np.nan
    export_data_to_csv(data , csv_fname_out)
    # plt.scatter(x, y) # plt.show()

def NN_with_tasks(cost_table ='cost_table_by_coordinates.csv', tasktime_table='tasktime_table.csv',csv_fname_out ='NN_best_route.csv' ):
    ## load data by coordinate table
    TT = import_data_from_csv(cost_table) 
    n = len(TT)
    L = len(TT)
    ## load data from task time table to parameter task time (T)
    T = import_data_from_csv(tasktime_table)
    # to create the sequence
    sequence = np.zeros((2*n+2,), dtype=int)
    result = np.zeros(shape=(n,2*n+2) , dtype=int) # print ( sequence )
    for w in range(n):
        #w start point of NN
        sequence = np.zeros((2*n+2,), dtype=int) 
        sequence[0] =w
        sequence[n] =sequence [0]
        k=1
        T2 = np.array(T)
        TT2 = np.array(TT) 
        TT2[:,sequence[0]] = np.nan
        for j in range(1,L):
            sequence[k]= np.nanargmin(TT2[sequence[k-1]])
            TT2[:,sequence[k]] = np.nan
            k+=1
        if w != 0:
            for i in range(n+1):
                if sequence[i] ==0 :
                    temp_sequence = np.append(sequence[i:n+1], sequence [ 1 : i +1])
                    temp_sequence = np.append(temp_sequence ,sequence [n+1:])
            sequence = temp_sequence
        for k3 in range(1,L+1):
            sequence[L+1+L-k3] = np.nanargmin(T2[sequence[L-k3]])
            T2[:,sequence[L+1+L-k3]] = np.nan
        min_cost=0
        cost =0
        len_sequence =len(sequence)
        #CHECK len sequence−1 test=np.zeros((n,), dtype=int)
        test=np.zeros((n,), dtype=int) 
        for s in range(0 ,L) :
            min_cost = np.nansum([min_cost ,TT[sequence[s],sequence[s +1]]])
        for s2 in range(0,L-1):
            cost = np.nansum([cost ,TT[sequence[s2],sequence[s2+1]]])
            #test[s2] =cost+T[sequence[s2+1],sequence[L+2+s2]]
            min_cost = max(min_cost,cost+ T[sequence[s2+1],sequence[L +2+s2]] )
        sequence[-1]=min_cost 
        result [w, :]= sequence
    #export data to csv(result , 'result.csv')
    NN_best_route = result[np.nanargmin([ result [: ,2*n+1]]) , :] 
    export_data_to_csv(NN_best_route , csv_fname_out)
    #print ( NN best route ) #print(result[np.nanargmin([result[: ,−1]]), −1: ])
    return ( NN_best_route )

def two_opt_with_tasks(sequence = 'NN_best_route.csv', cost_table ='cost_table_by_coordinates.csv' , tasktime_table='tasktime_table.csv',csv_fname_out='NN_best_route_2opt.csv', min_cost = 0):
    improved = True
    ## load data by coordinate table
    TT = import_data_from_csv(cost_table)
    ## load sequence from NN
    sequence = import_data_from_csv(sequence).astype(int)
    ## load data from task time table to parameter task time (T)
    T = import_data_from_csv(tasktime_table)
    L = len(TT)
    min_cost =0
    cost =0
    tour=np.zeros((L+1,), dtype=int)
    #CHECK len sequence−1 #calculate the tour length
    for s in range (0 ,L) :
        min_cost = np.nansum([min_cost ,TT[sequence[s],sequence[s+1]]])
    
    #calculate the completion time
    for s2 in range(0,L-1):
        cost = np.nansum([cost ,TT[sequence[s2],sequence[s2+1]] ]) 
        tour[s2+1] =cost+ T[sequence[s2+1],sequence[L+2+s2]]
        min_cost = max(min_cost,cost+ T[sequence[s2+1],sequence[L+2+s2]] )
    twoopt_counter =0
    while improved : 
        improved = False
        for i in range(1,L-1):#range(L−2,0,−1)
            for j in range( i+1,L+1):
                if j-i == 1: continue # changes nothing , skip then
                new_route = 1*sequence[ : ]
                if (TT[new_route[i],new_route[i-1]]+TT[new_route[j], new_route[j-1]]) >= (TT[new_route[i],new_route[j ]]+TT[ new_route [ i -1] , new_route [ j -1]]) :
                    new_route[i:j] = sequence[j-1:i-1:-1] # this is the 2woptSwap
                    #cost function by swapping the nodes
                    new_cost =0
                    for s in range(0,L):
                        new_cost = np.nansum([new_cost ,TT[new_route[
                        s],new_route[s+1]]])
                    T2 = np.array(T)
                    for k3 in range(1,L+1):
                        new_route[L+1+L-k3] = np.nanargmin(T2[ new_route[L-k3]])
                        T2[:,new_route[L+1+L-k3]] = np.nan
                    #test=np.zeros((L,), dtype=int)
                    cost =0
                    new_cost=0
                    for s in range(0,L):
                        new_cost = np.nansum([new_cost ,TT[sequence[s ] , sequence [ s +1]]])
                    for s2 in range(0,L-1):
                        cost = np.nansum([cost ,TT[new_route[s2],new_route[s2+1]] ])
                        #test [ s2 ] =cost+T[ sequence [ s2+1],sequence [L +2+s2 ] ]
                        new_cost = max(new_cost,cost+ T[new_route[s2 +1],new_route[L+2+s2]] )
                    if new_cost < min_cost: 
                        sequence = 1*new_route
                        sequence[-1] = new_cost 
                        improved = True
                        min_cost = 1*new_cost
                        twoopt_counter +=1
                        #print ( min cost )
    cost=0
    for s in range (0 ,L) :
        cost = np.nansum([cost ,TT[sequence[s],sequence[s+1]]])
        #print('tour ',cost)
        export_data_to_csv(sequence , csv_fname_out)
        return ( cost , min_cost , twoopt_counter )

                    
def NN( cost_table ='cost_table_by_coordinates.csv ' , csv_fname_out = 'NN_best_route.csv' ):
    ## load data by coordinate table
    TT = import_data_from_csv ( cost_table ) 
    n=len(TT)
    # to create the sequence
    sequence = np.zeros((n+1,), dtype=int)
    result = np.zeros(shape=(n,n+2) , dtype=int) # print ( sequence )
    for w in range(n):
        #w start point of NN
        sequence=np.zeros((n+1,), dtype=int) 
        sequence[0] =w
        sequence[n]=sequence[0]
        L = len(TT) 
        k=1
        TT2 = np.array(TT) 
        TT2[:,sequence[0]] = np.nan
        for j in range(1,L):
            sequence[k]= np.nanargmin(TT2[sequence[k-1]])
            TT2[:,sequence[k]] = np.nan 
            k+=1
        result [w, :-1]= sequence
        cost =0
        len_sequence =len(sequence)
        for s in range(0,len_sequence-1):
            cost = np.nansum([cost ,TT[sequence[s],sequence[s+1]]]) 
            result[w, n+1:n+2]= cost
    #export data to csv(result , 'result.csv')
#print('the best route is: ', result[np.nanargmin([result[:,−1]]), :])
    if result [np.nanargmin([ result [: ,-1]]) , :][0] ==0 :
        NN_best_route = result [np.nanargmin([ result [: ,-1]]) , :-1]
        export_data_to_csv(NN_best_route , csv_fname_out)
    else :
        for i in range(n+1):
            if result [np.nanargmin([ result [: ,-1]]) , :][ i ] ==0 :
                NN_best_route = np.append(result[np.nanargmin([result [: ,-1]]), i:-1], result[np.nanargmin([result[: ,-1]]) , : i+1])
                for j in range(n-1):
                    if NN_best_route[j] == NN_best_route[j+1]:
                        NN_best_route = np.delete(NN_best_route, j)
                export_data_to_csv(NN_best_route , csv_fname_out)
                break
    #print ( NN best route ) #print(result[np.nanargmin([result[: ,−1]]), −1: ])
    return (result[np.nanargmin([result[: ,-1]]), -1: ])

def two_opt_tsp(sequence = 'NN_best_route.csv', cost_table ='cost_table_by_coordinates.csv' , tasktime_table='tasktime_table.csv',csv_fname_out='NN_best_route_2opt.csv', min_cost = 0):
    improved = True
    ## load data by coordinate table
    TT = import_data_from_csv( cost_table ) 
    L = len(TT)
    ## load sequence
    sequence = import_data_from_csv( sequence ).astype ( int ) 
    sequence = np.append(sequence ,[0])
    ## load data from task time table to parameter task time (T)
    T = import_data_from_csv(tasktime_table)
    min_cost =0 
    cost =0
    #CHECK len sequence−1 
    for s in range (0 ,L) :
        min_cost = np.nansum([min_cost ,TT[sequence[s],sequence[s +1]]])
        
    twoopt_counter =0 
    while improved :
        improved = False
        for i in range(1, L-1):
            for j in range(i+1, L+1):
                if j-i == 1: continue # changes nothing , skip then
                if (TT[ sequence [ i ] , sequence [ i -1]]+TT[ sequence [ j ] , sequence [ j -1]]) >= (TT[ sequence [ i ] , sequence [ j ]]+TT [sequence[i-1],sequence[j-1]]) :
                    new_route = 1*sequence
                    new_route[i:j] = 1*sequence[j-1:i-1:-1] # this is the_2woptSwap
                    #cost function by swapping the nodes
                    new_cost =0
                    for s in range(0,L):
                        new_cost = np.nansum([new_cost ,TT[new_route[ s],new_route[s+1]]])

                    if new_cost < min_cost: 
                        sequence = 1*new_route 
                        sequence[-1] = new_cost 
                        improved = True
                        min_cost = new_cost
                        twoopt_counter =+1
    sequence2 = np. zeros ((2*L+2,) , dtype=int ) 
    sequence2 [0:L+1]=sequence [0:L+1]
    T2 = np.array(T)
    for s in range(1,L+1):
        sequence2[L+1+L-s] = np.nanargmin(T2[sequence2[L-s]])
        T2[:,sequence2[L+1+L-s]] = np.nan
    cost =0
    min_cost = 0
    for s in range(0,L):
        min_cost = np.nansum([min_cost,TT[sequence[s],sequence[s+1]]])

    for s2 in range(0,L-1):
        cost = np.nansum([cost,TT[sequence2[s2],sequence2[s2+1]]])
        min_cost = max(min_cost,cost+ T[sequence2[s2+1],sequence2[L+2+s2]])
    sequence2[-1] =min_cost
    export_data_to_csv(sequence2 , csv_fname_out)
    #print ( sequence2 )
    return ( cost , min_cost , twoopt_counter )

def Conditions(sequence = 'NN_best_route_2opt.csv', cost_table ='cost_table_by_coordinates.csv' , tasktime_table='tasktime_table. csv',csv_fname_out='NN_best_route_2opt.csv', min_cost = 0):
    ## load data by coordinate table
    TT = import_data_from_csv ( cost_table ) 
    L = len(TT)
    ## load sequence
    sequence=import_data_from_csv( sequence ).astype( int )
    ## load data from task time table to parameter task time (T)
    T = import_data_from_csv(tasktime_table)
    min_cost =0 
    cost =0
    comp=np.zeros((L+1,), dtype=int)
    tour=np.zeros((L+1,), dtype=int)
    T2 = np.array(T)
    for s in range(1,L+1):
        sequence[L+1+L-s] = np.nanargmin(T2[sequence[L-s]]) 
        T2[:,sequence[L+1+L-s]] = np.nan
    
    #calculate the tour length
    for s in range (0 ,L) :
        min_cost = np.nansum([min_cost ,TT[sequence[s],sequence[s
        +1]]])

    for s2 in range(0,L-1):
        cost = np.nansum([cost ,TT[sequence[s2],sequence[s2+1]] ]) 
        tour [ s2+1] =cost
        comp[s2+1] =cost+ T[sequence[s2+1],sequence[L+2+s2]] 
        min_cost=max(min_cost,comp[s2+1] )
    sequence[-1] = min_cost

    min_counter =0 
    forward_counter=0 
    backward_counter=0
    improved_task = True 
    improved = True
    improve_reverse_swap = True
    
    #condition 1: is the task time of Cmax minimum task time for the cmax node?
    while improve_reverse_swap : 
        improve_reverse_swap =False
        while improved : 
            improved = False
            cmax_node = np.nanargmax(comp)
            if np.nanargmin(T[sequence[cmax_node]]) != sequence[L+1+ cmax_node ] :
                new_route=1*sequence 
                T2 = np.array(T)
                new_route[L+1+cmax_node] = np.nanargmin(T2[new_route[ cmax_node]])
                T2[:,new_route[L+1+cmax_node]]= np.nan
                for k3 in range(1,L+1):
                    if (L-k3)==cmax_node: continue
                    new_route[L+1+L-k3] = np.nanargmin(T2[ new_route[L-k3]])
                    T2[:,new_route[L+1+L-k3]] = np.nan
                improved_task = True
                for s in range (1 ,L) :
                    if s == cmax_node: continue
                    if tour[s]+T[new_route[s],new_route[L+1+s]]>min_cost :
                        improved_task = False 
                        break
                new_cost=0
                for s in range (0 ,L) :
                    new_cost = np.nansum([new_cost ,TT[new_route[s], new_route [ s +1]]])
                if new_cost > min_cost: 
                    improved_task = False
                
                if improved_task :
                    sequence = 1*new_route
                    improved = True
                    for s2 in range(0,L-1):
                        comp[ s2+1] =tour[s2+1]+ T[sequence[s2+1], sequence[L+2+ s2]]
                        min_cost=max(min_cost,comp[s2+1] )
                    sequence[-1] = min_cost
                    min_counter +=1
                    #print ( ' Reordering task ' , min cost )
                else:
                    new_route=1*sequence
                    for s in range(cmax_node+1, L):
                        if T[sequence[cmax_node],sequence[L+1+s]] < T [ sequence [ cmax_node ] , sequence [L+1+cmax_node]]:
                            if tour[s] +T[ sequence[ s ] , sequence[L+1+ cmax_node]] < min_cost:
                                new_route[L+1+cmax_node] =sequence[L +1+s ]
                                new_route [L+1+s ] =sequence [L+1+ cmax_node ]
                                new_cost = 0
                                for s2 in range (0 ,L) :
                                    new_cost = np.nansum([ new_cost , TT[new_route[s2] ,new_route[s2
                                    +1]]])
                                if new_cost < min_cost:
                                    sequence = 1*new_route
                                    improved = True
                                    min_cost = max(tour[s] +T[sequence [ s ] , sequence [L+1+s ]] ,tour[ s ] +T[ sequence [ cmax_node],sequence[L+1+cmax_node]])
                                    sequence[-1] = min_cost
                                    for s2 in range(0,L-1):
                                        comp[s2+1] =tour [ s2+1]+ T[ sequence[ s2+1],sequence[L+2+s2 ]]
                                    forward_counter +=1
        if cmax_node >2 or cmax_node<L-1:
            for p in range(1, cmax_node-1):
                check = True
                TC = TT[sequence[cmax_node-p-1],sequence[cmax_node]]+TT[ sequence[ cmax_node-p ] , sequence[ cmax_node +1]] - TT[ sequence[ cmax_node-p-1] , sequence[ cmax_node-p]] - TT[ sequence[ cmax_node ] , sequence[ cmax_node +1]]
                for s in range(cmax_node+1, L): 
                    if TC+comp[s] > min_cost:
                        check = False
                        break
                if check:
                    new_route = 1*sequence
                    new_route[cmax_node-p: cmax_node+1] = sequence[cmax_node : cmax_node-p-1:-1] # this is the 2woptSwap
                    new_route [L+1+cmax_node-p :L+1+cmax_node+1] = 1* sequence [L+1+cmax_node :L+1+cmax_node-p-1:-1]
                    cost = 0
                    for s2 in range(0,L-1):
                        cost = np.nansum([cost ,TT[new_route[s2],new_route[s2+1]] ])
                        tour [ s2+1] =cost
                        comp[ s2+1] =cost+ T[ new_route [ s2+1],new_route[ L+2+ s2 ] ]
                        if comp[s2+1]>=min_cost:
                            check = False
                            break
                    if check:
                        sequence = 1*new_route 
                        improve_reverse_swap = True 
                        improved = True
                        cost=0
                        min_cost=0
                        for s2 in range (0 ,L) :
                            min_cost = np.nansum([min_cost ,TT[ sequence [ s2 ] , sequence [ s2 +1]]])
                        for s2 in range(0,L-1):
                            min_cost=max(min_cost,comp[s2+1] )
                        sequence[-1] = min_cost
                        backward_counter +=1
                    else:
                        T2 = np.array(T)
                        for k3 in range(1,L+1):
                            new_route[L+1+L-k3] = np.nanargmin(T2[ new_route[L-k3]])
                            T2[:,new_route[L+1+L-k3]] = np.nan
                        cost =0
                        new_cost=0
                        for s2 in range (0 ,L) :
                            new_cost = np.nansum([new_cost ,TT[ new_route[s2],new_route[s2+1]]])
                        for s2 in range(0,L-1):
                            cost = np.nansum([cost ,TT[new_route[s2],
                            new_route[s2+1]] ])
                            tour [ s2+1] =cost
                            comp[ s2+1] =cost+ T[ new_route [ s2+1],new_route[L+2+s2]]
                            new_cost=max(new_cost,comp[s2+1] )
                        if new_cost < min_cost:
                            new_route[-1] = new_cost 
                            min_cost=new_cost
                            sequence = 1*new_route 
                            improve_reverse_swap = True 
                            improved = True
                            backward_counter +=1
                if improve_reverse_swap : break
    cost=0
    for s in range (0 ,L) :
        cost = np.nansum([cost ,TT[sequence[s],sequence[s+1]]])
    export_data_to_csv(sequence , csv_fname_out)
    min_cost = sequence[-1]
    return ( cost , min_cost , min_counter , forward_counter ,backward_counter )

def NNII( cost_table ='cost_table_by_coordinates.csv' , tasktime_table ='tasktime_table.csv',csv_fname_out ='NN_best_route_2opt.csv'):
    TT = import_data_from_csv ( cost_table ) 
    n=len(TT)
    T = import_data_from_csv(tasktime_table)
    sequence = np.zeros((n+1,), dtype=int)
    result = np.zeros(shape=(n,n+2) , dtype=int)
    long= np.nanargmax(TT[sequence[0]])
    sequence=np.zeros((n+1,), dtype=int) 
    sequence [0] =0
    sequence [ n]=0
    L = len(TT)
    k=1
    w=0
    TT2 = np.array(TT) 
    TT2[:,sequence[0]] = np.nan

    sequence[L-1]= np.nanargmax(TT2[sequence[0]]) 
    TT2[:,sequence[L-1]] = np.nan
    for j in range(1,L-1):
        sequence[k]= np.nanargmin(TT2[sequence[k-1]])
        TT2[:,sequence[k]] = np.nan 
        k+=1
    result [w, :-1]= sequence
    cost =0
    len_sequence =len(sequence)
    for s in range(0,len_sequence-1):
        cost = np.nansum([cost ,TT[sequence[s],sequence[s+1]]])
    result [w, n+1:n+2]= cost
    for w in range (1 ,n) :
        #w start point of NN
        sequence=np.zeros((n+1,), dtype=int) 
        sequence [0] =w
        sequence [n]=sequence [0]
        L = len(TT) 
        k=1
        TT2 = np.array(TT) 
        TT2[:,sequence[0]] = np.nan 
        TT2[:,0]= np.nan
        for j in range(1,L):
            if sequence[k-1] == long : 
                sequence[k] = 0
                TT2[:,sequence[k]] = np.nan
            else :
                sequence[k]= np.nanargmin(TT2[sequence[k-1]]) 
                TT2[:,sequence[k]] = np.nan
            k += 1
        result[w,:-1] = sequence
        cost = 0
        len_sequence = len(sequence)
        for s in range(0,len_sequence-1):
            cost = np.nansum([cost ,TT[sequence[s],sequence[s+1]]])
        result [w, n+1:n+2]= cost
    if result [np.nanargmin([ result [: ,-1]]) , :][0] ==0 :
        NN_best_route = result [np.nanargmin([ result [: ,-1]]) , :-1]
    else :
        for i in range(n+1):
            if result [np.nanargmin([ result [: ,-1]]) , :][ i ] ==0 :
                NN_best_route = np.append(result[np.nanargmin([result[: ,-1]]), i:-1], result[np.nanargmin([result[:,-1]]) , : i+1])
                for j in range(n-1):
                    if NN_best_route[j] == NN_best_route[j+1]:
                        NN_best_route = np.delete(NN_best_route, j )
                break

    sequence2 = np. zeros ((2*L+2,) , dtype=int ) 
    sequence2 [0:L+1]=NN_best_route [0:L+1]
    T2 = np.array(T)
    for s in range(1,L+1):
        sequence2[L+1+L-s] = np.nanargmin(T2[sequence2[L-s]]) 
        T2[:,sequence2[L+1+L-s]] = np.nan
    cost = 0
    min_cost = 0
    for s2 in range(0,L):
        min_cost = np.nansum([min_cost ,TT[sequence2[s2],sequence2[s2 +1]]])
    for s2 in range(0,L-1):
        cost = np.nansum([cost ,TT[sequence2[s2],sequence2[s2+1]] ])
        min_cost = max(min_cost,cost+ T[sequence2[s2+1],sequence2[L+2+s2]] )
        sequence2[-1] = min_cost
    export_data_to_csv(sequence2 ,csv_fname_out)
    return (cost,min_cost)

def main():
    log = open("output.txt", "w")
    #print( 'Sample #,' , 'Group , ' , 'Sample size , ' , 'Model, ' , 'Tour length , ' , 'Completion time , ' , ' operation time , ' , ' 2opt , ' , ' min jobtim , ' , ' forward job swapping , ' , ' backward node swapping ' )
    #print( 'Sample #,' , 'Group , ' , 'Sample size , ' , 'Model, ' , 'Tour length , ' , 'Completion time , ' , ' operation time , ' , ' 2opt , ' , ' min jobtim , ' , ' forward job swapping , ' , 'backward node swapping ' , file = log)
    print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format("Sample","Gr","SS","Mo","TL","Cmax","Time","2opt","min JT","FJS","BNS"))
    print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format("Sample","Gr","SS","Mo","TL","Cmax","Time","2opt","min JT","FJS","BNS"),file = log)
    
    modelos_imprimir = ["I","II","III","IV"] #["I","II","III","IV"]
    instancias_tsplib = ["gr17","gr21","gr24","fri26","bays29","gr48","eil51","berlin52","eil76","eil101"]
    for group in ["Tsplib","Small","Medium","Large"]: #["Tsplib","Small","Medium","Large"]:
        if group == "Tsplib":
            inicio = 0
            fin = 10
        else:
            inicio = 1
            fin = 101

        for sample in range (inicio ,fin) : #range (1 ,101) :
            if group != "Tsplib":
                batch = str((sample-1)//25+1)
                CoT = "Data/"+group+"_problems/Batch_0"+batch+'/TSPJ_'+ str(sample) + str(group[0])+ '_cost_table_by_coordinates.csv'
                TaT = "Data/"+group+"_problems/Batch_0"+batch+'/TSPJ_'+ str (sample) + str (group[0])+ '_tasktime_table.csv'
            else:
                CoT = "Data/"+group+"_problems/TT_"+instancias_tsplib[sample]+".csv"
                TaT = "Data/"+group+"_problems/JT_"+instancias_tsplib[sample]+".csv"

            NNB = 'salida_tesis/TSPJ_'+ str (sample) + str (group[0])+ '_NN_best_route.csv'
            NNBO= 'salida_tesis/TSPJ_'+ str(sample) + str(group[0])+ "_NN_best_route_2opt.csv"
             

            TT = import_data_from_csv (CoT) 
            node_number=len(TT)
            results = [0 ,0 ,0 ,0 ,0 ,0] 
            temp= [0 ,0 ,0 ,0 ,0 ,0]

            if "I" in modelos_imprimir:
                #first model
                start = time.time ()
                NN_with_tasks(CoT,TaT , NNB )
                results[:2] = two_opt_with_tasks(NNB, CoT , TaT)
                end = time.time()
                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(sample,group,node_number,"I",'%.2f' % results[0],'%.2f' % results[1],'%.2f' % (end - start),results[2],results[3],results[4],results[5]))
                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(sample,group,node_number,"I",'%.2f' % results[0],'%.2f' % results[1],'%.2f' % (end - start),results[2],results[3],results[4],results[5]),file=log)
                #print(sample,'  ', group, ' , ', node_number,', ' , 'Model: I ' , ' , ' , results [0] , ' , ', results[1],' , ', '%.2f' % (end - start),', ',results[2] ,' , ',results[3],' , ',results[4] ,' , ',results[5] )
                #print(sample,'  ', group, ' , ', node_number,', ' , 'Model: I ' , ' , ' , results [0] , ' ,', results[1],' , ', '%.2f' % (end - start) ,' ,',results[2] ,' , ',results[3],' , ',results [4] ,' , ',results[5], file = log)

            if "II" in modelos_imprimir:
                #Second model
                results =[0 ,0 ,0 ,0 ,0 ,0] 
                temp=[0 ,0 ,0 ,0 ,0 ,0]
                start = time . time ()
                NN(CoT,NNB) 
                results[:2] = two_opt_tsp(NNB,CoT,TaT ,NNBO)
                temp [ : ] = Conditions (NNBO,CoT,TaT,NNBO)
                results[:1]= temp[:1] 
                results[3:] = temp[2:]
                end = time.time()
                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(sample,group,node_number,"II",'%.2f' % results[0],'%.2f' % results[1],'%.2f' % (end - start),results[2],results[3],results[4],results[5]))
                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(sample,group,node_number,"II",'%.2f' % results[0],'%.2f' % results[1],'%.2f' % (end - start),results[2],results[3],results[4],results[5]),file=log)
            

            if "III" in modelos_imprimir:
                #Third model
                results =[0 ,0 ,0 ,0 ,0 ,0] 
                temp=[0 ,0 ,0 ,0 ,0 ,0]
                start = time . time ()
                NN_with_tasks (CoT,TaT , NNB )
                results [1:] = Conditions(NNB,CoT,TaT,NNBO)
                results [:3] = two_opt_with_tasks(NNBO,CoT,TaT,NNBO )
                end = time.time()

                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(sample,group,node_number,"III",'%.2f' % results[0],'%.2f' % results[1],'%.2f' % (end - start),results[2],results[3],results[4],results[5]))
                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(sample,group,node_number,"III",'%.2f' % results[0],'%.2f' % results[1],'%.2f' % (end - start),results[2],results[3],results[4],results[5]),file=log)

            if "IV" in modelos_imprimir:
                #Forth model
                results =[0 ,0 ,0 ,0 ,0 ,0] 
                temp=[0 ,0 ,0 ,0 ,0 ,0]
                start = time.time() 
                NNII(CoT,TaT ,NNBO)
                results [1:] = Conditions(NNBO,CoT,TaT,NNBO)
                results [:3] = two_opt_with_tasks(NNBO,CoT,TaT,NNBO )
                end = time.time()
                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(sample,group,node_number,"IV",'%.2f' % results[0],'%.2f' % results[1],'%.2f' % (end - start),results[2],results[3],results[4],results[5]))
                print("{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}{:<10}".format(sample,group,node_number,"IV",'%.2f' % results[0],'%.2f' % results[1],'%.2f' % (end - start),results[2],results[3],results[4],results[5]),file=log)



if __name__ =='__main__': 
    main()