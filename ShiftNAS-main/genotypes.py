from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# NASNet
# shiftNAS_final_C10 = Genotype(normal = [('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)],normal_concat = range(2, 7), reduce = [('sep_conv_5x5', 1), ('sep_conv_7x7', 0), ('max_pool_3x3', 1), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 1)], reduce_concat = range(4, 7))
# AmoebaNet 
# shiftNAS_final_C10 = Genotype(normal = [('avg_pool_3x3', 0),('max_pool_3x3', 1),('sep_conv_3x3', 0),('sep_conv_5x5', 2),('sep_conv_3x3', 0),('avg_pool_3x3', 3),('sep_conv_3x3', 1),('skip_connect', 1),('skip_connect', 0),('avg_pool_3x3', 1)],normal_concat = range(4, 7),reduce = [('avg_pool_3x3', 0),('sep_conv_3x3', 1),('max_pool_3x3', 0),('sep_conv_7x7', 2),('sep_conv_7x7', 0),('avg_pool_3x3', 1),('max_pool_3x3', 0),('max_pool_3x3', 1),('conv_7x1_1x7', 0),('sep_conv_3x3', 5)],reduce_concat = [3, 4, 6])
# DARTS-v1
# shiftNAS_final_C10 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
# DARTS-v2
# shiftNAS_final_C10 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
# GDAS
# shiftNAS_final_C10 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1),('skip_connect', 0), ('sep_conv_5x5', 2),('sep_conv_3x3', 3), ('skip_connect', 0),('sep_conv_5x5', 4), ('sep_conv_3x3', 3)],normal_concat=[2, 3, 4, 5],reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 2), ('sep_conv_5x5', 1),('dil_conv_5x5', 2), ('sep_conv_3x3', 1),('sep_conv_5x5', 0), ('sep_conv_5x5', 1)],reduce_concat=[2, 3, 4, 5])
# P-DARTS
# shiftNAS_final_C10 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
# PC-DARTS
# shiftNAS_final_C10 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
# DARTS-
# shiftNAS_final_C10=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 2), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
# DOTS
# shiftNAS_final_C10 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1), ('dil_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('max_pool_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
#shiftnas 1
# shiftNAS_final_C10=Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
#shiftnas lrrenew1
# shiftNAS_final_C10=Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 4)], reduce_concat=range(2, 6))
#shiftnas lrrenew2
# shiftNAS_final_C10=Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('skip_connect', 2), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
#shiftnas lrrenew3
# shiftNAS_final_C10=Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 2), ('sep_conv_3x3', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
#shiftnas lrrenew4
# shiftNAS_final_C10=Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('skip_connect', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
# ShiftNAS cifar100_best
# shiftNAS_final_C10=Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
# ShiftNAS c10-arch2
# shiftNAS_final_C10=Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('dil_conv_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
# ShiftNAS archshift1
# shiftNAS_final_C10=Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('max_pool_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
# ShiftNAS archshift-2no
shiftNAS_final_C10=Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('skip_connect', 2), ('dil_conv_5x5', 3), ('sep_conv_5x5', 1), ('sep_conv_3x3', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 3), ('sep_conv_5x5', 2), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))


# NASNet
# shiftNAS_final_C100 = Genotype(normal = [('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 0), ('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 0), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 1)],normal_concat = range(2, 7), reduce = [('sep_conv_5x5', 1), ('sep_conv_7x7', 0), ('max_pool_3x3', 1), ('sep_conv_7x7', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('skip_connect', 3), ('avg_pool_3x3', 2), ('sep_conv_3x3', 2), ('max_pool_3x3', 1)], reduce_concat = range(4, 7))
# ENAS
# shiftNAS_final_C100 = 
# AmoebaNet 
# shiftNAS_final_C100 = Genotype(normal = [('avg_pool_3x3', 0),('max_pool_3x3', 1),('sep_conv_3x3', 0),('sep_conv_5x5', 2),('sep_conv_3x3', 0),('avg_pool_3x3', 3),('sep_conv_3x3', 1),('skip_connect', 1),('skip_connect', 0),('avg_pool_3x3', 1)],normal_concat = range(4, 7),reduce = [('avg_pool_3x3', 0),('sep_conv_3x3', 1),('max_pool_3x3', 0),('sep_conv_7x7', 2),('sep_conv_7x7', 0),('avg_pool_3x3', 1),('max_pool_3x3', 0),('max_pool_3x3', 1),('conv_7x1_1x7', 0),('sep_conv_3x3', 5)],reduce_concat = [3, 4, 6])
# DARTS-v1
# shiftNAS_final_C100 = Genotype(normal=[('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 0), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('avg_pool_3x3', 0)], reduce_concat=[2, 3, 4, 5])
# DARTS-v2
# shiftNAS_final_C100 = Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 0), ('skip_connect', 0), ('dil_conv_3x3', 2)], normal_concat=[2, 3, 4, 5], reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('skip_connect', 2), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('skip_connect', 2), ('skip_connect', 2), ('max_pool_3x3', 1)], reduce_concat=[2, 3, 4, 5])
# GDAS
# shiftNAS_final_C100 = 
# P-DARTS
# shiftNAS_final_C100 = Genotype(normal=[('skip_connect', 0), ('dil_conv_3x3', 1), ('skip_connect', 0),('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3',0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('max_pool_3x3', 0), ('dil_conv_3x3', 1), ('dil_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
# PC-DARTS
# shiftNAS_final_C100 = Genotype(normal=[('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 0), ('dil_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('avg_pool_3x3', 0), ('dil_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 2), ('sep_conv_3x3', 0), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))
# DARTS-
# shiftNAS_final_C100=Genotype(normal=[('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 0), ('dil_conv_5x5', 0), ('dil_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 0)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('skip_connect', 2), ('sep_conv_5x5', 0), ('skip_connect', 2), ('skip_connect', 3), ('skip_connect', 2), ('skip_connect', 3)], reduce_concat=range(2, 6))
# DOTS
# shiftNAS_final_C100=Genotype(normal=[('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 3), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 3), ('dil_conv_3x3', 4)], reduce_concat=range(2, 6))
#shiftnas 1
# shiftNAS_final_C100=Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3), ('skip_connect', 1), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('dil_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('skip_connect', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
#shiftnas new_0.01 (bad, 74.45%)
# shiftNAS_final_C100 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_3x3', 0), ('avg_pool_3x3', 1), ('skip_connect', 0), ('avg_pool_3x3', 1), ('sep_conv_5x5', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
# shiftnas archlrrenew
# shiftNAS_final_C100 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
#shiftnas archlrrenew_1
# shiftNAS_final_C100 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('skip_connect', 2), ('sep_conv_3x3', 1), ('skip_connect', 2)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('sep_conv_5x5', 1), ('avg_pool_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3), ('skip_connect', 1), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))
#shiftnas archlrrenew_2
# shiftNAS_final_C100 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 1), ('dil_conv_5x5', 3)], reduce_concat=range(2, 6))
#shiftnas archlrrenew_3
# shiftNAS_final_C100 = Genotype(normal=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_5x5', 4)], reduce_concat=range(2, 6))
#shiftnas archlrrenew_4.1
# shiftNAS_final_C100 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2)], reduce_concat=range(2, 6))
#shiftnas archlrrenew_4.2
# shiftNAS_final_C100 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1)], reduce_concat=range(2, 6))
#shiftnas archshift1
# shiftNAS_final_C100 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('dil_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 0), ('skip_connect', 1), ('sep_conv_3x3', 1), ('dil_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('skip_connect', 0), ('skip_connect', 1), ('max_pool_3x3', 1), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 2), ('sep_conv_3x3', 0), ('dil_conv_5x5', 4)], reduce_concat=range(2, 6))
#shiftnas archshift2
# shiftNAS_final_C100 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('sep_conv_3x3', 1), ('sep_conv_3x3', 4)], normal_concat=range(2, 6), reduce=[('avg_pool_3x3', 0), ('skip_connect', 1), ('avg_pool_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
# shiftnas dgx_c100_archshift1
# shiftNAS_final_C100 = Genotype(normal=[('sep_conv_3x3', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('max_pool_3x3', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_5x5', 0), ('sep_conv_3x3', 3)], reduce_concat=range(2, 6))
# shiftnas archshift-2no
shiftNAS_final_C100 = Genotype(normal=[('skip_connect', 0), ('skip_connect', 1), ('skip_connect', 0), ('skip_connect', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 4)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 0), ('skip_connect', 1), ('sep_conv_5x5', 1), ('dil_conv_5x5', 2), ('sep_conv_5x5', 0), ('sep_conv_5x5', 2)], reduce_concat=range(2, 6))


shiftNAS_ImageNet=Genotype(normal=[('dil_conv_5x5', 0), ('skip_connect', 1), ('skip_connect', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1)], normal_concat=range(2, 6), reduce=[('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 0), ('sep_conv_5x5', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 3), ('sep_conv_3x3', 0), ('max_pool_3x3', 1)], reduce_concat=range(2, 6))


