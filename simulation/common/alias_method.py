file_name = "ALIAS_TABLE_rate{}.txt"
    
def gen_alias(weights):
    '''
    @brief:
        generate alias from a list of weights (where every weight should be no less than 0)
    '''
    n_weights = len(weights)
    avg = sum(weights)/(n_weights+1e-6)
    aliases = [(1, 0)]*n_weights
    smalls = ((i, w/(avg+1e-6)) for i, w in enumerate(weights) if w < avg)
    bigs = ((i, w/(avg+1e-6)) for i, w in enumerate(weights) if w >= avg)
    small, big = next(smalls, None), next(bigs, None)
    while big and small:
        aliases[small[0]] = (float(small[1]), int(big[0]))
        big = (big[0], big[1] - (1-small[1]))
        if big[1] < 1:
            small = big
            big = next(bigs, None)
        else:
            small = next(smalls, None)
    
    for i in range(len(aliases)):
        tmp = list(aliases[i])
        aliases[i] = (int(tmp[0] * (n_weights-1)),tmp[1])

    return aliases

def write_table(table,file):
    for i in table:
       tmp = list(i)
       file.write(str(tmp[0]) + " " + str(tmp[1]) + "\n")
    file.close()

def read_table(file):
    table = []
    data = file.read().split("\n")
    file.close()
    for i in data:
        if i == "":
            continue
        tmp = i.split()
        table.append((int(tmp[0]),int(tmp[1])))
    return table

def init_alias(weights):
    file = open(file_name.format(0),"w")
    table = gen_alias(weights)
    write_table(table,file)
    
def update_alias(weights):    
    file = open(file_name.format(0),"w")
    table = gen_alias(weights)
    write_table(table,file)

def get_index_alias_method(indexes, rand_num):
    file = open(file_name.format(0),"r")
    table = read_table(file)
    out = []
    for i in range(2):
        ind = indexes[i]
        rand = rand_num[i]
        if(table[ind][0]<rand):
            out.append(table[ind][1])
        else:
            out.append(ind)
    return out
    
    