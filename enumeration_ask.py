'''
This py script implement the enumerate query
'''

import csv
import numpy as np

class Node():
    '''
    Variable node class
    '''
    def __init__(self, name):
        self.node_name = name
        self.parent = []
        self.children = []
        self.cpt = {}

def sort_evidence(evidence_string):
    '''
    sort the evidence to a specific format
    In this case, sort by Variable's alphabet order
    '''
    if evidence_string:
        evidence_list = evidence_string.split(",")
        evidence_list.sort(key=lambda x: x[0])
        return ','.join(evidence_list)
    else:
        return "T"

def bn_reader(path):
    '''
    read the bn file and build a bayes net with cpt
    '''
    file = open(path)
    file_reader = csv.reader(file)
    for row in file_reader:
        line = ','.join(row)
        if "% Random" in line:
            model = "var"
            continue
        if "% Graph" in line:
            model = "graph"
            continue
        if "% Probability" in line:
            model = "p values"
            continue

        if model == "var":
            var_list = line.split(", ")
            var_dict = {}
            for i in var_list:
                var_dict[i] = Node(i)
        if model == "graph":
            parent = line.split(", ")[0]
            child = line.split(", ")[1]
            var_dict[parent].children.append(var_dict[child])
            var_dict[child].parent.append(var_dict[parent])
        if model == "p values":
            if "|" not in line:
                # no evidence
                name = line.split(")=")[0][2:-2]
                var_dict[name].cpt["T"] = float(line.split(")=")[1])
                # print(name)
            else:
                #have evidence
                part = line.split("|")
                name = part[0][2:-2]
                # print(name)
                evidence = part[1].split(")=")[0]
                evidence = sort_evidence(evidence)
                prob = float(part[1].split(")=")[1])
                var_dict[name].cpt[evidence] = prob
    file.close()
    return var_dict

def query_reader(path):
    '''
    read the input query
    '''
    file = open(path)
    file_reader = csv.reader(file)
    for row in file_reader:
        line = ','.join(row)
        if "% Query" in line:
            model = "q"
            continue
        if "% Evidence" in line:
            model = "evidence"
            continue
        if "% End" in line:
            break
        if model == "q":
            q_var = line
        if model == "evidence":
            if line:
                evidence = line.split(", ")
            else:
                print("no evidence")
                evidence = []
    return q_var, evidence

def enumeration_ask(var, evidence, b_net):
    '''
    x is the query var
    e is all the given evidence. should be a list
    bn is the bayes net
    '''
    q_x = []
    for s in ['T', 'F']:
        e_extend = evidence + [b_net[var].node_name+"="+s]
        bn_vars = list(b_net.keys()) #in test case: ['A','B','C','D']
        q_x.append(enumerate_all(bn_vars, e_extend, b_net))
    q_x = np.array(q_x)
    q_x = q_x/np.sum(q_x)
    return q_x

def enumerate_all(vars, evidence, b_net):
    '''
    enumerate all the branches
    '''
    if len(vars) == 0:
        return 1.0
    y = vars[0]
    rest_vars = vars[1:]
    y_parent = b_net[y].parent
    extend_e = []
    for p in y_parent:
        for evid in evidence:
            if p.node_name in evid:
                extend_e.append(evid)
    extend_e_str = ','.join(extend_e)
    if (y+"=T" in evidence) or (y+"=F" in evidence):
        if y+"=T" in evidence:
            p = b_net[y].cpt[sort_evidence(extend_e_str)]
        if y+"=F" in evidence:
            p = 1.0 - b_net[y].cpt[sort_evidence(extend_e_str)]
        return p * enumerate_all(rest_vars, evidence, b_net)
    else:
        p = b_net[y].cpt[sort_evidence(extend_e_str)]
        b1 = p * enumerate_all(rest_vars, evidence+[y+"=T"], b_net)
        b2 = (1.0 - p) * enumerate_all(rest_vars, evidence+[y+"=F"], b_net)
        return b1+b2

def main():
    '''
    enumerate a bayes networks tree and return the distribution of the query variable
    '''
    bn_path = "bn.txt"
    input_path = "input.txt"
    bn = bn_reader(bn_path)
    q_var, e = query_reader(input_path)
    print("Query begin!\nQuery variable:", q_var, "\nEvidence:", e)
    q = enumeration_ask(q_var, e, bn)
    print("Distribution:(T,F)")
    print(q)

if __name__ == "__main__":
    main()
