import urllib.request
from tqdm import tqdm
import pandas as pd
from biopandas.pdb import PandasPdb
import numpy as np
import random
import collections
import pickle
random.seed(10)





#################################################################################
################################################################################
# a=np.asarray([1,2,3])
# b=np.asarray([4,2,3])
# c=np.asarray([3,2,3])
# d=np.asarray([5,2,3])
# e=np.asarray([a,
#               b,
#               c,
#               d])
# ci = np.asarray(['a','a','b','b'])
# dis_cal(a,b)
# ci = np.asarray(['a','a','b','b']) chain index
def dis_cal(a,b):
    dis=sum((a-b)*(a-b))
    return dis


# given: given aa loc, all: all aa loc, k: top k
# i: find_nearest_aa(a, e, 2)
# o: [0, 1, 2]
def find_nearest_aa(given, all, k):
    nearest = []
    for i in all:
        nearest.append(dis_cal(given,i))
    # result = [i for i, x in enumerate(np.argsort(nearest) <= k) if x]
    result = list(np.argsort(nearest)[:k])
    return result

def find_rbd(aa_coordinate, k, chain_index, cutoff):
    result = []
    t_aa_coordinate = tqdm(aa_coordinate)
    for i in t_aa_coordinate:
        t_aa_coordinate.set_description("finding nearest aa")
        result.append(find_nearest_aa(i, aa_coordinate, k))
    nearest_list =  [i for i in enumerate(result)]
    nearest_index = [j for i, j in nearest_list]
    counter_list = []
    m = 0
    t_nearest_index = tqdm(nearest_index)
    for i in t_nearest_index:
        t_nearest_index.set_description("finding rbd aa")
        counter = 0
        n = 0
        for j in i:
            if chain_index[m] != chain_index[i[n]]:
                counter = counter + 1
            n = n + 1
        ratio = counter / (n + 1)
        counter_list.append(ratio)
        m = m + 1
    rbd_aa = [i for i, x in enumerate(np.array(counter_list) >= cutoff) if x]
    rbd = np.zeros(len(aa_coordinate), dtype=int)
    rbd[rbd_aa] = 1
    return rbd


def mutate_ab_bind(pdb_number):
    if pdb_number[0] == 'H': pdb_number = pdb_number[3:7]
    ab_data = pd.read_csv('data/AB-Bind_experimental_data.csv', encoding="ISO-8859-1")
    data = ab_data[ab_data['#PDB'] == pdb_number]['Mutation']
    mut_indexes = data.str.split(',')
    ppdb = PandasPdb().read_pdb('data/' + pdb_number + '.pdb')
    residue_seq = ppdb.amino3to1()
    residue_seq_number = ppdb.df['ATOM']['residue_number'][residue_seq.index]
    pre_mut_seq = residue_seq['residue_name'].copy()
    aft_mut_seq_list = []
    ddg = np.asarray(ab_data[ab_data['#PDB'] == pdb_number]['ddG(kcal/mol)'])
    k = -1
    k_list = []
    error = []
    for mut_index in mut_indexes:
        error_n = 0
        k = k + 1
        try:
            mut_residue_seq = residue_seq.copy()
            for i in mut_index:
                # mutate
                mut_data = i.split(':')
                mut_chain = mut_data[0]
                mut_aa = mut_data[1]
                mut_pos = int(mut_aa[1:len(mut_aa) - 1])
                pre_mut = mut_aa[0]
                aft_mut = mut_aa[-1]
                # select chain, residue
                mut_residue_seq['residue_number'] = residue_seq_number
                mut_subset_index = (np.asarray(mut_residue_seq['chain_id'] == mut_chain)) & \
                                   (np.asarray(mut_residue_seq['residue_number'] == mut_pos))
                # check the mutation pos
                mut_subset_index_pos = int(np.where(mut_subset_index)[0])

                if pre_mut == mut_residue_seq[mut_subset_index].iloc[0][1]:
                    mut_residue_seq.iat[mut_subset_index_pos, 1] = aft_mut
                    # mut_residue_seq[mut_subset_index].iloc[0][1] = aft_mut
                    # aft_mut_seq = mut_residue_seq['residue_name'].to_string(index=False).replace("\n", '')
                    # aft_mut_seq_list.append([aft_mut_seq])
                else:
                    print('{} aa {} before mutation not match at chain {} position {}'.format\
                              (pdb_number, pre_mut, mut_chain, mut_pos))
                    error.append('{} aa {} before mutation not match at chain {} position {}, deleting mutation at {}'.format\
                              (pdb_number, pre_mut, mut_chain, mut_pos, str(k)))
                    k_list.append(k)
                    error_n = 1
                    break
        except (ValueError, IndexError, TypeError) as e:
            print(e)
            print(pdb_number + ' delete mutation at ' + str(k))
            error.append('{} delete mutation at {}'.format(pdb_number, str(k)))
            k_list.append(k)
            error_n = 1

        if error_n == 0:
            aft_mut_seq = mut_residue_seq['residue_name'].to_string(index=False).replace("\n", '')
            aft_mut_seq_list.append([aft_mut_seq])
    ddg = np.delete(ddg, k_list)
    residue_seq['residue_number'] = residue_seq_number
    return {'pre_mut_seq': pre_mut_seq.to_string(index = False).replace("\n",''),
            'aft_mut_seq_list': aft_mut_seq_list, 'mutated_info': mut_indexes,
            'ddg': ddg,
            'chain_index': residue_seq['chain_id'].to_string(index = False).replace("\n",''),
            'residue_number': np.asarray(residue_seq['residue_number']), 'error': error}

# a = mutate_ab_bind('3BN9')
# for subset the sequences based on chain index
# subset_ab_bind_partner('1BJ1')
def subset_ab_bind_partner(pdb_number):
    if pdb_number[0] == 'H': pdb_number = pdb_number[3:7]
    ab_data = pd.read_csv('data/AB-Bind_experimental_data.csv', encoding="ISO-8859-1")
    ppdb = PandasPdb().read_pdb('data/' + pdb_number + '.pdb')
    residue_seq = ppdb.amino3to1()

    chain_index = residue_seq['chain_id'].to_string(index=False).replace("\n", '')
    partners = ab_data[ab_data['#PDB'] == pdb_number]['Partners(A_B)']
    partners = partners.str.split('_')
    same_index = {}
    for partner in partners:
        index = 0
        for i in partner:
            for ch in range(len(i)):
                # exec('%s = %d' % (i[ch], index))
                same_index[i[ch]] = index
            index = index + 1
    chain_to_subset = list(same_index.keys())
    match_list = [chain in chain_to_subset for chain in chain_index]
    return {'match_list': match_list, 'same_index': same_index}



# after subset_ab_bind_partner_data
def subset_to_partner_info(mutate_ab_bind_data, subset_ab_bind_partner_data):
    subset_pre_mutated_seq = [x for x, y in zip(mutate_ab_bind_data['pre_mut_seq'], subset_ab_bind_partner_data['match_list']) if y]
    subset_chain_index = [x for x, y in zip(mutate_ab_bind_data['chain_index'], subset_ab_bind_partner_data['match_list']) if y]
    subset_same_index = mutate_ab_bind_data['chain_index']
    for c, i in subset_ab_bind_partner_data['same_index'].items():
        subset_same_index = subset_same_index.replace(c, str(i))
    subset_same_index = [x for x, y in zip(subset_same_index, subset_ab_bind_partner_data['match_list']) if y]
    subset_after_mutated_seq = mutate_ab_bind_data['aft_mut_seq_list'].copy()

    n = 0
    for i in mutate_ab_bind_data['aft_mut_seq_list']:
        subset_after_mutated_seq[n][0] = [x for x, y in zip(mutate_ab_bind_data['aft_mut_seq_list'][n][0], subset_ab_bind_partner_data['match_list']) if y]
        n = n + 1
    subset_after_mutated_seq = [i for x in subset_after_mutated_seq for i in x]
    # subset_ddg = [x for x, y in zip(mutate_ab_bind_data['ddg'], subset_ab_bind_partner_data['match_list']) if y]
    subset_ddg = mutate_ab_bind_data['ddg']
    error = mutate_ab_bind_data['error']
    return {'subset_pre_mutated_seq': subset_pre_mutated_seq, 'subset_after_mutated_seq': subset_after_mutated_seq,
            'subset_chain_index': subset_chain_index, 'subset_same_index': subset_same_index, 'subset_ddg': subset_ddg,
            'error': error}


#################################################################################################################
if __name__ == "__main__":
    ab_data = pd.read_csv('data/AB-Bind_experimental_data.csv', encoding = "ISO-8859-1")
    ab_pdb_name = set(ab_data['#PDB'])

    # download
    # path = '/home/tialan/Data/gan/data'
    #
    # for i in ab_pdb_name:
    #     if i[0:3] == 'HM_':
    #         i = i[3:]
    #     ab_path = 'http://files.rcsb.org/download/'+ i +'.pdb'
    #     file_name = 'data/' + i +'.pdb'
    #     urllib.request.urlretrieve(ab_path, file_name)

    ab_info_dict = {}
    ab_all_result_dict = {}
    for i in ab_pdb_name:
        if i[0] == 'H': i = i[3:7]
        ppdb = PandasPdb().read_pdb('data/' + i + '.pdb')
        residue_seq = ppdb.amino3to1()
        alpha_carbon_coordinate = ppdb.df['ATOM'][['x_coord', 'y_coord', 'z_coord']].iloc[residue_seq.index + 1].to_numpy()

        a = mutate_ab_bind(i)
        b = subset_ab_bind_partner(i)
        result = subset_to_partner_info(a, b)

        index = np.array(b['match_list'])
        rbd_index = find_rbd(alpha_carbon_coordinate[index], 50, result['subset_same_index'], 0.1)
        ab_info_dict['result'] = result
        ab_info_dict['rbd_index'] = rbd_index
        ab_info_dict['subset_alpha_carbon_coordinate'] = alpha_carbon_coordinate[index]
        ab_all_result_dict[i] = ab_info_dict.copy()

    with open('data/ab_all_result_dict.pkl', 'wb') as f:
        pickle.dump(ab_all_result_dict, f)






