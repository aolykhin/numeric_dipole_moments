def sort_by_lst(filename, lst):
    with open(filename, 'r') as f:
        lines = f.readlines()
    if len(lines)!=len(lst): raise RuntimeError('numbers of records do not match')
    names = [line[:30].strip() for line in lines]
    if len(list(set(names))) != len(names):
        raise RuntimeError('Duplicates were fond')
    ind_lst = []
    for mol in lst:
        for i, line in enumerate(lines):
            if mol==line[:30].strip():
                ind_lst.append(i)
    lines = [lines[i] for i in ind_lst]
    with open(filename, 'w') as f:
        for line in lines:
            f.writelines(line)

def main():
    ordered_lst = [
    'indole'              ,
    'x2_cyanoindole'      ,
    'x3_cyanoindole'      ,
    'x4_cyanoindole'      ,
    'x5_cyanoindole'      ,
    'x4_fluoroindole'     ,
    'x5_fluoroindole'     ,
    'x6_fluoroindole'     ,
    'x6_methylindole'     ,
    'anti_5_hydroxyindole',
    'anti_5_methoxyindole',
    'syn_6_methoxyindole' ,
    'x7_azaindole'        ,
    'cis_2_naphthol'      ,
    'trans_2_naphthol'    ,
    'benzonitrile'        ,
    'phenol'              ,
    'anisole'             ,
    'x13_dimethoxybenzene',
    'x14_dimethoxybenzene'
    ]

    import glob
    file_list = []
    file_list += glob.glob('*angles*')
    file_list += glob.glob('*energies*')
    file_list += glob.glob('*PDM*')
    file_list += glob.glob('*TDM*')
    # print(file_list)

    for file in file_list:
        sort_by_lst(file, ordered_lst)

if __name__ == "__main__":
    main()