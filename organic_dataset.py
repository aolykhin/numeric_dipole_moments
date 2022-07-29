from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
from pyscf.tools import molden
from pyscf.geomopt import geometric_solver
from pyscf.gto import inertia_moment

def transform_dip(eq_dip,mol_eq):
    matrix = inertia_moment(mol_eq, mass=None, coords=None)
    moments, abc_vec=np.linalg.eigh(matrix) #ascending order
    origin=np.eye(3)
    rot_mat=np.linalg.solve(origin,abc_vec)
    eq_dip_abc=np.dot(eq_dip,rot_mat)
    return eq_dip_abc

def cms_dip(x):
    out = x.iname+'_'+str(x.nel)+'e'+str(x.norb)+'o'+'_'+str(x.istate)
    mol = gto.M(atom=open('geom_'+x.iname+'.xyz').read(), charge=x.icharge, spin=x.ispin,
                    output=out+'.log', verbose=4, basis=x.ibasis)
    weights=[1/x.iroots]*x.iroots #Equal weights only
    #HF step
    mf = scf.RHF(mol).run()
    molden.from_mo(mol, out+'_hf.molden', mf.mo_coeff)


    # mc = mcscf.CASSCF(mf, x.norb, x.nel)
    # mc.fcisolver = csf_solver(mol, smult=x.ispin+1)
    # mo = mcscf.sort_mo(mc, mf.mo_coeff, x.cas_list)
    # e = mc.kernel(mo)[0]
    # molden.from_mo(mol, out+'_cas_ini.molden', mc.mo_coeff)

    # mc = mcscf.CASSCF(mf, x.norb, x.nel)
    # mc.fcisolver = csf_solver(mol, smult=x.ispin+1)
    # mc.state_average_([0.5, 0.5])
    # mo = mcscf.sort_mo(mc, mf.mo_coeff, x.cas_list)
    # e = mc.kernel(mo)[0]

    # molden.from_mo(mol, out+'_cas_ini.molden', mc.mo_coeff)

    # mo=mc.mo_coeff


    # ###-------------------- MC-PDFT Energy ---------------------------
    mc = mcpdft.CASSCF(mf, x.ifunc, x.norb, x.nel, grids_level=9)
    mc.fcisolver = csf_solver(mol, smult=x.ispin+1)
    mc.max_cycle_macro = 200
    mo = mcscf.sort_mo(mc, mf.mo_coeff, x.cas_list)
    e_pdft = mc.kernel(mo)[0]
    mo_ss = mc.mo_coeff #SS MOs
    # molden.from_mo(mol, out+'_ss_cas.molden', mc.mo_coeff)


    #-------------------- CMS-PDFT Energy ---------------------------
    # mc.natorb=True
    mc=mc.state_interaction(weights,'cms').run(mo_ss)
    # mo_sa = mc.mo_coeff #SS MOs
    molden.from_mo(mol, out+'_cms.molden', mc.mo_coeff)
    print('CMS energy is completed')

    if x.opt == True: #Optimize geometry
        
        mol_eq=mc.nuc_grad_method().as_scanner(state=x.istate).optimizer().kernel()

        #Equilibrium dipole moment
        mf_eq = scf.RHF(mol_eq).run()
        mc_eq = mcpdft.CASSCF(mf_eq, x.ifunc, x.norb, x.nel, grids_level=9)
        mc_eq.fcisolver = csf_solver(mol_eq, smult=x.ispin+1)
        mc_eq.max_cycle_macro = 200
        mo = mcscf.project_init_guess(mc_eq, mo_ss)
        e_pdft = mc_eq.kernel(mo)[0]
        mo_ss = mc_eq.mo_coeff #SS MOs
        mc_eq = mc_eq.state_interaction(weights,'cms').run(mo_ss)

        e_cms=mc_eq.e_states.tolist() #List of CMS energies
        molden.from_mo(mol_eq, out+'_opt.molden', mc_eq.mo_coeff)
        print('CMS-PDFT energies at the optimized geometry\n',e_cms)

        eq_dip=mc_eq.dip_moment(state=x.istate, unit='Debye')

    else: #Preoptimized geometry
        e_cms_preopt=mc.e_states.tolist() #List of CMS energies
        molden.from_mo(mol, out+'_opt.molden', mc.mo_coeff)
        print('CMS-PDFT energies at the PREoptimized geometry\n',e_cms_preopt)

        eq_dip=mc.dip_moment(state=x.istate, unit='Debye')
    
    #Save the optimized geometry to the xyz file
    with open('geom_opt_'+out+'.xyz', 'w') as f:
        for i in range(mol_eq.natm):
            print('%s  %s' % (mol_eq.atom_symbol(i), str(mol_eq.atom_coord(i,unit='Angstrom'))[1:-1]), file=f)

    #Save final energies
    with open('energies', 'a') as f:
        print('%s  %s' % (out, e_cms), file=f)
            

    #Compute and save dipole moments at the optimized geometry 
    eq_dip_abc=transform_dip(eq_dip,mol)
    val=np.linalg.norm(eq_dip)
    print(x.iname, 'state is ', x.istate, 'ABS = %.2f' % (val) )
    print('Dipole along XYZ-coord axes: %.4f %.4f %.4f \n' % (eq_dip[0],eq_dip[1],eq_dip[2]))
    print('Dipole along principle axes: %.4f %.4f %.4f \n' % (eq_dip_abc[0],eq_dip_abc[1],eq_dip_abc[2]))

    #Comopute state-specific NOONs
    mc = mcscf.CASCI(mf, x.norb, x.nel).state_specific_(x.istate)
    mc.natorb=True
    mc.fix_spin_(ss=x.ispin)
    emc = mc.casci(mo_ss)[0]
    mc.analyze(large_ci_tol=0.05)
    print(emc)
    print(mc.mo_occ)

    return
class Molecule:
    def __init__(self, iname, iroots, nel, norb, istate, cas_list,
                opt=True, icharge=0, isym='C1', irep='A', ispin=0, ibasis='julccpvdz', ifunc='tPBE'):
        self.iname    = iname
        self.iroots   = iroots
        self.nel      = nel
        self.norb     = norb
        self.istate   = istate
        self.cas_list = cas_list
        self.opt      = opt
        self.icharge  = icharge
        self.isym     = isym
        self.irep     = irep
        self.ispin    = ispin
        self.ibasis   = ibasis
        self.ifunc    = ifunc


# OH_phenol_3_10e9o_0           = Molecule('OH_phenol'           , 3, 10,9, 0, [19,20,23,24,25,26,31,33,34])
x2_aminobenzonitrile_12e11o_0 = Molecule('x2_aminobenzonitrile', 2, 12,11, 0, [23,27,28,29,30,31, 35,38,39,49,50])
x3_aminobenzonitrile_12e11o_0 = Molecule('x3_aminobenzonitrile', 2, 12,11, 0, [23,27,28,29,30,31, 34,35,38,39,49])
x4_aminobenzonitrile_12e11o_0 = Molecule('x4_aminobenzonitrile', 2, 12,11, 0, [24,27,28,29,30,31, 35,36,38,39,49])
aniline_8e7o_en_0             = Molecule('aniline'             , 2, 8,7,   0, [20,23,24,25,31,33,34], opt=False)
aniline_8e7o_0                = Molecule('aniline'             , 2, 8,7,   0, [20,23,24,25,31,33,34])
trans_aminophenol_10e8o_0     = Molecule('trans_aminophenol'   , 2, 10,8,  0, [24,25,27,28,29,35,47,49])
cis_aminophenol_10e8o_0       = Molecule('cis_aminophenol'     , 2, 10,8,  0, [22,24,27,28,29,35,37,38])
aminobenzoic_10e8o_0          = Molecule('aminobenzoic'        , 2, 10,8,  0, [22,24,27,28,29,35,37,38])
OH_phenol_10e9o_0             = Molecule('OH_phenol'           , 2, 10,9, 0, [19,20,23,24,25,26,31,33,34])
fluorobenzene_8e7o_0          = Molecule('fluorobenzene'       , 2, 8,7,  0, [16,23,24,25,31,32,34])
fluorobenzene_8e7o_1          = Molecule('fluorobenzene'       , 2, 8,7,  1, [16,23,24,25,31,32,34])
fluorobenzene_6e6o_0_tz       = Molecule('fluorobenzene'       , 2, 6,6,  0, [23,24,25,31,32,34], ibasis='julccpvtz',)
fluorobenzene_6e6o_1_tz       = Molecule('fluorobenzene'       , 2, 6,6,  1, [23,24,25,31,32,34], ibasis='julccpvtz')
#!!!
x7_azaindole_10e9o_0          = Molecule('x7_azaindole'        , 2, 10,9, 0, [22,27,29,30,31,35,38,43,44])
x7_azaindole_10e9o_1          = Molecule('x7_azaindole'        , 2, 10,9, 1, [22,27,29,30,31,35,38,43,44])
benzonitrile_10e10o_0         = Molecule('benzonitrile'        , 2, 10,10,0, [21,24,25,26,27,29,32,43,45,46])
benzonitrile_10e10o_1         = Molecule('benzonitrile'        , 2, 10,10,1, [21,24,25,26,27,29,32,43,45,46])
x5_cyanoindole_14e13o_0       = Molecule('x5_cyanoindole'      , 2, 14,13,0, [26,31,33,34,35,36,37, 41,44,45,46,49,56])
x5_cyanoindole_14e13o_1       = Molecule('x5_cyanoindole'      , 2, 14,13,1, [26,31,33,34,35,36,37, 41,44,45,46,49,56])
dimethoxybenzene_10e8o_0      = Molecule('dimethoxybenzene'    , 2, 10,8, 0, [24,25,35,36,37,47,50,56])
dimethoxybenzene_10e8o_1      = Molecule('dimethoxybenzene'    , 2, 10,8, 1, [24,25,35,36,37,47,50,56])
fluorobenzene_6e6o_0          = Molecule('fluorobenzene'       , 2, 6,6,  0, [23,24,25,31,32,34])
fluorobenzene_6e6o_1          = Molecule('fluorobenzene'       , 2, 6,6,  1, [23,24,25,31,32,34])
x4_fluoroindole_10e9o_0       = Molecule('x4_fluoroindole'     , 2, 10,9, 0, [27,32,33,34,35,41,43,46,47])
x4_fluoroindole_10e9o_1       = Molecule('x4_fluoroindole'     , 2, 10,9, 1, [27,32,33,34,35,41,43,46,47])
x5_fluoroindole_10e9o_0       = Molecule('x5_fluoroindole'     , 2, 10,9, 0, [27,32,33,34,35,40,44,45,51])
x5_fluoroindole_10e9o_1       = Molecule('x5_fluoroindole'     , 2, 10,9, 1, [27,32,33,34,35,40,44,45,51])
x6_fluoroindole_10e9o_0       = Molecule('x6_fluoroindole'     , 2, 10,9, 0, [28,32,33,34,35,40,44,45,48])
x6_fluoroindole_10e9o_1       = Molecule('x6_fluoroindole'     , 2, 10,9, 1, [28,32,33,34,35,40,44,45,48])
x1_fluoronaphthalene_10e10o_0 = Molecule('1-fluoronaphthalene' , 2, 10,10,0, [32,35,36,37,38,42,45,47,50,53])
x1_fluoronaphthalene_10e10o_1 = Molecule('1-fluoronaphthalene' , 2, 10,10,1, [32,35,36,37,38,42,45,47,50,53])
x2_fluoronaphthalene_10e10o_0 = Molecule('2-fluoronaphthalene' , 2, 10,10,0, [31,35,36,37,38,42,45,48,50,52])
x2_fluoronaphthalene_10e10o_1 = Molecule('2-fluoronaphthalene' , 2, 10,10,1, [31,35,36,37,38,42,45,48,50,52])
formaldehyde_6e6o_0           = Molecule('formaldehyde'        , 2, 6,6,  0, [6,7,8,9,10,11])
formaldehyde_6e6o_1           = Molecule('formaldehyde'        , 2, 6,6,  1, [6,7,8,9,10,11])
anti_5_hydroxyindole_12e10o_0 = Molecule('anti_5_hydroxyindole', 2, 12,10,0, [35,34,33,32,27,25,41,44,46,47])
anti_5_hydroxyindole_12e10o_1 = Molecule('anti_5_hydroxyindole', 2, 12,10,1, [35,34,33,32,27,25,41,44,46,47])
indole_10e9o_0                = Molecule('indole'              , 2, 10, 9,0, [23,28,29,30,31,37,40,42,45])
indole_10e9o_1                = Molecule('indole'              , 2, 10, 9,1, [23,28,29,30,31,37,40,42,45])
anti_4_methoxyindole_12e10o_0 = Molecule('anti_4_methoxyindole', 2, 12,10,0, [25,28,36,37,38,39,45,46,50,51])
anti_4_methoxyindole_12e10o_1 = Molecule('anti_4_methoxyindole', 2, 12,10,1, [25,28,36,37,38,39,45,46,50,51])
anti_5_methoxyindole_12e10o_0 = Molecule('anti_5_methoxyindole', 2, 12,10,0, [28,33,36,37,38,39,45,46,50,52])
anti_5_methoxyindole_12e10o_1 = Molecule('anti_5_methoxyindole', 2, 12,10,1, [28,33,36,37,38,39,45,46,50,52])
syn_6_methoxyindole_12e10o_0  = Molecule('syn_6_methoxyindole' , 2, 12,10,0, [29,33,36,37,38,39,45,47,50,52])
syn_6_methoxyindole_12e10o_1  = Molecule('syn_6_methoxyindole' , 2, 12,10,1, [29,33,36,37,38,39,45,47,50,52])
cis_2_naphthol_12e11o_0       = Molecule('cis_2_naphthol'      , 2, 12,11,0, [27,32,35,36,37,38,42,46,48,50,53])
cis_2_naphthol_12e11o_1       = Molecule('cis_2_naphthol'      , 2, 12,11,1, [27,32,35,36,37,38,42,46,48,50,53])
trans_2_naphthol_12e11o_0     = Molecule('trans_2_naphthol'    , 2, 12,11,0, [28,32,35,36,37,38,42,45,48,50,53])
trans_2_naphthol_12e11o_1     = Molecule('trans_2_naphthol'    , 2, 12,11,1, [28,32,35,36,37,38,42,45,48,50,53])
phenol_8e7o_0                 = Molecule('phenol'              , 2,  8, 7,0, [19,23,24,25,31,33,34])
phenol_8e7o_1                 = Molecule('phenol'              , 2,  8, 7,1, [19,23,24,25,31,33,34])
propynal_8e7o_0               = Molecule('propynal'            , 2,  8, 7,0, [11,12,13,14,16,21,22])
propynal_8e7o_1               = Molecule('propynal'            , 2,  8, 7,1, [11,12,13,14,16,21,22])
#formaldehyde_6e6o_0           = Molecule('formaldehyde'        , 2, 6,6,  0, [6,7,8,9,10,12])

species=[phenol_8e7o_0]
species=[x3_aminobenzonitrile_12e11o_0]
species=[x4_aminobenzonitrile_12e11o_0]
species=[x4_aminobenzonitrile_12e11o_0]
species=[x3_aminobenzonitrile_12e11o_0]
species=[x2_aminobenzonitrile_12e11o_0]
species=[aminobenzoic_10e8o_0]
species=[cis_aminophenol_10e8o_0]
species=[trans_aminophenol_10e8o_0]
species=[benzonitrile_10e10o_0]
species=[x7_azaindole_10e9o_0]
species=[x7_azaindole_10e9o_1]
species=[fluorobenzene_6e6o_0]
species=[fluorobenzene_6e6o_1]
species=[x4_fluoroindole_10e9o_0]
species=[x4_fluoroindole_10e9o_1]
species=[x5_fluoroindole_10e9o_0]
species=[x5_fluoroindole_10e9o_1]
species=[x6_fluoroindole_10e9o_0]
species=[x6_fluoroindole_10e9o_1]
species=[x5_cyanoindole_14e13o_0]
species=[x5_cyanoindole_14e13o_1]
species=[dimethoxybenzene_10e8o_0]
species=[dimethoxybenzene_10e8o_1]
species=[fluorobenzene_6e6o_1_tz]
species=[fluorobenzene_8e7o_1]
species=[OH_phenol_10e9o_0]
species=[aniline_8e7o_en_0]
species=[formaldehyde_6e6o_0]
species=[x1_fluoronaphthalene_10e10o_0]
species=[propynal_8e8o_0]
for x in species:
    cms_dip(x)
