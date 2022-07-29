from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
from pyscf.tools import molden
from pyscf.geomopt import geometric_solver
from pyscf.gto import inertia_moment
from pyscf.data import nist

def transform_dip(eq_dip,mol_eq):
    matrix = inertia_moment(mol_eq, mass=None, coords=None)
    moments, abc_vec=np.linalg.eigh(matrix) #ascending order
    origin=np.eye(3)
    rot_mat=np.linalg.solve(origin,abc_vec)
    eq_dip_abc=np.dot(eq_dip,rot_mat)
    return eq_dip_abc

def tdm_casci(mo,mc_eq,mol,state=[0,1]):
    from functools import reduce
    mass = mol.atom_mass_list()
    coords = mol.atom_coords()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    orb = mo[:,mc_eq.ncore:mc_eq.ncore+mc_eq.ncas]
    with mol.with_common_orig(mass_center):
        dip_ints = mol.intor('cint1e_r_sph', comp=3)
    ci_buf=mc_eq.get_ci_adiabats(uci='MCSCF')

    m=state[0]
    n=state[1]
    t_dm1 = mc_eq.fcisolver.trans_rdm1(ci_buf[n], ci_buf[m], mc_eq.ncas, mc_eq.nelecas)
    t_dm1_ao = reduce(np.dot, (orb, t_dm1, orb.T))
    tdm = np.einsum('xij,ji->x', dip_ints, t_dm1_ao)
    return tdm

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
    # mc = mcpdft.CASSCF(mf, x.ifunc, x.norb, x.nel, grids_level=9)
    mc = mcpdft.CASSCF(mf, x.ifunc, x.norb, x.nel)
    mc.fcisolver = csf_solver(mol, smult=x.ispin+1)
    mc.max_cycle_macro = 200
    mo = mcscf.sort_mo(mc, mf.mo_coeff, x.cas_list)
    e_pdft = mc.kernel(mo)[0]
    mo_ss = mc.mo_coeff #SS MOs
    # molden.from_mo(mol, out+'_ss.molden', mc.mo_coeff)


    #-------------------- CMS-PDFT Energy ---------------------------
    # mc.natorb=True
    mc=mc.multi_state(weights,'cms')
    mc.kernel(mo_ss)
    ci=mc.ci
    # mo_sa = mc.mo_coeff #SS MOs
    molden.from_mo(mol, out+'_cms.molden', mc.mo_coeff)
    print('CMS energy is completed')

    if x.opt == True: #Optimize geometry
        
        mol_eq=mc.nuc_grad_method().as_scanner(state=x.istate).optimizer().kernel()
        # mol_eq=mc.nuc_grad_method().as_scanner(state=x.istate).optimizer().kernel()

        #Equilibrium dipole moment
        mf_eq = scf.RHF(mol_eq).run()
        # mc_eq = mcpdft.CASSCF(mf_eq, x.ifunc, x.norb, x.nel, grids_level=9)
        mc_eq = mcpdft.CASSCF(mf_eq, x.ifunc, x.norb, x.nel)
        mc_eq.fcisolver = csf_solver(mol_eq, smult=x.ispin+1)
        mc_eq.max_cycle_macro = 200
        mo = mcscf.project_init_guess(mc_eq, mo_ss)
        e_pdft = mc_eq.kernel(mo)[0]
        mo_ss = mc_eq.mo_coeff #SS MOs
        mc_eq = mc_eq.multi_state(weights,'cms')
        mc_eq.kernel(mo_ss,ci0=ci)
        # mc_eq.kernel(mo_ss)
        # mc_eq = mc_eq.state_interaction(weights,'cms').run(mo_ss)

    else: #Preoptimized geometry
        mc_eq=mc
        mol_eq=mol

    mo_sa = mc_eq.mo_coeff #SS MOs
    eq_dip=mc_eq.dip_moment(state=x.istate, unit='Debye')
    e_opt=mc_eq.e_states.tolist() #List of CMS energies
    print('CMS-PDFT energies at the pre- or optimized geometry\n',e_opt)
    molden.from_mo(mol_eq, out+'_opt.molden', mc_eq.mo_coeff)

    mass = mol_eq.atom_mass_list()
    coords = mol_eq.atom_coords(unit='ANG')
    com = np.einsum('i,ij->j', mass, coords)/mass.sum()


    #Compute & save TDMs from the optimized excited to ground state
    if x.istate==0:
        tdm=[None]*(x.iroots)
        for n in range(x.iroots):
            if n!=0:# Units should be set off to get dimensionless oscilator strengths 
                # tdm[n]=mc.trans_moment(state=[0,n])
                tdm[n]=tdm_casci(mo_sa,mc_eq,mol_eq,state=[0,n])
                abs = np.linalg.norm(tdm[n])
                oscil= (2/3)*(e_opt[n]-e_opt[0])*(abs**2)
                tdm[n]*= nist.AU2DEBYE
                abs*= nist.AU2DEBYE
                with open('tdms_at_opt_ground', 'a') as f:
                    print(x.iname, f'TDM <0|mu|{n}> ABS (D) = {abs:7.2f} Oscil = {oscil:7.2f} \
                        XYZ (D) = {tdm[n][0]:7.3f} {tdm[n][1]:7.3f} {tdm[n][2]:7.3f}', file=f)
                with open('tdms_at_opt_ground_COM', 'a') as f:
                    tdm[n]+=com
                    print(x.iname, f'<0|mu|{n}> vec V {com[0]:7.3f} {com[1]:7.3f} {com[2]:7.3f} {tdm[n][0]:7.3f} {tdm[n][1]:7.3f} {tdm[n][2]:7.3f} S', file=f)
                    # print(x.iname, 'TDM %s0 ABS = %.2f Oscillator = %.2f XYZ: %.4f %.4f %.4f' % \
                    # (x.istate,abs,oscil, tdm[0],tdm[1],tdm[2]), file=f)

    else:    
        # tdm=mc.trans_moment(state=[x.istate,0])
        tdm=tdm_casci(mo_sa,mc_eq,mol_eq,state=[x.istate,0])
        abs = np.linalg.norm(tdm)
        oscil= (2/3)*(e_opt[x.istate]-e_opt[0])*(abs**2)
        tdm*= nist.AU2DEBYE
        abs*= nist.AU2DEBYE
        with open('tdms_from_excited_states', 'a') as f:
            print(x.iname, f'TDM <{x.istate}|mu|0> ABS = {abs:7.2f} Oscil = {oscil:7.2f} \
                XYZ: {tdm[0]:7.3f} {tdm[1]:7.3f} {tdm[2]:7.3f}', file=f)
            with open('tdms_from_excited_states_COM', 'a') as f:
                tdm+=com
                print(x.iname, f'<{x.istate}|mu|0> vec V {com[0]:7.3f} {com[1]:7.3f} {com[2]:7.3f} {tdm[0]:7.3f} {tdm[1]:7.3f} {tdm[2]:7.3f} S', file=f)
            # print(x.iname, 'TDM %s0 ABS = %.2f Oscillator = %.2f XYZ: %.4f %.4f %.4f' % \
            # (x.istate,abs,oscil, tdm[0],tdm[1],tdm[2]), file=f)



    # #Save TDMs
    # with open('tdms_from_excited', 'a') as f::q
    #     print(x.iname, 'TDM is %s0 ', 'ABS = %.2f', 'Oscillator = %.2f' % (x.istate,abs,oscil), file=f )
    #     print('TDM along XYZ-coord axes: %.4f %.4f %.4f \n' % (tdm_opt[0],tdm_opt[1],tdm_opt[2]), file=f)
  
    #Save the optimized geometry to the xyz file
    with open('geom_opt_'+out+'.xyz', 'w') as f:
        for i in range(mol_eq.natm):
            print('%s  %s' % (mol_eq.atom_symbol(i), str(mol_eq.atom_coord(i,unit='Angstrom'))[1:-1]), file=f)

    #Save final energies
    with open('energies', 'a') as f:
        print('%s  %s' % (out, e_opt), file=f)
            

    #Compute and save dipole moments at the optimized geometry 
    eq_dip_abc=transform_dip(eq_dip,mol)
    val=np.linalg.norm(eq_dip)
    print(x.iname, 'state is ', x.istate, 'ABS = %.2f' % (val) )
    print('Dipole along XYZ-coord axes: %8.4f %8.4f %8.4f \n' % (eq_dip[0],eq_dip[1],eq_dip[2]))
    print('Dipole along principle axes: %8.4f %8.4f %8.4f \n' % (eq_dip_abc[0],eq_dip_abc[1],eq_dip_abc[2]))

    #Save final dipoles
    with open('dipoles', 'a') as f:
        print('%s  %8.2f' % (out, val), file=f)
    with open('dipoles_abs', 'a') as f:
        print('%s  %8.2f  %8.2f  %8.2f' % (out, eq_dip_abc[0],eq_dip_abc[1],eq_dip_abc[2]), file=f)

    #Compute state-specific NOONs
    mc = mcscf.CASCI(mf, x.norb, x.nel).state_specific_(x.istate)
    mc.natorb=True
    mc.fix_spin_(ss=x.ispin)
    emc = mc.casci(mo_sa)[0]
    mc.analyze(large_ci_tol=0.05)
    print(emc)
    print(mc.mo_occ)

    return
class Molecule:
    def __init__(self, iname, nel, norb, cas_list, iroots=3, istate=0, 
                opt=True, icharge=0, isym='C1', irep='A', ispin=0, ibasis='julccpvdz', ifunc='tPBE'):
        self.iname    = iname
        self.nel      = nel
        self.norb     = norb
        self.cas_list = cas_list
        self.iroots   = iroots
        self.istate   = istate
        self.opt      = opt
        self.icharge  = icharge
        self.isym     = isym
        self.irep     = irep
        self.ispin    = ispin
        self.ibasis   = ibasis
        self.ifunc    = ifunc

x=[None]*20
x[0]  = Molecule('x7_azaindole'        , 10,9,  [22,27,29,30,31,35,38,43,44])
x[1]  = Molecule('benzonitrile'        , 10,10, [21,24,25,26,27,29,32,43,45,46])
x[2]  = Molecule('dimethoxybenzene'    , 10,8,  [24,25,35,36,37,47,50,56])
x[3]  = Molecule('fluorobenzene'       , 6,6,   [23,24,25,31,32,34])
x[4]  = Molecule('x5_cyanoindole'      , 14,13, [26,31,33,34,35,36,37, 41,44,45,46,49,56])
x[5]  = Molecule('x4_fluoroindole'     , 10,9,  [27,32,33,34,35,41,43,46,47])
x[6]  = Molecule('x5_fluoroindole'     , 10,9,  [27,32,33,34,35,40,44,45,51])
x[7]  = Molecule('x6_fluoroindole'     , 10,9,  [28,32,33,34,35,40,44,45,48])
x[8]  = Molecule('1-fluoronaphthalene' , 10,10, [32,35,36,37,38,42,45,47,50,53])
x[9]  = Molecule('2-fluoronaphthalene' , 10,10, [31,35,36,37,38,42,45,48,50,52])
x[10] = Molecule('formaldehyde'        ,  6, 6, [6,7,8,9,10,11])
x[11] = Molecule('anti_5_hydroxyindole', 12,10, [35,34,33,32,27,25,41,44,46,47])
x[12] = Molecule('indole'              , 10,9,  [23,28,29,30,31,37,40,42,45])
x[13] = Molecule('anti_4_methoxyindole', 12,10, [25,28,36,37,38,39,45,46,50,51])
x[14] = Molecule('anti_5_methoxyindole', 12,10, [28,33,36,37,38,39,45,46,50,52])
x[15] = Molecule('syn_6_methoxyindole' , 12,10, [29,33,36,37,38,39,45,47,50,52])
x[16] = Molecule('cis_2_naphthol'      , 12,11, [27,32,35,36,37,38,42,46,48,50,53])
x[17] = Molecule('trans_2_naphthol'    , 12,11, [28,32,35,36,37,38,42,45,48,50,53])
x[18] = Molecule('phenol'              ,  8,7,  [19,23,24,25,31,33,34])
x[19] = Molecule('propynal'            ,  8,7,  [11,12,13,14,16,21,22])


# species=[x[0]]
# for y in species:
#     cms_dip(y)

x[12].istate = 0
cms_dip(x[12])

# x[10].istate = 0
# x[10].opt = False
# cms_dip(x[10])
