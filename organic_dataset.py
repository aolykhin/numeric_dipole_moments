from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.prop.dip_moment.mcpdft import nuclear_dipole
from scipy import linalg
import numpy as np
import os
from pyscf.tools import molden
from pyscf.gto import inertia_moment
from pyscf.data import nist
from pyscf.gto import inertia_moment
from pyscf.data import nist
from numpy.linalg import norm as norm
import math

def dm_casci(mo,mc,mol,state):
    '''Return diagonal (PDM) and off-diagonal (TDM) dipole moments'''
    from functools import reduce
    orb = mo[:,mc.ncore:mc.ncore+mc.ncas]
    mass = mol.atom_mass_list()
    coords = mol.atom_coords()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    ci = mc.get_ci_adiabats(uci='MCSCF')
    
    if state[1] != state[0]: 
        with mol.with_common_orig(mass_center):
            dip_ints = mol.intor('cint1e_r_sph', comp=3)
        t_dm1 = mc.fcisolver.trans_rdm1(ci[state[1]], ci[state[0]], mc.ncas, mc.nelecas)
        t_dm1_ao = reduce(np.dot, (orb, t_dm1, orb.T))
        dip = np.einsum('xij,ji->x', dip_ints, t_dm1_ao)
    else:
        ncore = mc.ncore
        ncas = mc.ncas
        nocc = ncore + ncas
        mo_core = mo[:,:ncore]
        mo_cas = mo[:,ncore:nocc]
        casdm1 = mc.fcisolver.make_rdm1([ci[state[1]]], mc.ncas, mc.nelecas)
        dm_core = np.dot(mo_core, mo_core.T) * 2
        dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
        dm = dm_core + dm_cas
        with mol.with_common_orig(mass_center):
            ao_dip = mol.intor_symmetric('int1e_r', comp=3)
        dip = -np.einsum('xij,ij->x', ao_dip, dm).real
        dip += nuclear_dipole(mc,origin='mass_center')
    dip *= nist.AU2DEBYE
    return dip

def render_tdm_pdm(x, mass_center, abc_vec, pdm, tdm, method):
    #set chemcraft parameters
    scale_axes = 2
    scale_tdm  = 2
    scale_pdm  = 1
    label_size = 0.15
    label_axes = ['a','b']
    black  = '0.01  0.1   0.0   0.0   0.0  0'
    blue   = '0.02  0.1   0.0   0.5   1.0  0'
    red    = '0.02  0.1   1.0   0.0   0.0  0'
    green  = '0.02  0.1   0.0   1.0   0.0  0'
    purple = '0.02  0.1   0.5   0.0   1.0  0'
    grey   = '0.02  0.1   0.5   0.5   0.5  0'

    #shift abc frame to the COM and print out coordinated of abc vectors
    out = x.iname+'_'+str(x.nel)+'e'+str(x.norb)+'o'+'_'+str(x.istate)
    with open('chemcraft_dipoles_'+method, 'a') as f:
        print(out, file=f)
        print('#Inertial axes', file=f)
        for i in range(2): #only a and b
            axis_main =  abc_vec[:,i]*scale_axes + mass_center
            axis_auxi = -abc_vec[:,i]*scale_axes + mass_center
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} N '.format(*mass_center,*axis_main)+black, file=f)
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} N '.format(*mass_center,*axis_auxi)+black, file=f)
            print('L {:8.4f} {:8.4f} {:8.4f} {:.2f} {}'.format(*axis_main*1.1, label_size, label_axes[i]), file=f)
        #ground state geometry may be not a good frame to render excited state dipole
        print('#Permanent dipole moment', file=f)
        color = [purple, grey, green]
        pdm_shifted = np.empty_like(pdm)
        for i in range(len(pdm)):
            if len(pdm)>len(color): RuntimeError('Add more colors to pdm')
            pdm_shifted[i] = pdm[i]*scale_pdm + mass_center
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} S '.format(*mass_center,*pdm_shifted[i]) + color[i], file=f)

        print('#Transiton dipole moment', file=f)
        color = [blue, red]
        tdm_main = np.empty_like(tdm)
        tdm_auxi = np.empty_like(tdm)
        for i in range(len(tdm)):
            if len(tdm)>len(color): RuntimeError('Add more colors to tdm')
            tdm_main[i] =  tdm[i]/norm(tdm[i])*scale_tdm + mass_center
            tdm_auxi[i] = -tdm[i]/norm(tdm[i])*scale_tdm + mass_center
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} S '.format(*mass_center,*tdm_main[i]) + color[i], file=f)
            print('V {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} {:8.4f} S '.format(*mass_center,*tdm_auxi[i]) + color[i], file=f)
        print(' ', file=f)
    return

# counterclockwise rotation from vec1 to vec2
def findClockwiseAngle(ini, fin):
    ind=2 # c-component in abc frame
    print(ini,fin)
    if len(ini)==3 and abs(ini[ind])<2e-2: ini=np.delete(ini,ind) 
    if len(fin)==3 and abs(fin[ind])<2e-2: fin=np.delete(fin,ind)
    if len(ini) != len(fin): raise ValueError('Initial and final vectors have different lengths')
    if norm(ini) <1e-3: #Trivial case of zero TDM 
        ang = 0
    else:
        ang = -math.degrees(math.asin(np.cross(ini,fin)/(norm(ini)*norm(fin))))
    return ang 

def transform_dip(x, mol, pdm, tdm, method):
    out = x.iname+'_'+str(x.nel)+'e'+str(x.norb)+'o'+'_'+str(x.istate)
    mass = mol.atom_mass_list()
    coords = mol.atom_coords(unit='ANG')
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()

    matrix = inertia_moment(mol, mass=mass, coords=coords)
    moments, abc_vec=np.linalg.eigh(matrix) #ascending order

    # Get vectors for Chemcraft
    render_tdm_pdm(x, mass_center, abc_vec, pdm, tdm, method)
    # Projection of PDM onto ABC frame 
    origin  = np.eye(3)
    rot_mat = np.linalg.solve(origin,abc_vec)
    pdm_abc, pdm_ang = get_ang_with_a_axis(pdm, rot_mat)
    tdm_abc, tdm_ang = get_ang_with_a_axis(tdm, rot_mat)
    with open(method+'_angles', 'a') as f:
        for i in range(len(pdm_ang)):
            print(f'{out:<30} theta with pdm is {pdm_ang[i]:5.1f} degrees', file=f)
        for i in range(len(tdm_ang)):
            print(f'{out:<30} theta with tdm is {tdm_ang[i]:5.1f} degrees', file=f)
    return pdm_abc, tdm_abc

def get_ang_with_a_axis(dm, rot_mat):
    dm_abc = np.empty_like(dm)
    angles = [None]*len(dm)

    for i in range(len(dm)):
        dm_abc[i]=np.dot(dm[i],rot_mat)
        #Angle of pdm wrt a-axis
        ini = dm_abc[i]
        fin = np.array([1,0,0])
        dot = np.dot(ini,fin)/(norm(ini)*norm(fin))
        if abs(dot) < 1e-3: #accurate to 0.06 degrees
            ang = 90
        elif abs(dot-1) < 1e-3:
            ang = 0
        elif abs(dot+1) < 1e-3:
            ang = 180
        elif dot < 0: 
            fin = -fin
            ang = findClockwiseAngle(ini, fin)
        else:
            ang = findClockwiseAngle(ini, fin)
        angles[i] = ang
    return dm_abc, angles

def save_dipoles(x, filename, frame, diptype, dip, oscil=None, n=None):
    tot = np.linalg.norm(dip)
    xyz = f' |{diptype}| = {tot:4.2f}    {frame} (D): {dip[0]:7.3f} {dip[1]:7.3f} {dip[2]:7.3f}'
    with open(filename, 'a') as f:
        if diptype == 'PDM':
            print(f'{x.iname:<30} {diptype} <{x.istate}|mu|{x.istate}>' + xyz, file=f)
        elif diptype == 'TDM':
            if x.istate == 0:
                print(f'{x.iname:<30} {diptype} <{x.istate}|mu|0> Oscil = {oscil:4.3f}' + xyz, file=f)
            else:
                print(f'{x.iname:<30} {diptype} <0|mu|{n}> Oscil = {oscil:4.3f}' + xyz, file=f)
        else:
            raise NotImplementedError

def cms_dip(x):
    out = x.iname+'_'+str(x.nel)+'e'+str(x.norb)+'o'+'_'+str(x.istate)
    mol = gto.M(atom=open('geom_'+x.iname+'.xyz').read(), charge=x.icharge, spin=x.ispin,
                    symmetry=x.isym, output=out+'.log', verbose=4, basis=x.ibasis)
    weights=[1/x.iroots]*x.iroots 

    # -------------------- HF ---------------------------
    mf = scf.RHF(mol).run()
    molden.from_mo(mol, out+'_hf.molden', mf.mo_coeff)

    # -------------------- MC-PDFT ---------------------------
    mc = mcpdft.CASSCF(mf, x.ifunc, x.norb, x.nel, grids_level=x.grid)
    mc.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
    mc.fcisolver.wfnsym = x.irep
    mo = mcscf.sort_mo(mc, mf.mo_coeff, x.cas_list)
    mc.kernel(mo)
    mo = mc.mo_coeff 
    molden.from_mo(mol, out+'_ini.molden', mc.mo_coeff)

    # ----------------- Geometry Optimization ----------------------
    if x.opt:
        mol_eq = mc.nuc_grad_method().as_scanner().optimizer().kernel()
    else: #Preoptimized geometry
        mol_eq = gto.M(atom=open('geom_opt_'+out+'.xyz').read(), charge=x.icharge, spin=x.ispin,
            symmetry=x.isym, output=out+'.log', verbose=4, basis=x.ibasis)

    # -------------------- CMS-PDFT ---------------------------
    mf_eq = scf.RHF(mol_eq).run()
    mc_eq = mcpdft.CASSCF(mf_eq, x.ifunc, x.norb, x.nel, grids_level=x.grid)
    mc_eq.fcisolver = csf_solver(mol_eq, smult=x.ispin+1, symm=x.isym)
    mc_eq.fcisolver.wfnsym = x.irep
    mc_eq = mc_eq.multi_state(weights,'cms')
    mc_eq.max_cyc = 500
    mo = mcscf.project_init_guess(mc_eq, mo)
    mc_eq.kernel(mo)
    mo = mc_eq.mo_coeff 
    molden.from_mo(mol_eq, out+'_opt.molden', mc_eq.mo_coeff)


    # ------------------- CMS-PDFT PDM AND TDM ------------------
    en = mc_eq.e_states.tolist() #List of CMS energies
    pdm = mc_eq.dip_moment(state=x.istate, unit='Debye')
    pdm = [pdm]
    save_dipoles(x, 'cms_pdm', 'XYZ', 'PDM', pdm[0])
    if x.istate==0:
        tdm = [mc_eq.trans_moment(state=[0,i]) for i in range(1,x.iroots)]
        for n in range(1,x.iroots):
            k = n-1 # TDM's id
            tot = np.linalg.norm(tdm[k])/nist.AU2DEBYE
            oscil = (2/3)*(en[n]-en[0])*(tot**2)
            save_dipoles(x, 'cms_tdm', 'XYZ', 'TDM', tdm[k], oscil=oscil, n=n)
    else:
        tdm = mc_eq.trans_moment(state=[0,x.istate])
        tot = np.linalg.norm(tdm)/nist.AU2DEBYE
        oscil = (2/3)*(en[x.istate]-en[0])*(tot**2)
        tdm = [tdm]
        save_dipoles(x, 'cms_tdm', 'XYZ', 'TDM', tdm[0], oscil=oscil, n=0)
        

    #Save TDMs in ABC frame
    pdm_abc, tdm_abc = transform_dip(x, mol, pdm, tdm, 'cms')
    save_dipoles(x, 'cms_pdm_ABC', 'ABC', 'PDM', pdm_abc[0])
    if x.istate==0: # from <0| to others
        for n in range(1,x.iroots):
            k = n-1 # TDM's id 
            tot = np.linalg.norm(tdm_abc[k])/nist.AU2DEBYE
            oscil = (2/3)*(en[n]-en[0])*(tot**2)
            save_dipoles(x, 'cms_tdm_ABC', 'ABC', 'TDM', tdm_abc[k], oscil=oscil, n=n)
    else:
        tot = np.linalg.norm(tdm_abc[0])/nist.AU2DEBYE
        oscil = (2/3)*(en[x.istate]-en[0])*(tot**2)
        save_dipoles(x, 'cms_tdm_ABC', 'ABC', 'TDM', tdm_abc[0], oscil=oscil, n=0)

    #Save final energies
    with open('cms_energies', 'a') as f:
        print(f'{out:<30} {en}', file=f)
    #-------------------------------------------------------------------
         
    # ------------------- CAS-CI PDM AND TDM ------------------
    pdm = dm_casci(mo,mc_eq,mol_eq,state=[x.istate,x.istate])
    pdm = [pdm]
    save_dipoles(x, 'casci_pdm', 'XYZ', 'PDM', pdm[0])
    #Compute & save TDMs from the optimized excited to ground state
    en = mc_eq.e_mcscf
    if x.istate==0: # from <0| to others
        tdm = [dm_casci(mo,mc_eq,mol_eq,state=[0,n]) for n in range(1,x.iroots)]
        for n in range(1,x.iroots):
            k = n-1 # TDM's id 
            tot = np.linalg.norm(tdm[k])/nist.AU2DEBYE
            oscil = (2/3)*(en[n]-en[0])*(tot**2)
            save_dipoles(x, 'casci_tdm', 'XYZ', 'TDM', tdm[k], oscil=oscil, n=n)
    else:    
        tdm = dm_casci(mo,mc_eq,mol_eq,state=[x.istate,0])
        tot = np.linalg.norm(tdm)/nist.AU2DEBYE
        oscil = (2/3)*(en[x.istate]-en[0])*(tot**2)
        tdm = [tdm]
        save_dipoles(x, 'casci_tdm', 'XYZ', 'TDM', tdm[0], oscil=oscil, n=0)

    #Save TDMs in ABC frame
    pdm_abc, tdm_abc = transform_dip(x, mol, pdm, tdm, 'casci')
    save_dipoles(x, 'casci_pdm_ABC', 'ABC', 'PDM', pdm_abc[0])
    if x.istate==0: # from <0| to others
        for n in range(1,x.iroots):
            k = n-1 # TDM's id 
            tot = np.linalg.norm(tdm_abc[k])/nist.AU2DEBYE
            oscil = (2/3)*(en[n]-en[0])*(tot**2)
            save_dipoles(x, 'casci_tdm_ABC', 'ABC', 'TDM', tdm_abc[k], oscil=oscil, n=n)
    else: 
        tot = np.linalg.norm(tdm_abc[0])/nist.AU2DEBYE
        oscil = (2/3)*(en[x.istate]-en[0])*(tot**2)
        save_dipoles(x, 'casci_tdm_ABC', 'ABC', 'TDM', tdm_abc[0], oscil=oscil, n=0)

  
    #Save the optimized geometry to the xyz file
    if x.opt:
        with open('geom_opt_'+out+'.xyz', 'w') as f:
            for i in range(mol_eq.natm):
                coord = str(mol_eq.atom_coord(i,unit='Angstrom'))[1:-1]
                print(f'{mol_eq.atom_symbol(i)} {coord}', file=f)

    #Save final energies
    with open('casci_energies', 'a') as f:
        print(f'{out:<30} {en}', file=f)
            
    return

from dataclasses import dataclass
@dataclass
class Molecule:
    iname   : str
    nel     : int
    norb    : int
    cas_list: list
    iroots  : int = 2
    istate  : int = 0
    icharge : int = 0
    isym    : str = "C1"
    irep    : str = "A" 
    ispin   : int = 0
    ibasis  : str = "julccpvdz"
    grid    : int = 3
    ifunc   : str = 'tPBE'
    opt     : bool = True

    def __post_init__(self):
        for func in self.ifunc:
            if len(func) > 10:
                raise NotImplementedError('Hybrid functionals were not tested')
            elif func[0] =='f':
                raise NotImplementedError('Fully-translated functionals were not tested')

x=[None]*24
x[0]  = Molecule('phenol'              ,  8,7,  [19,23,24,25,31,33,34])
x[1]  = Molecule('benzonitrile'        , 10,10, [21,24,25,26,27,29,32,43,45,46])
x[2]  = Molecule('dimethoxybenzene'    , 10,8,  [24,25,35,36,37,47,50,56])
x[3]  = Molecule('fluorobenzene'       , 6,6,   [23,24,25,31,32,34])
x[4]  = Molecule('indole'              , 10,9,  [23,28,29,30,31,37,40,42,45])
x[5]  = Molecule('x7_azaindole'        , 10,9,  [22,27,29,30,31,35,38,43,44])
x[6]  = Molecule('x4_fluoroindole'     , 10,9,  [27,32,33,34,35,41,43,46,47])
x[7]  = Molecule('x5_fluoroindole'     , 10,9,  [27,32,33,34,35,40,44,45,51])
x[8]  = Molecule('x6_fluoroindole'     , 10,9,  [28,32,33,34,35,40,44,45,48])
x[9]  = Molecule('anti_5_hydroxyindole', 12,10, [35,34,33,32,27,25,41,44,46,47])
x[10] = Molecule('anti_4_methoxyindole', 12,10, [25,28,36,37,38,39,45,46,50,51])
x[11] = Molecule('anti_5_methoxyindole', 12,10, [28,33,36,37,38,39,45,46,50,52])
x[12] = Molecule('syn_6_methoxyindole' , 12,10, [29,33,36,37,38,39,45,47,50,52])
x[13] = Molecule('x6_methylindole'     , 10,9,  [25,32,33,34,35,41,43,45,47])
x[14] = Molecule('x5_cyanoindole'      , 14,13, [26,31,33,34,35,36,37, 41,44,45,46,49,56])
x[15] = Molecule('x4_cyanoindole'      , 14,13, [26,31,33,34,35,36,37, 40,44,47,49,51,53])
x[16] = Molecule('x3_cyanoindole'      , 14,13, [25,32,33,34,35,36,37, 41,45,47,49,50,52])
x[17] = Molecule('x2_cyanoindole'      , 14,13, [25,32,33,34,35,36,37, 41,45,47,49,50,52])
x[18] = Molecule('cis_2_naphthol'      , 12,11, [27,32,35,36,37,38,42,46,48,50,53])
x[19] = Molecule('trans_2_naphthol'    , 12,11, [28,32,35,36,37,38,42,45,48,50,53])
x[20] = Molecule('propynal'            ,  8,7,  [11,12,13,14,16,21,22])
x[21] = Molecule('formaldehyde'        ,  6, 6, [6,7,8,9,10,11])
x[22] = Molecule('x1_fluoronaphthalene', 10,10, [32,35,36,37,38,42,45,47,50,53])
x[23] = Molecule('x2_fluoronaphthalene', 10,10, [31,35,36,37,38,42,45,48,50,52])


# x[12].istate = 0
# cms_dip(x[12])

# x[10].istate = 0
# x[10].opt = False
cms_dip(x[10])

# cms_dip(x[18])

# x[3].istate = 0
# x[3].opt = False
# cms_dip(x[3])