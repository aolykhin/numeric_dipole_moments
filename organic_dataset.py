from pyscf import gto, scf, mcscf
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from mrh.my_pyscf.prop.dip_moment.mcpdft import nuclear_dipole
import numpy as np
from pyscf.tools import molden
from pyscf.gto import inertia_moment
from pyscf.data import nist
from numpy.linalg import norm as norm
import math

def dm_casci(mc,mol,mo,ci,state):
    '''Return diagonal (PDM) and off-diagonal (TDM) dipole moments'''
    from functools import reduce
    ncore = mc.ncore
    ncas = mc.ncas
    nocc = ncore + ncas
    mo_core = mo[:,:ncore]
    mo_cas = mo[:,ncore:nocc]
    nelecas = mc.nelecas

    orbcas = mo[:, ncore:ncore+ncas]
    mass = mol.atom_mass_list()
    coords = mol.atom_coords()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    with mol.with_common_orig(mass_center):
        dip_ints = mol.intor('cint1e_r_sph', comp=3)

    # t_dm1 = mc.fcisolver.trans_rdm1(ci[state[1]], ci[state[0]], ncas, nelecas)
    # t_dm1_ao = reduce(np.dot, (orbcas, t_dm1, orbcas.T))
    # dip = -np.einsum('xij,ji->x', dip_ints, t_dm1_ao)
    # if state[1] == state[0]: 
    #     dip += nuclear_dipole(mc,origin='mass_center')
    
    
    if state[1] != state[0]: 
        t_dm1 = mc.fcisolver.trans_rdm1(ci[state[1]], ci[state[0]], ncas, nelecas)
        t_dm1_ao = reduce(np.dot, (orbcas, t_dm1, orbcas.T))
        dip = np.einsum('xij,ji->x', dip_ints, t_dm1_ao)
    else:
        casdm1 = mc.fcisolver.make_rdm1([ci[state[1]]], ncas, nelecas)
        dm_core = np.dot(mo_core, mo_core.T) * 2
        dm_cas = reduce(np.dot, (mo_cas, casdm1, mo_cas.T))
        dm = dm_core + dm_cas
        dip = -np.einsum('xij,ij->x', dip_ints, dm)
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

def findClockwiseAngle(ini, fin):
    '''Takes '''
    ind=2 # c-component in abc frame
    print(ini,fin)
    if len(ini)==3 and abs(ini[ind])<2e-2: ini=np.delete(ini,ind) 
    if len(fin)==3 and abs(fin[ind])<2e-2: fin=np.delete(fin,ind)
    assert len(ini) == len(fin), 'The c-component of dipole is too large, molecule is not planar'
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
    origin  = np.eye(3)
    rot_mat = np.linalg.solve(origin,abc_vec)
    pdm_abc, pdm_ang = get_ang_with_a_axis(pdm, rot_mat)
    tdm_abc, tdm_ang = get_ang_with_a_axis(tdm, rot_mat)
    return pdm_abc, tdm_abc, pdm_ang, tdm_ang

def get_ang_with_a_axis(dm, rot_mat):
    dm_abc = np.empty_like(dm)
    angles = [None]*len(dm)

    for i in range(len(dm)):
        dm_abc[i]=np.dot(dm[i],rot_mat)
        ini = dm_abc[i]
        fin = np.array([1,0,0])
        dot = np.dot(ini,fin)/(norm(ini)*norm(fin))
        if abs(dot) < 1e-3:
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

def save_dipoles(x, method, frame, dip, ini=None, fin=None, oscil=None):
    tot = np.linalg.norm(dip)
    diptype = 'PDM' if ini == fin else 'TDM'
    id =f'< {ini}|mu|{fin}> = {tot:4.2f}'
    xyz = f'     {frame} (D): {dip[0]:7.3f} {dip[1]:7.3f} {dip[2]:7.3f}'
    output = method+'_'+diptype+'_'+str(ini)+str(fin)+'_'+frame
    with open(output, 'a') as f:
        if oscil: 
            print(f'{x.iname:<30}' + id + xyz + f' {oscil=:4.3f}', file=f)
        else:
            print(f'{x.iname:<30}' + id + xyz, file=f)

def save_angles(x, method, ang, ini=None, fin=None):
    diptype = 'PDM' if ini == fin else 'TDM'
    id =f'< {ini}|mu|{fin}>'
    output = method+'_'+'angles'+'_'+diptype+'_'+str(ini)+str(fin)
    with open(output, 'a') as f:
        print(f'{x.iname:<30} {diptype} theta for {id} is {ang:5.1f} degrees', file=f)

def main(x):
    out = x.iname+'_'+str(x.nel)+'e'+str(x.norb)+'o'+'_'+str(x.istate)
    mol = gto.M(atom=open('geom_'+x.iname+'.xyz').read(), charge=x.icharge, spin=x.ispin,
                    output=out+'.log', verbose=4, basis=x.ibasis)
    
    # -------------------- Single Point at Initial Geometry -----------------------
    weights=[1/x.iroots]*x.iroots 
    mf = scf.RHF(mol).run()
    molden.from_mo(mol, out+'_hf.molden', mf.mo_coeff)
    mc = mcpdft.CASSCF(mf, x.ifunc, x.norb, x.nel, grids_level=x.grid)
    mc.fcisolver = csf_solver(mol, smult=x.ispin+1)
    mo = mcscf.sort_mo(mc, mf.mo_coeff, x.cas_list)
    if x.opt_method == 'MC-PDFT': None
    elif x.opt_method == 'CMS-PDFT': mc = mc.multi_state(weights,'cms')
    elif x.opt_method == 'SA-PDFT': mc.state_average_(weights)
    else: raise NotImplemented('Geometry optimization method is not recognized')
    mc.max_cycle_macro = 600
    mc.max_cycle = 600
    mc.kernel(mo)
    mo = mc.mo_coeff 
    molden.from_mo(mol, out+'_ini.molden', mc.mo_coeff)

    # ----------------- Geometry Optimization ----------------------
    if x.opt:
        opt_id = 0 if x.opt_method == 'MC-PDFT' else x.istate
        mol_eq = mc.nuc_grad_method().as_scanner(state=opt_id).optimizer().kernel()
        with open('geom_opt_'+out+'.xyz', 'w') as f:
            for i in range(mol_eq.natm):
                coord = str(mol_eq.atom_coord(i,unit='Angstrom'))[1:-1]
                print(f'{mol_eq.atom_symbol(i)} {coord}', file=f)
    else: #Read preoptimized geometry
        mol_eq = gto.M(atom=open('geom_opt_'+out+'.xyz').read(), charge=x.icharge, spin=x.ispin,
            symmetry=x.isym, output=out+'.log', verbose=4, basis=x.ibasis)

    # -------------------- Single Point at Final Geometry -----------------------
    mf_eq = scf.RHF(mol_eq).run()
    mc_eq = mcpdft.CASSCF(mf_eq, x.ifunc, x.norb, x.nel, grids_level=x.grid)
    mc_eq.fcisolver = csf_solver(mol_eq, smult=x.ispin+1)
    if x.opt_method == 'MC-PDFT': None
    elif x.opt_method =='CMS-PDFT': mc_eq = mc_eq.multi_state(weights,'cms')
    elif x.opt_method == 'SA-PDFT': mc_eq.state_average_(weights)
    else: raise NotImplemented('Geometry optimization method is not recognized')
    mc_eq.max_cyc = 500
    mo = mcscf.project_init_guess(mc_eq, mo)
    mc_eq.kernel(mo)
    mo = mc_eq.mo_coeff
    molden.from_mo(mol_eq, out+'_opt.molden', mc_eq.mo_coeff)
    
    if x.opt_method == 'SA-PDFT': ci = mc_eq.ci 
    elif x.opt_method == 'CMS-PDFT': ci = mc_eq.get_ci_adiabats(uci='MCSCF')
    elif x.opt_method == 'MC-PDFT':
        if x.dip_method == 'CMS-PDFT':
            mf_eq = scf.RHF(mol_eq).run()
            mc_eq = mcpdft.CASSCF(mf_eq, x.ifunc, x.norb, x.nel, grids_level=x.grid)
            mc_eq.fcisolver = csf_solver(mol_eq, smult=x.ispin+1)
            mc_eq.kernel(mo)
            mo = mc_eq.mo_coeff
            ci = mc_eq.get_ci_adiabats(uci='MCSCF')
        if x.dip_method == 'CAS-CI':
            mc_eq = mcscf.CASSCF(mf_eq, x.norb, x.nel)
            mc_eq.state_average(weights)
            mc_eq.kernel(mo)
            mo = mc_eq.mo_coeff
            ci = mc_eq.ci
    get_dipoles(x, mc_eq, mol_eq, mo, ci, out)
    return

def get_dipoles(x, mc, mol, mo, ci, out):
    # fake casscf is required to avoid inheretence issue in mc object
    from pyscf.grad.sacasscf import Gradients
    fcasscf = Gradients(mc).make_fcasscf (x.istate)
    fcasscf.mo_coeff = mo
    fcasscf.ci = ci[x.istate]
    if ((x.opt_method == 'CMS-PDFT') and (x.dip_method == 'CMS-PDFT')):
        method = 'cms'
        en = mc.e_states.tolist()
        pdm = [mc.dip_moment(state=x.istate, unit='Debye')]
    elif x.dip_method == 'SA-PDFT':
        method = 'sapdft'
        en = mc.e_states
        pdm = [mc.dip_moment(state=x.istate, unit='Debye')]
    elif x.dip_method == 'CAS-CI':
        method = 'cas'
        en = mc.e_mcscf
        pdm = [dm_casci(fcasscf, mol, mo, ci, state=[x.istate,x.istate])]

    if x.dip_method == 'CMS-PDFT':
        if x.istate==0: tdm = [mc.trans_moment(state=[0,n]) for n in range(1,x.iroots)]
        else:           tdm = [mc.trans_moment(state=[x.istate,0])]
    elif ((x.dip_method == 'CAS-CI') or (x.dip_method == 'SA-PDFT')):
        if x.istate==0: tdm = [dm_casci(fcasscf, mol, mo, ci, state=[0,n]) for n in range(1,x.iroots)]
        else:           tdm = [dm_casci(fcasscf, mol, mo, ci, state=[x.istate,0])]
        
    pdm_abc, tdm_abc, pdm_ang, tdm_ang = transform_dip(x, mol, pdm, tdm, method)
    
    save_dipoles(x, method, 'XYZ', pdm[0], ini=x.istate, fin=x.istate)
    save_dipoles(x, method, 'ABC', pdm_abc[0], ini=x.istate, fin=x.istate)
    save_angles(x, method, pdm_ang[0], ini=x.istate, fin=x.istate)
    if x.istate==0: # from <0| to others
        for n in range(1,x.iroots):
            k = n-1 # TDM's id 
            tot = np.linalg.norm(tdm_abc[k])/nist.AU2DEBYE
            oscil = (2/3)*(en[n]-en[0])*(tot**2)
            save_dipoles(x, method, 'XYZ', tdm[k], ini=0, fin=n, oscil=oscil)
            save_dipoles(x, method, 'ABC', tdm_abc[k], ini=0, fin=n, oscil=oscil)
            save_angles(x, method, tdm_ang[k], ini=0, fin=n)
    else:
        tot = np.linalg.norm(tdm_abc[0])/nist.AU2DEBYE
        oscil = (2/3)*(en[x.istate]-en[0])*(tot**2)
        save_dipoles(x, method, 'XYZ', tdm[0], ini=x.istate, fin=0, oscil=oscil)
        save_dipoles(x, method, 'ABC', tdm_abc[0], ini=x.istate, fin=0, oscil=oscil)
        save_angles(x, method, tdm_ang[0], ini=x.istate, fin=0,)

    #Save final energies
    output = method+'_'+'energies'+'_'+str(x.istate)
    with open(output, 'a') as f:
        if len(en) > 2: raise RuntimeError('number of roots is limited to 2')
        print(f'{x.iname:<30} {en[0]:6.12f}  {en[1]:6.12f}', file=f)
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
    opt     : bool= False
    opt_method  : str = 'CMS-PDFT'
    dip_method  : str = 'CMS-PDFT'

    def __post_init__(self):
        for func in self.ifunc:
            if len(func) > 10:
                raise NotImplementedError('Hybrid functionals were not tested')
            elif func[0] =='f':
                raise NotImplementedError('Fully-translated functionals were not tested')
        if self.opt_method == 'MC-PDFT' and self.istate !=0:
            raise ValueError('MC-PDFT support only the ground state but Fully-translated functionals were not tested')

x=[None]*100
x[0]  = Molecule('indole'              , 10,9,  [23,28,29,30,31,37,40,42,45])
x[1]  = Molecule('x2_cyanoindole'      , 14,13, [25,31,33,34,35,36,37, 41,44,48,49,53,57])
x[2]  = Molecule('x3_cyanoindole'      , 14,13, [25,32,33,34,35,36,37, 41,45,47,49,50,57])
x[3]  = Molecule('x4_cyanoindole'      , 14,13, [27,32,33,34,35,36,37, 39,44,47,49,50,59])
x[4]  = Molecule('x5_cyanoindole'      , 14,13, [26,31,33,34,35,36,37, 41,44,46,49,50,58])
x[5]  = Molecule('x4_fluoroindole'     , 10,9,  [27,32,33,34,35,41,43,46,47])
x[6]  = Molecule('x5_fluoroindole'     , 10,9,  [27,32,33,34,35,40,44,45,51])
x[7]  = Molecule('x6_fluoroindole'     , 10,9,  [28,32,33,34,35,40,44,45,48])
x[8]  = Molecule('x6_methylindole'     , 10,9,  [25,32,33,34,35,41,43,45,47])
x[9]  = Molecule('anti_5_hydroxyindole', 12,10, [35,34,33,32,27,25,41,44,46,47])
x[10] = Molecule('anti_5_methoxyindole', 12,10, [28,33,36,37,38,39,45,46,50,52])
x[11] = Molecule('syn_6_methoxyindole' , 12,10, [29,33,36,37,38,39,45,47,50,52])
x[12] = Molecule('x7_azaindole'        , 10,9,  [22,27,29,30,31,35,38,43,44])
x[13] = Molecule('cis_2_naphthol'      , 12,11, [27,32,35,36,37,38,42,46,48,50,53])
x[14] = Molecule('trans_2_naphthol'    , 12,11, [28,32,35,36,37,38,42,45,48,50,53])
x[15] = Molecule('benzonitrile'        , 10,10, [21,24,25,26,27,29,32,43,45,46])
x[16] = Molecule('phenol'              ,  8,7,  [19,23,24,25,31,33,34])
x[17] = Molecule('anisole'             ,  8,7,  [23,27,28,29,36,39,40])
x[18] = Molecule('x13_dimethoxybenzene', 10,8,  [24,25,35,36,37,47,50,56])
x[19] = Molecule('x14_dimethoxybenzene', 10,8,  [30,32,35,36,37,47,50,54])
#------------------------------- Extra -------------------------------------------
# x[3]  = Molecule('x4_cyanoindole'      , 14,13, [26,31,33,34,35,36,37, 40,44,47,49,51,59])
# x[3]  = Molecule('x4_cyanoindole'      , 14,13, [26,30,33,34,35,36,37, 42,44,48,49,53,61])
# x[20] = Molecule('fluorobenzene'       , 6,6,   [23,24,25,31,32,34])
# x[21] = Molecule('anti_4_methoxyindole', 12,10, [25,28,36,37,38,39,45,46,50,51])
# x[22] = Molecule('propynal'            ,  8,7,  [11,12,13,14,16,21,22])
# x[23] = Molecule('formaldehyde'        ,  6, 6, [6,7,8,9,10,11])
# x[24] = Molecule('x1_fluoronaphthalene', 10,10, [32,35,36,37,38,42,45,47,50,53])
# x[25] = Molecule('x2_fluoronaphthalene', 10,10, [31,35,36,37,38,42,45,48,50,52])


x[16].istate = 0
x[16].opt = False
x[16].opt_method = 'SA-PDFT'
x[16].dip_method = 'SA-PDFT'
main(x[16])

# x[15].istate = 1
# x[15].opt_method = 'CMS-PDFT'
# x[15].dip_method = 'CMS-PDFT'
# main(x[15])
