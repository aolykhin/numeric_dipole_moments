from logging import raiseExceptions
from black import is_type_comment
from tabulate import tabulate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from pyscf import gto, scf, mcscf, lib, fci
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
import os
from pyscf.tools import molden
import copy
from colored import fg, attr
from pyscf.data import nist
from pyscf.fci.addons import overlap as braket
import itertools
from numpy import unravel_index
from icecream import ic

# os.environ['OMP_NUM_THREADS'] = "4"
# os.environ['MKL_NUM_THREADS'] = "4"
# os.environ['OPENBLAS_NUM_THREADS'] = "4"

# List of molecules and molecular parameters
geom_phenol_opt= '''
C       -1.182853111      1.200708431      0.00000000
C        0.212397551      1.216431879      0.00000000
C       -1.877648981     -0.011392736      0.00000000
H        0.770332141      2.159905341      0.00000000
H       -2.974311304     -0.021467798      0.00000000
C        0.919152698      0.005414786      0.00000000
C       -1.164162291     -1.211169261      0.00000000
H       -1.697713380     -2.169828607      0.00000000
C        0.232754869     -1.212093364      0.00000000
H        0.793712181     -2.157907002      0.00000000
H       -1.735970731      2.149003232      0.00000000
O        2.284795534      0.064463230      0.00000000
H        2.640406066     -0.838707361      0.00000000
'''
geom_OH_phenol= '''
C       -2.586199811     -1.068328539     -2.343227944
C       -1.383719126     -0.571709401     -1.839139664
C       -3.598933171     -1.486733966     -1.476306928
H       -0.580901222     -0.240090937     -2.507686505
H       -4.543398025     -1.876891664     -1.874465443
C       -1.192810178     -0.493086491     -0.452249284
C       -3.398663509     -1.403899151     -0.097309120
H       -4.185490834     -1.729852747      0.594389093
C       -2.200418093     -0.909124041      0.423137737
H       -2.045275617     -0.844707274      1.509797436
H       -2.734632202     -1.130309287     -3.429202743
O        0.000000000      0.000000000      0.000000000
H        0.000000000      0.000000000      {}
'''
geom_phenol_12e11o= '''
C       -2.622730874     -1.013542229     -2.331647466
C       -1.406745441     -0.543467761     -1.836239389
C       -3.642230632     -1.407856573     -1.459091122
H       -0.600882859     -0.232144838     -2.511037202
H       -4.597665462     -1.776495891     -1.851638458
C       -1.204492019     -0.465471372     -0.448760082
C       -3.431148118     -1.326314087     -0.081269194
H       -4.220682665     -1.631881105      0.616909336
C       -2.218870662     -0.857801984      0.431663464
H       -2.057527810     -0.795466793      1.517843236
H       -2.777706083     -1.073150292     -3.417047881
O        0.000000000      0.000000000      0.000000000
H        0.000000000      0.000000000      {}
'''
geom_phenol= '''
C        1.214793846     -1.192084882      0.000000000
C       -0.181182920     -1.223420370      0.000000000
C        1.885645321      0.032533975      0.000000000
H       -0.703107911     -2.176177034      0.000000000
H        2.971719251      0.056913772      0.000000000
C       -0.906309250     -0.030135294      0.000000000
C        1.160518991      1.225819051      0.000000000
H        1.682443982      2.178575715      0.000000000
C       -0.235457775      1.194483563      0.000000000
H       -0.799607233      2.122861284      0.000000000
H        1.778943305     -2.120462603      0.000000000
O       -2.345946581     -0.062451754      0.000000000
H       -2.680887890      0.843761215      0.000000000
'''
geom_h2co= '''
H       -0.000000000      0.950627350     -0.591483790
H       -0.000000000     -0.950627350     -0.591483790
C        0.000000000      0.000000000      0.000000000
O        0.000000000      0.000000000     {}
'''
geom_furan= '''
C        0.000000000     -0.965551055     -2.020010585
C        0.000000000     -1.993824223     -1.018526668
C        0.000000000     -1.352073201      0.181141565
O        0.000000000      0.000000000      0.000000000
C        0.000000000      0.216762264     -1.346821565
H        0.000000000     -1.094564216     -3.092622941
H        0.000000000     -3.062658055     -1.175803180
H        0.000000000     -1.688293885      1.206105691
H        0.000000000      1.250242874     -1.655874372
'''
geom_butadiene= '''
C       -1.723496679     -0.622891260      0.000000000
H       -2.263989282     -1.564248373      0.000000000
H       -2.311546164      0.289288564      0.000000000
C       -0.385732145     -0.608988883      0.000000000
C        0.385732145      0.608988883      0.000000000
H        0.153350318     -1.554122238      0.000000000
H       -0.153350318      1.554122238      0.000000000
C        1.723496679      0.622891260      0.000000000
H        2.311546164     -0.289288564      0.000000000
H        2.263989282      1.564248373      0.000000000
'''
geom_h2o='''
O  0.00000000   0.08111156   0.00000000
H  0.78620605   0.66349738   0.00000000
H -0.78620605   0.66349738   0.00000000
'''

def cs(text): return fg('light_green')+str(text)+attr('reset')

def pdtabulate(df, line1, line2): return tabulate(df, headers=line1, tablefmt='psql', floatfmt=line2)

def get_sign_list(x):
    sign_list=[]
    combos=[]
    tmp=[1]*x.iroots
    # for i in range(x.iroots): #all except all states flipped
    for i in range(x.iroots+1):
        if i!=0: tmp[i-1]=-1
        combos=itertools.permutations(tmp,x.iroots)
        for i in combos:
            if i not in sign_list:
                sign_list.append(i)
    return sign_list

def fix_sa_reference_order(x, mol, ci_buf, ci_ref, si_sa_buf, si_sa_zero):
    regular=list(range(x.iroots))
    over=np.zeros((x.iroots,x.iroots),dtype=int)
    ci_order=[]
    ci_signs=[]
    for i in range(x.iroots):
        for j in range(x.iroots):
            val=braket(ci_ref[i],ci_buf[j],norb=x.norb,nelec=x.nel)   
            print('VAL before rounding\n',i,j,val)
            over[i,j]=int(round(val))
            if over[i,j] !=  0: ci_order.append(j)
            if over[i,j] == -1: ci_signs.append(-1)
            if over[i,j] ==  1: ci_signs.append(1)
    if (len(ci_order) > x.iroots) or (len(ci_signs) > x.iroots):
        raise ValueError('CI vectors are not orthogonal enough')
    
    # # For some reason the above code doesn't capture change of a sign
    # # in reference states (columns) -> An extra step here resolves the problem
    # # Reflect reference states to better fit si_in_zero 
    # sign_list=get_sign_list(x)
    # overlap=np.zeros(len(sign_list))
    # for i, signs in enumerate(sign_list):
    #     for j, coeff in enumerate(signs): 
    #         si_sa_buf[:,j]=coeff*si_sa_buf[:,j]
    #         diff=abs(si_sa_buf-si_sa_zero)
    #         overlap[i]=diff.sum()
    # ind=np.argmin(overlap)
    # ic(ind)
    # ci_signs=sign_list[ind]
    # ic(sign_list)
    # ic(overlap)
    # ic(ci_signs)

    # #Print overlap of adjusted ci_buf with ci_ref
    # if mol.verbose >= lib.logger.DEBUG:
    #     print('ci_order\n',ci_order)
    #     print('ci_signs\n',ci_signs)
    #     ci_buf=np.asanyarray(ci_buf)
    #     ci_buf[regular]=ci_buf[ci_order]
    #     for i, coeff in enumerate(ci_signs): ci_buf[i]=coeff*ci_buf[i]
    #     ci_buf=list(ci_buf)
    #     for i in range(x.iroots):
    #         for j in range(x.iroots):
    #             val=braket(ci_ref[i],ci_buf[j],norb=x.norb,nelec=x.nel)   
    #             over[i,j]=int(round(val))
    #     print('Overlap of CI vectors after adjustment\n',over)

    #Adjust order & signs of SA states in si_sa_buf 
    # i <-- j, the ci_signs array is given for the rotated states 
    # print('BEFORE CI ADJUSTMENT\n',si_sa_buf)
    print('ci_order\n',ci_order)
    print('ci_signs\n',ci_signs)
    si_sa_buf[:,regular]=si_sa_buf[:,ci_order]
    for i, coeff in enumerate(ci_signs): si_sa_buf[:,i]=coeff*si_sa_buf[:,i]
    # print('AFTER CI ADJUSTMENT\n',si_sa_buf)

    return si_sa_buf

def fix_order_of_states(x, mol, ci_buf, ci_zero, si_sa_ref, si_in_ref, ham_ref, si_sa_buf, si_in_buf, ham_buf):
    #Rotate sa reference states in expansion over sa reference states
    si_sa_buf=fix_sa_reference_order(x, mol, ci_buf, ci_zero, si_sa_buf, si_sa_ref)

    # Get all variations of signs & oredr of intermediate states (rows) 
    order=list(itertools.permutations(range(x.iroots)))
    sign_list=get_sign_list(x)
    regular=list(range(x.iroots))

    n_order=len(order)
    n_sign=len(sign_list)

    # Rotate ROWS of a field-dependent matrix
    sign=np.zeros((x.iroots,x.iroots))
    wrk=np.zeros((x.iroots,x.iroots))
    overlap=np.zeros((n_order,n_sign))
    overlap_si=np.zeros((n_order,n_sign))

    print('Original Ham\n=',ham_buf)
    print('Original si_sa\n',si_sa_buf)
    
    old_diff=100 
    for k, new_order in enumerate(order):# over all orders of intermediate states
        for m, new_signs in enumerate(sign_list): 
            #Selected order of intermediate states with all variations of signs
            for i in regular:
                for j in regular:
                    sign[i,j]=new_signs[i]*new_signs[j]
                    wrk[i,j] = sign[i,j]*ham_buf[new_order[i],new_order[j]]

            overlap[k,m]=abs(wrk-ham_ref).sum()

    num_smallest=x.iroots*2
    for i in range(num_smallest):
        ind = np.unravel_index(np.argmin(overlap, axis=None), overlap.shape)
        overlap[ind]=100
        new_order = order[ind[0]]       
        signs_col = sign_list[ind[1]]       
        
        for signs_row in sign_list:
            for signs_col in sign_list:
                wrk_si=si_sa_buf.copy()
                for ik in regular:
                    wrk_si[ik,:]=signs_row[ik]*wrk_si[ik,:]
                    wrk_si[:,ik]=signs_col[ik]*wrk_si[:,ik]
                wrk_si[regular,:]=wrk_si[new_order,:]

                new_diff=abs(wrk_si-si_sa_ref).sum()

                if new_diff<old_diff:
                    old_diff=new_diff
                    fin_order=list(new_order)
                    fin_s_row=list(signs_row)
                    fin_s_col=list(signs_col)


    print('Overlap=',overlap[k,m])

    # ind_ham, ind_sign = np.unravel_index(np.argmin(overlap, axis=None), overlap.shape)
    # fin_order=list(order[ind_ham])
    # fin_signs=list(sign_list[ind_sign])

    print('final new_order=',fin_order)        
    print('final new_s_row=',fin_s_row)       
    print('final new_s_col=',fin_s_col)       
    
    #Rotate intermediate states in MCSCF unitary
    for ii in range(x.iroots):
        si_sa_buf[ii,:]=fin_s_row[ii]*si_sa_buf[ii,:] 
        si_sa_buf[:,ii]=fin_s_col[ii]*si_sa_buf[:,ii] 
    si_sa_buf[regular,:]=si_sa_buf[fin_order,:]
    print('FINAL si_sa_buf\n',si_sa_buf)
    
    #Rotate intermediate states in PDFT unitary
    for ii in range(x.iroots):
        si_in_buf[ii,:]=fin_s_row[ii]*si_in_buf[ii,:] 
    si_in_buf[regular,:]=si_in_buf[fin_order,:]
    
    #Reflect CMS states (columns) if necessary
    for i in range(x.iroots):
        same=abs(si_in_buf[:,i]-si_in_ref[:,i]).sum()
        oppo=abs(-si_in_buf[:,i]-si_in_ref[:,i]).sum()
        print('cms vector=',i, same, oppo)
        if same>oppo:
            si_in_buf[:,i]=-si_in_buf[:,i]
    print('AOL',si_in_buf)

    #Rotate Hamiltonian
    tmp=ham_buf.copy()
    for i in regular:
        for j in regular:
            sign[i,j]=fin_s_row[i]*fin_s_row[j]
            ham_buf[i,j] = sign[i,j]*tmp[fin_order[i],fin_order[j]]

    print(overlap)

    return si_sa_buf, si_in_buf, ham_buf

def fix_signs_of_final_states(x, si_in_ref, si_in_buf):
    #Reflect CMS states (columns) if necessary
    for i in range(x.iroots):
        same=abs(si_in_buf[:,i]-si_in_ref[:,i]).sum()
        oppo=abs(-si_in_buf[:,i]-si_in_ref[:,i]).sum()
        if same>oppo:
            si_in_buf[:,i]=-si_in_buf[:,i]
    return si_in_buf

# ------------------ NUMERICAL DIPOLE MOMENTS ----------------------------
def numer_run(dist, x, mol, mo_zero, ci_zero, ci0_zero, method, field, formula, ifunc, out, dip_cms, \
    si_sa_zero, si_in_zero, ham_zero, ntdm, unit='Debye', origin='Charge_center'):

    # Set reference point to be center of charge
    mol.output='num_'+ out
    mol.build()
    charges = mol.atom_charges()
    coords  = mol.atom_coords()
    mass = mol.atom_mass_list()
    nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    # mol.set_common_orig_(mass_center)
    mol.set_common_orig_(nuc_charge_center)
    if unit.upper() == 'DEBYE':
        fac = nist.AU2DEBYE
    elif unit.upper() == 'AU':
        fac = 1
    else:
        RuntimeError   

    h_field_off = mol.intor('cint1e_kin_sph') + mol.intor('cint1e_nuc_sph')
    # 1st column is the field column
    dip_num = np.zeros((len(field), 1+4*ntdm))
    tot_der = np.zeros((len(field), 1+4*ntdm))
    rem_der = np.zeros((len(field), 1+4*ntdm))
    ham_der = np.zeros((len(field),3,x.iroots,x.iroots)) 
    si_der  = np.zeros((len(field),3,x.iroots,x.iroots)) 
    for i, f in enumerate(field): # Over field strengths
        dip_num[i][0]=f #  first column is the field column 
        if formula == "2-point":
            disp = [f, -f]
        elif formula == "4-point":
            disp = [2*f, f, -f, -2*f]
        si_in = np.zeros((len(disp),3,x.iroots,x.iroots)) #
        si_sa = np.zeros((len(disp),3,x.iroots,x.iroots)) #
        ham = np.zeros((len(disp),3,x.iroots,x.iroots)) 
        ci_sa = [ [0] * 3] * len(disp)
        ci_buf  = [None]*x.iroots
        # e = [ [] for _ in range(2) ]
        if i==0: #set zero-field MOs as initial guess 
            mo_field = []
            ci_field = []
            ci0_field = []
            for _ in range(3):
                mo_field.append([mo_zero]*len(disp))
                ci_field.append([ci_zero]*len(disp))
                ci0_field.append([ci0_zero]*len(disp))

        for j in range(3): # Over x,y,z directions
            for k, v in enumerate(disp): # Over stencil points 
                if j==0:   # X-axis
                    E = [v, 0, 0]
                elif j==1: # Y-axis
                    E = [0, v, 0]
                elif j==2: # Z-axis
                    E = [0, 0, v]
                # HF step
                h_field_on = np.einsum('x,xij->ij', E, mol.intor('cint1e_r_sph', comp=3))
                h = h_field_off + h_field_on
                mf_field = scf.RHF(mol)
                mf_field.get_hcore = lambda *args: h
                mf_field.max_cycle = 1  # next CASSCF starts from CAS orbitals not RHF
                mf_field.kernel()

                
                mc = mcpdft.CASSCF(mf_field, ifunc, x.norb, x.nel, grids_level=x.grid)
                mc.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
                mc.fcisolver.wfnsym = x.irep
                coord = ['x','y','z']
                point = ['-F','F']
                print('j=%s and k=%s' %(coord[j],point[k]))
                mo=mo_zero 
                ci0=ci0_zero 
                # mc.max_cycle_macro = 5
                # if i==0: #First field starts with zero-field MOs
                #     mc.max_cycle_macro = 200
                #     mo=mo_zero 
                #     ci=ci0_zero 
                # else: # Try MOs from the previous filed/point
                #     mc.max_cycle_macro = 5
                #     mo=mo_field[j][k]
                #     ci=ci_field[j][k]

                # if the threshold is too tight the active space is unstable
                if thresh: 
                    mc.conv_tol = conv_tol
                    mc.conv_tol_grad = conv_tol_grad
                weights=[1/x.iroots]*x.iroots #Equal weights only
                if method == 'SS-PDFT':
                    raise NotImplementedError
                elif method == 'SA-PDFT':
                    raise NotImplementedError
                elif method == 'CMS-PDFT':
                    mc.ci=ci_zero
                    mc=mc.multi_state(weights,'cms')
                    mc.kernel(mo,ci0=ci0)
                    if not mc.converged:
                        print('Number of cycles increased')
                        mc.max_cycle_macro = 600
                        mc.kernel(mo_zero, ci0=ci0_zero)
                    mc.analyze()
                    # molden.from_mo(mol, out+coord+point+'_sa.molden', mc.mo_coeff)
                
                    print('NEXT HAMILTONIAN')
                    e_cms   = mc.e_states.tolist() #List of CMS energies
                    # ci_buf = mc.ci
                    ci_buf = mc.get_ci_adiabats(uci='MCSCF')
                    ham_buf = mc.get_heff_pdft()
                    si_in_buf  = mc.si_pdft
                    si_sa_buf  = mc.si_mcscf

                    # # In the presence of field, the sign and order of sa and intermediate states can change
                    # if k!=0: # Adjust states to k==0
                    #     ham_ref   = ham[0,j,:,:].copy()
                    #     si_in_ref = si_in[0,j,:,:].copy()
                    #     si_sa_ref = si_sa[0,j,:,:].copy()
                    #     ci_ref    = ci_sa[0][j]
                    #     si_sa_buf, si_in_buf, ham_buf = fix_order_of_states(x, mol, ci_buf, ci_ref, \
                    #     si_sa_ref, si_in_ref, ham_ref, si_sa_buf, si_in_buf, ham_buf)
                        
                    # The following code aims to preserve the order and sings as in the zero-field case 
                    si_sa_buf, si_in_buf, ham_buf = fix_order_of_states(x, mol, ci_buf, ci_zero, \
                        si_sa_zero, si_in_zero, ham_zero, si_sa_buf, si_in_buf, ham_buf)
                    si_in_buf = fix_signs_of_final_states(x, si_in_zero, si_in_buf)

                    ham  [k,j,:,:] = ham_buf.copy()
                    si_in[k,j,:,:] = si_in_buf.copy()
                    si_sa[k,j,:,:] = si_sa_buf.copy()
                    ci_sa[k][j] = ci_buf
                    # e[k] = mc.e_states.tolist() #List of  energies
                else:
                    raise NotImplementedError
                mo_field[j][k] = mc.mo_coeff #save MOs for the next stencil point k and axis j 

            #Loop over j = {x,y,z} directions
            #Get dH_PQ/dF derivative numerically        
            for p in range(x.iroots):
                for q in range(x.iroots):
                    if formula == "2-point":
                        ham_der[i,j,p,q]= (ham[0,j,p,q]-ham[1,j,p,q])/(2*f)
                        si_der[i,j,p,q] = (si_in[0,j,p,q]-si_in[1,j,p,q])/(2*f)
        #Loop over i = fields
        id_tdm=0 #enumerate TDM
        for m in range(x.iroots): # TDMs between <m| and |n> states
            for n in range(m):
                if m==1 and n==0: print(f'm={m}, n={n}\n')
                shift=id_tdm*4 # shift to the next state by 4m columns (x,y,z,mu)
                for j in range(3): #over j = {x,y,z} directions
                    print(f'j={j}\n')
                    for p in range(x.iroots):
                        for q in range(x.iroots):
                            # dip_num[i,1+j+shift]+=der[i,j,p,q]*si_in_zero[p][m]*si_in_zero[q][n]

                            # rem_der[i,1+j+shift]+=(si_in_zero[p][m]*si_der[i,j,q,n]+si_in_zero[q][n]*si_der[i,j,p,m])*ham_zero[p][q]

                            # tot_der[i,1+j+shift]+=der[i,j,p,q]*si_in_zero[p][m]*si_in_zero[q][n] + \
                            #     (si_in_zero[p][m]*si_der[i,j,q,n]+si_in_zero[q][n]*si_der[i,j,p,m])*ham_zero[p][q]
                            
                            tmp_dip_num = ham_der[i,j,p,q]*si_in_zero[p][m]*si_in_zero[q][n]
                            dip_num[i,1+j+shift]+=tmp_dip_num 

                            tmp_rem_der = (si_in_zero[p][m]*si_der[i,j,q,n]+si_in_zero[q][n]*si_der[i,j,p,m])*ham_zero[p][q]
                            rem_der[i,1+j+shift]+=tmp_rem_der

                            tmp_tot_der = ham_der[i,j,p,q]*si_in_zero[p][m]*si_in_zero[q][n] + \
                                (si_in_zero[p][m]*si_der[i,j,q,n]+si_in_zero[q][n]*si_der[i,j,p,m])*ham_zero[p][q]
                            tot_der[i,1+j+shift]+= tmp_tot_der
                            
                            if m==1 and n==0: print(f'p={p} q={q} dip_num={tmp_dip_num:8.4f}, rem_der={tmp_rem_der:8.4f}, tot_der={tmp_tot_der:8.4f} ')
                    dip_num[i,1+j+shift]=(-1)*fac*dip_num[i,1+j+shift]
                    rem_der[i,1+j+shift]=(-1)*fac*rem_der[i,1+j+shift]
                    tot_der[i,1+j+shift]=(-1)*fac*tot_der[i,1+j+shift]
                id_tdm+=1

        print('der\n',ham_der) 
        print('si_der\n',si_der) 
           
        # Get permamment/transition dipole moment
        for mn in range(ntdm):
            shift=mn*4 # shift to the next state by 4m columns (x,y,z,mu)    
            dip_num[i,4+shift] = np.linalg.norm(dip_num[i,1+shift:4+shift])
            rem_der[i,4+shift] = np.linalg.norm(rem_der[i,1+shift:4+shift])
            tot_der[i,4+shift] = np.linalg.norm(tot_der[i,1+shift:4+shift])
                    
    print('si_in_zero\n',si_in_zero)    
    print('si_in\n',si_in)
    print('si_sa_zero\n',si_sa_zero)    
    print('si_sa\n',si_sa)
    print('ham_zero\n',ham_zero) 
    print('H_PQ\n',ham)

    np.set_printoptions(precision=4)
    print('1-0 TDM componenets')
    print('dip_num',dip_num[0,1:5])
    print('rem_der',rem_der[0,1:5])
    print('tot_der',tot_der[0,1:5])
  
    #Save covergence plots
    # num_conv_plot(x, field, dip_num, dist, method, dip_cms)
    return dip_num

def init_guess(y, analyt, numer):
    out = y.iname+'_cas'
    xyz = open('00.xyz').read() if y.geom == 'frames' else y.geom
    mol = gto.M(atom=xyz, charge=y.icharge, spin=y.ispin,
                output=y.iname+'_init.log', verbose=4, basis=y.ibasis, symmetry=y.isym)
    weights=[1/x.iroots]*x.iroots
    mf = scf.RHF(mol).run()
    molden.from_mo(mol,'orb_'+y.iname+'_init_hf.molden', mf.mo_coeff)

    fname = 'orb_'+ out
    cas = mcscf.CASSCF(mf, y.norb, y.nel)
    # if y.ispin != 0 and y.isym.lower() != "c2v": 
    #     cas.fcisolver = fci.direct_spin1_symm.FCISolver(mol)
    cas.chkfile = fname
    cas.fcisolver.wfnsym = y.irep
    cas.fix_spin_(ss=y.ispin) 
    if thresh: 
        cas.conv_tol = conv_tol 
        cas.conv_tol_grad = conv_tol_grad 

    print(f'Guess MOs from HF at {y.init:3.5f} ang')
    mo = mcscf.sort_mo(cas, mf.mo_coeff, y.cas_list)
    sa_required_methods=['CMS-PDFT','SA-PDFT','SA-CASSCF']
    if any(x in analyt+numer for x in sa_required_methods):
        cas.state_average_(weights)
        cas.fcisolver.wfnsym = y.irep
        cas.fix_spin_(ss=y.ispin) 
        cas.kernel(mo)
    cas.analyze()
    mo = cas.mo_coeff
    ci = cas.ci
    molden.from_mo(mol, out+'_init_cas.molden', cas.mo_coeff)
    return mo, ci

#-------------Compute energies and dipoles for the given geometry-----------
def get_dipole(x, field, formula, numer, analyt, mo, ci, dist, ontop, ntdm, unit="Debye", dmcFLAG=True):
    out = x.iname+'_'+f"{dist:.2f}"
    xyz = open(str(dist).zfill(2)+'.xyz').read() if x.geom == 'frames' else x.geom
    mol = gto.M(atom=xyz,charge=x.icharge,spin=x.ispin,output=out+'.log',
                verbose=4, basis=x.ibasis, symmetry=x.isym)
        
    #Determine origin
    mass = mol.atom_mass_list()
    charges = mol.atom_charges()
    coords = mol.atom_coords()
    nuc_charge_center = np.einsum('z,zx->x', charges, coords) / charges.sum()
    mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
    # mol.set_common_orig_(mass_center)
    
    weights=[1/x.iroots]*x.iroots

    #HF step
    mf = scf.RHF(mol).set(max_cycle = 1).run()

    #SS/SA-CASSCF step
    fname = 'orb_'+ out #file with orbitals
    cas = mcscf.CASSCF(mf, x.norb, x.nel)
    # if x.ispin != 0 and x.isym.lower() != "c2v": 
    #     cas.fcisolver = fci.direct_spin1_symm.FCISolver(mol)
    # cas.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
    cas.chkfile = fname
    if thresh: 
        cas.conv_tol = conv_tol
        cas.conv_tol_grad = conv_tol_grad
    #! may be redundant if geometry is the same as ini 
    # print('Project orbs from the previous point')
    # mo = mcscf.project_init_guess(cas, mo)
    
    if 'MC-PDFT' in (analyt + numer):
        cas.fcisolver.wfnsym = x.irep
        cas.fix_spin_(ss=x.ispin) 
        e_casscf = cas.kernel(mo,ci)[0]
        mo_ss=cas.mo_coeff
        cas.analyze()
        molden.from_mo(mol, out+'_ss.molden', cas.mo_coeff)

    sa_required_methods=['CMS-PDFT','SA-PDFT','SA-CASSCF']
    if any(x in analyt+numer for x in sa_required_methods):
        cas.state_average_(weights)
        cas.fcisolver.wfnsym = x.irep
        cas.fix_spin_(ss=x.ispin)
        e_casscf = cas.kernel(mo,ci)[0]
        mo_sa = cas.mo_coeff
        ci = cas.ci
        cas.analyze()
        molden.from_mo(mol, out+'_sa.molden', cas.mo_coeff)
    mo=cas.mo_coeff #ONLY TRUE FOR TDM!!!!!!!!!!!!!!!
    ci=cas.ci #ONLY TRUE FOR TDM!!!!!!!!!!!!!!!
    
    #MC-PDFT step
    numeric  = [None]*len(ontop)
    analytic = [None]*len(ontop)
    en_dist  = [None]*len(ontop) # energy array indexed by functional

    origin = "Charge_center" if x.icharge !=0 else "Coord_center"

    for k, ifunc in enumerate(ontop): # Over on-top functionals

        ot='htPBE0' if len(ifunc)>10 else ifunc
        mol.output=x.iname+'_'+ot+'_'+f"{dist:.2f}"+'.log'
        mol.build()
        if len(ifunc) > 10 or ifunc=='ftPBE':
            raise NotImplementedError

        #Initialize to zero
        dip_cms  = np.zeros(4*ntdm).tolist()
        abs_pdft = 0
        abs_cas  = 0
        e_pdft   = 0
#-------------------- Energy ---------------------------
        #---------------- Make a PDFT object ---------------------------
        mc = mcpdft.CASSCF(mf, ifunc, x.norb, x.nel, grids_level=x.grid)
        mc.fcisolver = csf_solver(mol, smult=x.ispin+1, symm=x.isym)
        mc.fcisolver.wfnsym = x.irep
        # mc.fix_spin_(ss=x.ispin)
        mc.chkfile = fname
        mc.max_cycle_macro = 300

        if 'MC-PDFT' in (analyt + numer): 
            raise NotImplementedError
                
        if 'CMS-PDFT' in (analyt + numer): 
            mc = mc.multi_state(weights,'cms')
            # mc.fix_spin_(ss=x.ispin)
            if thresh: 
                mc.conv_tol = conv_tol
                mc.conv_tol_grad = conv_tol_grad
            mc.max_cyc=max_cyc
            mc.kernel(mo,ci)
            e_states=mc.e_states.tolist() 
            mo_sa = mc.mo_coeff #SA-CASSCF MOs
            molden.from_mo(mol, out+'_sa.molden', mc.mo_coeff)
            if dmcFLAG:
                print("Working on Analytic CMS-PDFT TDM")
                id_tdm=0 #enumerate TDM
                for m in range(x.iroots):
                    for n in range(m):
                        shift=id_tdm*4
                        tdm=mc.trans_moment(state=[n,m],unit=unit,origin=origin)
                        abs_cms = np.linalg.norm(tdm)
                        dip_cms[shift:shift+3] = tdm
                        dip_cms[shift+3] = abs_cms
                        id_tdm+=1
            else:
                print("Analytic CMS-PDFT TDM is ignored")
                dip_cms = np.zeros(4*ntdm).tolist()
        
        if 'SA-PDFT' in (analyt + numer): 
            raise NotImplementedError

        if 'SA-CASSCF' in (analyt + numer): 
            mc = mc.state_average_(weights).kernel(mo,ci)
            # mc = mc.state_average_(weights).run(mo,ci)
            e_states = cas.e_states.tolist() #Note cas object not mc 
            # mo_sa = cas.mo_coeff 
            if dmcFLAG:
                print("Working on Analytic SA-CASSCF TDM")
                from functools import reduce
                orbcas = mo[:,cas.ncore:cas.ncore+cas.ncas]
                with mol.with_common_orig(mass_center):
                # with mol.with_common_orig(nuc_charge_center):
                    dip_ints = mol.intor('cint1e_r_sph', comp=3)
                    ic(dip_ints)
                ci_buf=cas.ci

                id_tdm=0 #enumerate TDM
                for m in range(x.iroots):
                    for n in range(m):
                        shift=id_tdm*4
                        t_dm1 = mc.fcisolver.trans_rdm1(ci_buf[n], ci_buf[m], cas.ncas, cas.nelecas)
                        t_dm1_ao = reduce(np.dot, (orbcas, t_dm1, orbcas.T))
                        tdm = nist.AU2DEBYE*np.einsum('xij,ji->x', dip_ints, t_dm1_ao)
                        ic(tdm)
                        abs_cms = np.linalg.norm(tdm)
                        oscil = 2/3*abs(e_states[m]-e_states[n])*abs_cms**2
                        dip_cms[shift:shift+3] = tdm
                        dip_cms[shift+3] = abs_cms
                        id_tdm+=1
            else:
                print("Analytic SA-CASSCF TDM is ignored")
                dip_cms = np.zeros(4*x.iroots).tolist()

       
        # ---------------- Numerical Dipoles ---------------------------
        #---------------------------------------------------------------
        if numer:
            for method in numer:
                if method == 'MC-PDFT': 
                    raise NotImplementedError
                    # mo=mo_ss
                elif method == 'CMS-PDFT' or method == 'SA-PDFT':
                    si_in_zero=mc.si_pdft
                    si_sa_zero=mc.si_mcscf
                    ham_zero=mc.get_heff_pdft()
                    ci0 = mc.ci

                dip_num = numer_run(dist, x, mol, mo, ci, ci0, method, \
                    field, formula, ifunc, out, dip_cms, \
                    si_sa_zero, si_in_zero, ham_zero, ntdm, unit=unit, origin = origin)
        else:
            print("Numerical dipole is ignored")
            dip_num = np.zeros((len(field), 4))
            
        analytic[k] = [dist, abs_cas, abs_pdft] + dip_cms
        numeric [k] = dip_num
        en_dist [k] = [dist, e_casscf, e_pdft] + e_states
    return numeric, analytic, en_dist, mo, ci


# Get dipoles & energies for a fixed distance
def run(x, field, formula, numer, analyt, mo, ci, dist, ontop, scan, dip_scan, en_scan, ntdm, unit='Debye', dmcFLAG=True):
    numeric, analytic, en_dist, mo, ci = \
        get_dipole(x, field, formula, numer, analyt, mo, ci, dist, ontop, ntdm, unit=unit, dmcFLAG=dmcFLAG)

    # Accumulate analytic dipole moments and energies
    for k, ifunc in enumerate(ontop):
        out = 'dmc_'+x.iname+'_'+ifunc+'.txt'
        dip_scan[k].append(analytic[k]) 
        en_scan[k].append(en_dist[k]) 

        # Print & save numeric dipole moments
        if numer:
            for method in numer:
                ot='htPBE0' if len(ifunc)>10 else ifunc
                print(f"Numeric dipole at {cs(dist)} ang found with {cs(method)} ({cs(ot)})")
                header=['Field']
                for i in range(x.iroots):
                    header+=['X', 'Y', 'Z', f'ABS ({i+1})'] 
                sigfig = (".4f",)+(".5f",)*4*ntdm
                numer_point = pdtabulate(numeric[k], header, sigfig)
                print(numer_point)
                action='w' if scan==False else 'a'
                with open(out, action) as f:
                    f.write(f"Numeric dipole at {dist:.3f} ang found with {method} ({ot})\n")
                    f.write(numer_point)
                    f.write('\n')
    return mo, ci


class Molecule:
    def __init__(self, iname, geom, nel, norb, cas_list, init=0, iroots=1, istate=0, 
                icharge=0, isym='C1', irep='A', ispin=0, ibasis='julccpvdz', grid=9):
        self.iname    = iname
        self.geom     = geom
        self.nel      = nel
        self.norb     = norb
        self.cas_list = cas_list
        self.init     = init
        self.iroots   = iroots
        self.istate   = istate
        self.icharge  = icharge
        self.isym     = isym
        self.irep     = irep
        self.ispin    = ispin
        self.ibasis   = ibasis
        self.grid     = grid

#--------------------- Set Up Parameters for Dipole Moment Curves--------------------
# Set the bond range
bonds = np.array([1.0])
# bonds = np.array([])
# bonds = np.array([0.969197274]) # equil 10e9o phenol
# bonds = np.array([0.965469282]) # equil 12e11o phenol 3 states
# bonds = np.array([1.214815430]) # equil 6e6o h2co

# inc=0.1
inc=0.02
# bonds = np.arange(2.0,2.2+inc,inc) # for energy curves
# bonds = np.arange(1.0,3.0+inc,inc) # for energy curves
# bonds = np.array([1.6,2.0,2.1,2.2,2.3,2.4,2.5]) # for energy curves

# External XYZ frames
# bonds = np.arange(0,31,1) 
# bonds = np.arange(0,2,1) 

# Set field range
field = np.linspace(1e-3, 1e-2, num=2)
field = np.linspace(1e-3, 1e-2, num=10)
# field = np.linspace(5e-3, 4e-3, num=2)
field = np.linspace(5e-4, 5e-3, endpoint=True, num=46)
# field = np.array([0.0009,0.001,0.002])
# inc= 1e-3
# field = np.arange(inc, 1e-2, inc)
# thresh = 5e-9
conv_tol = 1e-11
conv_tol_grad= 1e-6
thresh = [conv_tol]+[conv_tol_grad]
max_cyc = 100

# Set name for tPBE0 (HMC-PDFT functional)
hybrid = 't'+ mcpdft.hyb('PBE', 0.25, 'average')
# Set on-top functional
ontop= ['tPBE']
ontop= ['tBLYP']
# ontop= ['tPBE', 'tBLYP', 'tOPBE']
# ontop= [hybrid,'ftPBE']

# Set differentiation technique
formula= "2-point"
# formula = "4-point"

# Set how dipole moments should be computed
numer  = []
analyt = []
numer = ['CMS-PDFT']
# numer = ['SA-PDFT']
analyt = ['CMS-PDFT']
# analyt = ['CMS-PDFT','SA-CASSCF']
# analyt = ['SA-CASSCF']
# analyt = ['SA-PDFT']
dmcFLAG=False 
dmcFLAG=True
unit = 'Debye'
unit = 'AU'

# See class Molecule for the list of variables.
butadiene_4e4o   = Molecule('butadiene_4e4o',geom_butadiene,  4,4, [14,15,16,17],          ibasis='631g*', iroots=2)
furan_6e5o       = Molecule('furan_6e5o',    geom_furan,      6,5, [12,17,18,19,20],       ibasis='631g*', iroots=3)
furan_6e5o_2       = Molecule('furan_6e5o',    geom_furan,      6,5, [12,17,18,19,20],       ibasis='631g*', iroots=2)
furan_6e5o_A1       = Molecule('furan_6e5o',    geom_furan,      6,5, [12,17,18,19,20],       ibasis='631g*', iroots=3,isym='C2v', irep='A1')
furan_6e5o_aug   = Molecule('furan_6e5o_aug',geom_furan,      6,5, [12,17,18,23,29],       ibasis='aug-cc-pvdz', iroots=3)
furan_6e5o_aug_A1   = Molecule('furan_6e5o_aug',geom_furan,      6,5, [12,17,18,23,29],       ibasis='aug-cc-pvdz', iroots=3,isym='C2v', irep='A1')
furan_6e5o_aug_A1_2   = Molecule('furan_6e5o_aug',geom_furan,      6,5, [12,17,18,23,29],       ibasis='aug-cc-pvdz', iroots=2,isym='C2v', irep='A1')
furan_6e5o_aug_2 = Molecule('furan_6e5o_aug',geom_furan,      6,5, [12,17,18,23,29],       ibasis='aug-cc-pvdz', iroots=2)
h2co_6e6o        = Molecule('h2co_6e6o',     geom_h2co,       6,6, [6,7,8,9,10,12],        init=1.20, iroots=2)
phenol_8e7o_sto  = Molecule('phenol_8e7o_sto',geom_phenol,    8,7, [19,23,24,25,26,27,28], iroots=2, ibasis='sto-3g')
phenol_8e7o      = Molecule('phenol_8e7o',  geom_phenol,      8,7, [19,23,24,25,31,32,34], init=0.0, iroots=2)
phenol_8e7o_opt  = Molecule('phenol_opt',  geom_phenol_opt,   8,7, [18,23,24,25,31,33,34], init=0.0, iroots=2)
OH_phenol_10e9o  = Molecule('OH_phenol_10e9o', geom_OH_phenol,10,9,[19,20,23,24,25,26,31,33,34], init=1.3, iroots=2)
OH_phenol3_10e9o =  copy.deepcopy(OH_phenol_10e9o)
OH_phenol3_10e9o.iroots=3
phenol_8e7o_sto_3      =  copy.deepcopy(phenol_8e7o_sto)
phenol_8e7o_sto_4      =  copy.deepcopy(phenol_8e7o_sto)
phenol_8e7o_sto_3.iroots=3
phenol_8e7o_sto_4.iroots=4
spiro_11e10o  = Molecule('spiro_11e10o','frames',11,10,[35,43,50,51,52,53, 55,76,87,100], iroots=2, icharge=1, ispin=1)
phenol_12e11o  = Molecule('phenol_12e11o', geom_phenol_12e11o,12,11,[19,20,21,23,24,25,26,31,33,34,58], init=3.0, iroots=3)
phenol_12e11o.grid=6

# unit tests
h2o_4e4o        = Molecule('h2o_4e4o', geom_h2o, 4,4, [4,5,8,9], iroots=3, grid=1, isym='c2v', irep='A1', ibasis='aug-cc-pVDZ')
furancat_5e5o   = Molecule('furancat_5e5o', geom_furan, 5,5, [12,17,18,19,20], iroots=3, grid=1, icharge=1, ispin=1, ibasis='sto-3g')
furancat_5e5o_2sym = Molecule('furancat_5e5o', geom_furan, 5,5, [12,17,18,19,20], iroots=2, grid=1, icharge=1, ispin=1, ibasis='sto-3g', isym='C2v', irep='A2')
furan_6e5o      = Molecule('furancat_6e5o', geom_furan, 6,5, [12,17,18,19,20], iroots=3, grid=1, isym='C2v', irep='A1', ibasis='sto-3g')
# furancat_5e5o_2 = Molecule('furancat_5e5o', geom_furan, 5,5, [12,17,18,19,20], iroots=2, grid=1, icharge=1, ispin=1, ibasis='sto-3g')
# furancat_5e5o_3 = Molecule('furancat_5e5o', geom_furan, 5,5, [12,17,18,19,20], iroots=3, grid=1, icharge=1, ispin=1, ibasis='sto-3g', isym='C2v', irep='A1')

#Select species for which the dipole moment curves will be constructed
species=[h2co_6e6o]
# species=[phenol_8e7o]
species=[spiro_11e10o]
species=[OH_phenol3_10e9o]
species=[OH_phenol_10e9o]
species=[phenol_8e7o]
species=[phenol_8e7o_opt]
species=[phenol_8e7o_sto]
species=[phenol_8e7o_sto_4]
species=[phenol_8e7o_sto_3]
species=[phenol_12e11o]
species=[butadiene_4e4o]
species=[furan_6e5o_aug_2]
# species=[furan_6e5o]
# species=[furan_6e5o_aug]
# species=[furan_6e5o_A1]
# species=[furan_6e5o_aug_A1]
# species=[furan_6e5o_aug_A1_2]
species=[furancat_5e5o]
species=[furan_6e5o]
# species=[furancat_5e5o_2]
# species=[furancat_5e5o_3]
species=[furancat_5e5o_2sym]


# ---------------------- MAIN DRIVER OVER DISTANCES -------------------
dip_scan = [ [] for _ in range(len(ontop)) ]
en_scan  = [ [] for _ in range(len(ontop)) ] # Empty lists per functional
scan=False if len(bonds)==1 else True #Determine if multiple bonds are passed
#In case if numer and analyt are not supplied
# exec('try:numer\nexcept:numer=None')
# exec('try:analyt\nexcept:analyt=None')

for x in species:
    # Get MOs before running a scan
    y=copy.deepcopy(x)
    y.geom=y.geom.format(y.init)
    mo, ci = init_guess(y, analyt, numer) if x !=spiro_11e10o else 0 #spiro MOs should be provided manually
    ntdm=np.math.factorial(x.iroots)//(np.math.factorial(x.iroots-2)*2)

    for i, dist in enumerate(bonds, start=0):
        # Update the bond length in the instance
        if i==0: template=x.geom
        x.geom=template.format(dist)

        # MOs and CI vectors are taken from the previous point
        mo, ci = run(x, field, formula, numer, analyt, mo, ci, dist, ontop, scan, dip_scan, en_scan, ntdm, unit=unit, dmcFLAG=dmcFLAG)
        
        dip_head = ['Distance','CASSCF','MC-PDFT']
        for j in range(x.iroots):
            dip_head+=['X', 'Y', 'Z', f'ABS ({cs(j+1)})']
        dip_sig = (".2f",)+(".5f",)*(2+4*ntdm)
        
        en_head=['Distance', 'CASSCF', 'MC-PDFT']
        for method in analyt:
            for jj in range(x.iroots): 
                line=f'{method} ({cs(jj+1)})'
                en_head.append(line)
            en_sig = (".2f",)+(".8f",)*(2+x.iroots)

        for k, ifunc in enumerate(ontop):
            out = x.iname+'_'+ifunc+'.txt'
            if analyt:
                print(f"Analytic dipole moments found with {cs(ifunc)}")
                dip_table = pdtabulate(dip_scan[k], dip_head, dip_sig)
                print(dip_table)

            print(f"Energies found with {cs(ifunc)}")
            en_table = pdtabulate(en_scan[k], en_head, en_sig)
            print(en_table)
            action='a' if numer else 'w'
            with open(out, action) as f:
                # f.write(f"TDMs are found w.r.t. nuclear charge center {ifunc} \n")
                f.write(f"The on-top functional is {ifunc} \n")
                f.write(en_table)
                f.write('\n')
                if analyt:
                    f.write(dip_table)
                    f.write('\n')
