
from tabulate import tabulate
from pyscf import gto, scf, mcscf
from pyscf.lib import logger
from mrh.my_pyscf import mcpdft
from mrh.my_pyscf.fci import csf_solver
from scipy import linalg
import numpy as np
from pyscf.tools import molden
import copy
from pyscf.geomopt import geometric_solver
from pyscf.gto import inertia_moment
import os

os.environ['OMP_NUM_THREADS'] = "4"

geom_h2co= '''
H       -0.000000000      0.950627350     -0.591483790
H       -0.000000000     -0.950627350     -0.591483790
C        0.000000000      0.000000000      0.000000000
O        0.000000000      0.000000000      1.214815430
'''
geom_phenol= '''
C   1.2088411311  -1.1934487860  -0.0000000000
C  -0.1934806202  -1.2232957757   0.0000000000
C   1.9008496952   0.0335152175  -0.0000000000
H  -0.7483838204  -2.1637040706  -0.0000000000
H   2.9936831984   0.0501093283   0.0000000000
C  -0.9113271438  -0.0126498741   0.0000000000
C   1.1741190915   1.2367572971   0.0000000000
H   1.6987306555   2.1964611483   0.0000000000
C  -0.2320396097   1.2204980464   0.0000000000
H  -0.7942897667   2.1611059180  -0.0000000000
H   1.7648611566  -2.1352555966  -0.0000000000
O  -2.3065373104  -0.1096748395   0.0000000000
H  -2.7134613170   0.7897986861   0.0000000000
'''
geom_aniline= '''
C       -0.002313810     -1.204111330     -0.228014550
C       -0.003527520      0.000028140     -0.945592500
C       -0.001640490     -1.198455380      1.161538220
H       -0.003562760     -2.148187680      1.696716190
C       -0.002283920      1.204152530     -0.227983130
C       -0.001128150     -0.000005380      1.870761020
H        0.002154720      2.150752830     -0.770697400
H       -0.003831400     -0.000019030      2.959438070
C       -0.001550560      1.198463750      1.161567130
H       -0.003446320      2.148181250      1.696771630
H        0.001897900     -2.150702490     -0.770750140
N        0.052865280      0.000067360     -2.336925680
H       -0.290675680     -0.840700830     -2.785247650
H       -0.291418730      0.840536260     -2.785239670
'''
geom_trans_aminophenol= '''
C       -0.002384343     -1.238959337     -0.208377558
C       -0.003598053     -0.034819867     -0.925955508
C       -0.001711023     -1.233303387      1.181175212
C       -0.002354453      1.169304523     -0.208346138
C       -0.001198683     -0.034853387      1.890398012
H        0.002084187      2.115904823     -0.751060408
H       -0.003901933     -0.034867037      2.979075062
C       -0.001621093      1.163615743      1.181204122
H       -0.003516853      2.113333243      1.716408622
H        0.001827367     -2.185550497     -0.751113148
N        0.052794747     -0.034780647     -2.317288688
H       -0.290746213     -0.875548837     -2.765610658
H       -0.291489263      0.805688253     -2.765602678
O       -0.004109138     -2.418135616      1.848832937
H       -0.007522253     -2.243681835      2.799075988
'''
geom_cis_aminophenol= '''
C       -0.002384343     -1.238959337     -0.208377558
C       -0.003598053     -0.034819867     -0.925955508
C       -0.001711023     -1.233303387      1.181175212
C       -0.002354453      1.169304523     -0.208346138
C       -0.001198683     -0.034853387      1.890398012
H        0.002084187      2.115904823     -0.751060408
H       -0.003901933     -0.034867037      2.979075062
C       -0.001621093      1.163615743      1.181204122
H       -0.003516853      2.113333243      1.716408622
H        0.001827367     -2.185550497     -0.751113148
N        0.052794747     -0.034780647     -2.317288688
H       -0.290746213     -0.875548837     -2.765610658
H       -0.291489263      0.805688253     -2.765602678
O       -0.004109138     -2.418135616      1.848832937
H       -0.001805237     -3.140612590      1.207405357
'''
geom_aminobenzoic= '''
C       -0.002313810     -1.204111330     -0.228014550
C       -0.003527520      0.000028140     -0.945592500
C       -0.001640490     -1.198455380      1.161538220
H       -0.003562760     -2.148187680      1.696716190
C       -0.002283920      1.204152530     -0.227983130
C       -0.001128150     -0.000005380      1.870761020
H        0.002154720      2.150752830     -0.770697400
C       -0.001550560      1.198463750      1.161567130
H       -0.003446320      2.148181250      1.696771630
H        0.001897900     -2.150702490     -0.770750140
N        0.052865280      0.000067360     -2.336925680
H       -0.290675680     -0.840700830     -2.785247650
H       -0.291418730      0.840536260     -2.785239670
C       -0.004952050     -0.000024689      3.410756272
O       -0.006681241      0.995900692      4.107151613
O       -0.006276915     -1.248686222      3.944317953
H       -0.008668024     -1.121212185      4.907286455
'''
geom_benzonitrile= '''
C        1.583040598     -1.217302916      0.000000000
C        2.284906162     -0.000021972      0.000000000
C        0.182483516     -1.224446918      0.000000000
H       -0.374872241     -2.170406195      0.000000000
C        1.583015797      1.217214297      0.000000000
C       -0.524318029     -0.000019519      0.000000000
H        2.130090821      2.169958229      0.000000000
C        0.182439029      1.224372513      0.000000000
H       -0.375081013      2.170189086      0.000000000
H        2.130291062     -2.169930639      0.000000000
H        3.383834171     -0.000016223      0.000000000
C       -1.959950220      0.000105353      0.000000000
N       -3.136456874      0.000304906      0.000000000
'''
geom_fluorobenzene= '''
C        0.183499999      1.226261448      0.000000000
C       -1.221421109      1.211246096      0.000000000
C        0.855142196      0.000760301      0.000000000
C       -1.923298868     -0.005687081      0.000000000
C        0.189109961     -1.227818476      0.000000000
H       -3.021636707     -0.008151529      0.000000000
H        0.764791554     -2.162287676      0.000000000
C       -1.215815905     -1.219374361      0.000000000
H       -1.759511085     -2.174539458      0.000000000
H        0.754810367      2.163408242      0.000000000
F        2.224458801      0.003499952      0.000000000
H       -1.769650313      2.163822617      0.000000000
'''
geom_x4_fluoroindole= '''
C       -0.997063747     -1.386858603      0.000000000
C        0.253698842     -0.741287275      0.000000000
C        0.245028672      0.691005496      0.000000000
C       -0.940919643      1.448959960      0.000000000
C       -2.149750740      0.745969844      0.000000000
C       -2.189059355     -0.672392841      0.000000000
C        1.625963469     -1.171070038      0.000000000
C        2.398342258     -0.023398458      0.000000000
N        1.571190784      1.090672642      0.000000000
F       -1.024387347     -2.755880000      0.000000000
H       -3.097389978      1.301576455      0.000000000
H        1.993715761     -2.199377587      0.000000000
H        3.485158617      0.092426458      0.000000000
H        1.888693781      2.055783218      0.000000000
H       -0.924203770      2.547295598      0.000000000
H       -3.143940997     -1.212805188      0.000000000
'''
geom_x5_fluoroindole= '''
C       -0.975668428     -1.439542967      0.000000000
C        0.260550416     -0.750475317      0.000000000
C        0.259931904      0.683272965      0.000000000
C       -0.925983675      1.436866071      0.000000000
C       -2.140202662      0.741644947      0.000000000
C       -2.132214869     -0.669337609      0.000000000
C        1.636753689     -1.170901938      0.000000000
C        2.410802065     -0.022475570      0.000000000
N        1.585670934      1.089897540      0.000000000
H       -1.036880198     -2.535231085      0.000000000
H       -3.101610468      1.270216491      0.000000000
H        2.014234155     -2.196774348      0.000000000
H        3.497899054      0.092268994      0.000000000
H        1.902663340      2.055054550      0.000000000
H       -0.912476767      2.535771394      0.000000000
F       -3.348391884     -1.309634438      0.000000000
'''
geom_x6_fluoroindole= '''
C       -0.977335879     -1.446391665      0.000000000
C        0.261510299     -0.765890733      0.000000000
C        0.259497960      0.668007468      0.000000000
C       -0.924546468      1.428181472      0.000000000
C       -2.109169526      0.697057290      0.000000000
C       -2.166580740     -0.711961762      0.000000000
C        1.641204094     -1.183280012      0.000000000
C        2.412509328     -0.036452321      0.000000000
N        1.580077664      1.077097036      0.000000000
H       -1.009210235     -2.544835395      0.000000000
F       -3.295479796      1.388106784      0.000000000
H        2.019820402     -2.208854674      0.000000000
H        3.498711877      0.084720356      0.000000000
H        1.894179832      2.043306240      0.000000000
H       -0.942539816      2.525444423      0.000000000
H       -3.147572390     -1.203634826      0.000000000
'''
geom_x7_azaindole= '''
C       -0.999660487     -1.474699415      0.000000000
C        0.243499923     -0.810207692      0.000000000
C        0.195667723      0.625072304      0.000000000
N       -0.892661381      1.406257824      0.000000000
C       -2.055248262      0.727702274      0.000000000
C       -2.155343066     -0.684761849      0.000000000
C        1.631975401     -1.184105052      0.000000000
C        2.362653910     -0.006888089      0.000000000
N        1.500705443      1.079244605      0.000000000
H       -1.061586315     -2.572388110      0.000000000
H       -2.969304761      1.341194501      0.000000000
H       -3.150560143     -1.149656422      0.000000000
H        2.051080397     -2.194090248      0.000000000
H        3.445221618      0.144098316      0.000000000
H        1.764120463      2.061928688      0.000000000
'''
geom_x5_cyanoindole= '''
C       -1.013027796     -1.455427845      0.000000000
C        0.217000512     -0.772635609      0.000000000
C        0.215435617      0.664275669      0.000000000
C       -0.973471568      1.419886362      0.000000000
C       -2.178436296      0.722680021      0.000000000
C       -2.205200802     -0.704684430      0.000000000
C        1.595056061     -1.190556478      0.000000000
C        2.367796183     -0.044447873      0.000000000
N        1.535308797      1.068563336      0.000000000
H       -1.054226845     -2.553935324      0.000000000
H       -3.133433725      1.265590052      0.000000000
H        1.975184310     -2.216167341      0.000000000
H        3.453737935      0.079973842      0.000000000
H        1.852871858      2.034006776      0.000000000
H       -0.952884938      2.518396271      0.000000000
C       -3.456774097     -1.387229296      0.000000000
N       -4.473605204     -1.967971798      0.000000000
'''
geom_x2_aminobenzonitrile= '''
C       -0.007747246     -1.253007735     -0.236429247
C       -0.020367732     -0.026888463     -0.971509973
C        0.009975114     -1.240087880      1.178334266
H        0.019493317     -2.199726587      1.711024103
C       -0.023495013      1.185640623     -0.241078345
C        0.012449237     -0.031705642      1.878136798
H       -0.028892827      2.138987343     -0.788634696
H        0.027070013     -0.028365175      2.975624948
C       -0.008379540      1.178011925      1.155766077
H       -0.012277749      2.136138503      1.694633765
N        0.025550135     -0.031688747     -2.350821924
H       -0.194528792     -0.904216389     -2.832242260
H       -0.265864012      0.810849895     -2.837784696
C       -0.024727541     -2.481850413     -0.962772878
N       -0.045681908     -3.456803503     -1.624337411
'''
geom_x3_aminobenzonitrile= '''
C       -0.005689717     -1.244422561     -0.225955364
C       -0.009221060     -0.027055456     -0.939143386
C       -0.002913054     -1.240870520      1.184413666
C       -0.006353375      1.186667037     -0.210645839
C        0.000417324     -0.026044935      1.905448688
H       -0.003381793      2.143905531     -0.753012007
H        0.003232439     -0.039764000      3.002278689
C       -0.000092204      1.178672820      1.189950428
H        0.000935437      2.133530601      1.733393670
H       -0.000267830     -2.202456921     -0.764633493
N        0.051904994     -0.021173944     -2.333827509
H       -0.279158860     -0.866842090     -2.796466920
H       -0.290277265      0.824396394     -2.788037505
C       -0.006197262     -2.492941982      1.887813036
N       -0.009140541     -3.518283818      2.464833467
'''
geom_x4_aminobenzonitrile= '''
C       -0.006049498     -1.219387052     -0.196906927
C       -0.015522354      0.000243852     -0.919089555
C       -0.004024722     -1.219067510      1.196814607
H       -0.004819295     -2.171322176      1.746829037
C       -0.013102332      1.217972381     -0.194144247
C       -0.004683074     -0.002884971      1.919716889
H       -0.001121622      2.172888892     -0.738564588
C       -0.006588404      1.215692859      1.199513270
H       -0.011341120      2.168661053      1.748480128
H       -0.008319787     -2.173754711     -0.742231417
N        0.036911079     -0.002026666     -2.306555337
H       -0.271724392     -0.847732344     -2.782989130
H       -0.260215254      0.851091026     -2.779483213
C        0.000214524      0.003701418      3.348694904
N        0.012202025      0.005875169      4.528327060
'''
geom_aniline_opt= '''
C       -0.003109000     -1.211463000     -0.228692000
C       -0.005053000      0.000019000     -0.942698000
C       -0.000388000     -1.203143000      1.166082000
H       -0.000369000     -2.157744000      1.708152000
C       -0.003089000      1.211523000     -0.228717000
C        0.001383000     -0.000011000      1.873081000
H       -0.000182000      2.163549000     -0.776989000
H        0.003461000     -0.000271000      2.969887000
C       -0.000361000      1.203140000      1.166051000
H       -0.000317000      2.157827000      1.707964000
H       -0.000257000     -2.163470000     -0.776989000
N        0.035931000      0.000015000     -2.326092000
H       -0.229140000     -0.853939000     -2.807344000
H       -0.229124000      0.853968000     -2.807353000
'''
geom_dimethoxybenzene= '''
C  -0.0301160972  -1.1867798744  -0.3601099668
C  -0.0339083072   0.0725848633  -0.9994776978
C  -0.0037246106  -1.2475775961   1.0411688058
C  -0.0063753683   1.2623993075  -0.2521140137
C   0.0212979886  -0.0612380954   1.8095933272
H  -0.0053035772   2.2464711668  -0.7334013362
H   0.0392176476  -0.1335886523   2.9043651420
C   0.0174435363   1.1699700517   1.1541018682
H   0.0250412219   2.0952738903   1.7483947236
H  -0.0570901464  -2.0821528852  -0.9921131488
O  -0.0712814591   0.0145987022  -2.3734442712
O  -0.0089519395  -2.4166816866   1.7625193514
C  -0.0363493513  -3.6422498982   1.0276405470
H   0.8718749528  -3.7571383129   0.3975478684
H  -0.0544956897  -4.4414162922   1.7909331590
H  -0.9405859409  -3.7158910054   0.3878950544
C  -0.0775476436   1.2578406710  -3.0779274175
H   0.8438593690   1.8402968479  -2.8729341549
H  -0.1249354297   0.9943901577  -4.1448238350
H  -0.9656811296   1.8687909865  -2.8162447619
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
H        0.000000000      0.000000000      1.3
'''
geom_OH_phenol3= '''
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
H        0.000000000      0.000000000      1.3
'''
geom_aniline_Cs_0= '''
C       -0.236910000     -1.210937000      0.000000000
C       -0.943365000      0.006348000     -0.000001000
C        1.157592000     -1.210319000      0.000000000
H        1.692842000     -2.168846000     -0.000001000
C       -0.220558000      1.213997000      0.000000000
C        1.873672000     -0.012648000      0.000000000
H       -0.761995000      2.169885000      0.000000000
H        2.970374000     -0.020042000      0.000001000
C        1.173808000      1.194573000      0.000000000
H        1.721937000      2.145793000     -0.000001000
H       -0.791188000     -2.159437000      0.000000000
N       -2.317779000      0.015618000      0.000000000
H       -2.845998000     -0.847930000      0.000000000
H       -2.834301000      0.886213000      0.000000000
'''


# def inertia_moment(mol, mass=None, coords=None):
#     if mass is None:
#         mass = mol.atom_mass_list()
#     if coords is None:
#         coords = mol.atom_coords()
#     mass_center = np.einsum('i,ij->j', mass, coords)/mass.sum()
#     coords = coords - mass_center
#     im = np.einsum('i,ij,ik->jk', mass, coords, coords)
#     im = np.eye(3) * im.trace() - im
#     return im, mass_center

def transform_dip(eq_dip,mol_eq):
    matrix = inertia_moment(mol_eq, mass=None, coords=None)
    moments, abc_vec=np.linalg.eigh(matrix) #ascending order
    origin=np.eye(3)
    rot_mat=np.linalg.solve(origin,abc_vec)
    eq_dip_abc=np.dot(eq_dip,rot_mat)
    return eq_dip_abc

def cms_dip(x, ifunc):
    out = x.iname+'_'+str(x.istate)
    mol = gto.M(atom=x.geom, charge=x.icharge, spin=x.ispin,
                    output=out+'.log', verbose=4, basis=x.ibasis)
    #HF step
    mf = scf.RHF(mol).run()
    molden.from_mo(mol, out+'_hf_ini.molden', mf.mo_coeff)


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
     
    # mc = mcscf.CASCI(mf, x.norb, x.nel).state_specific_(0)
    # mc.fix_spin_(ss=0)
    # mc.natorb=True
    # emc = mc.casci(mo)[0]
    # mc.analyze(large_ci_tol=0.05)
    # print(emc)
    # print(mc.mo_occ)

    # mc = mcscf.CASCI(mf, x.norb, x.nel).state_specific_(1)
    # mc.natorb=True
    # mc.fix_spin_(ss=0)
    # emc = mc.casci(mo)[0]
    # mc.analyze(large_ci_tol=0.05)
    # print(emc)
    # print(mc.mo_occ)

    # ###-------------------- MC-PDFT Energy ---------------------------
    mc = mcpdft.CASSCF(mf, ifunc, x.norb, x.nel, grids_level=9)
    mc.fcisolver = csf_solver(mol, smult=x.ispin+1)
    mc.max_cycle_macro = 200
    mo = mcscf.sort_mo(mc, mf.mo_coeff, x.cas_list)
    e_pdft = mc.kernel(mo)[0]
    # mo_ss = mc.mo_coeff #SS-CASSCF MOs
    # molden.from_mo(mol, out+'_ss_cas.molden', mc.mo_coeff)
    

    #-------------------- CMS-PDFT Energy ---------------------------
    # mc.natorb=True
    weights=[1/x.iroot]*x.iroot #Equal weights only
    mc=mc.state_interaction(weights,'cms').run(mo)
    molden.from_mo(mol, out+'_cms.molden', mc.mo_coeff)
    print('CMS energy is completed')

    if x.opt == True:

        #Optimize geometry
        mol_eq=mc.nuc_grad_method().as_scanner(state=x.istate).optimizer().kernel()

        #Equilibrium dipole moment
        mf_eq = scf.RHF(mol_eq).run()
        mc_eq = mcpdft.CASSCF(mf_eq, ifunc, x.norb, x.nel, grids_level=9)
        mc_eq.fcisolver = csf_solver(mol_eq, smult=x.ispin+1)
        mc_eq.max_cycle_macro = 200
        mo = mcscf.project_init_guess(mc_eq, mo)
        e_pdft = mc_eq.kernel(mo)[0]
        weights=[1/x.iroot]*x.iroot #Equal weights only
        mc_eq = mc_eq.state_interaction(weights,'cms').run(mo)
                
        e_cms=mc_eq.e_states.tolist() #List of CMS energies
        molden.from_mo(mol_eq, out+"_"+f"{x.istate}"+'_opt.molden', mc_eq.mo_coeff)
        print('CMS-PDFT energies at the optimized geometry\n',e_cms)
       
        eq_dip=mc_eq.dip_moment(state=x.istate, unit='Debye')
    
    else: #Preoptimized geometry
        e_cms=mc.e_states.tolist() #List of CMS energies
        molden.from_mo(mol, out+"_"+f"{x.istate}"+'_opt.molden', mc.mo_coeff)
        print('CMS-PDFT energies at the PREoptimized geometry\n',e_cms)

        eq_dip=mc.dip_moment(state=x.istate, unit='Debye')
        
    eq_dip_abc=transform_dip(eq_dip,mol)
    val=np.linalg.norm(eq_dip)
    print(x.iname, 'state is ', x.istate, 'ABS = ', val)
    print('Dipole along XYZ axes:\n', eq_dip)
    print('Dipole along principle axes\n', eq_dip_abc)

    return

class Molecule:
    def __init__(self, iname, geom, icharge, isym, irep, ispin, ibasis, iroot,
                #  nel, norb, istate, cas_list, opt):
                 nel, norb, istate, cas_list, opt=True):
        self.iname    = iname
        self.geom     = geom
        self.icharge  = icharge
        self.isym     = isym
        self.irep     = irep
        self.ispin    = ispin
        self.ibasis   = ibasis
        self.iroot    = iroot
        self.nel      = nel
        self.norb     = norb
        self.istate    = istate
        self.cas_list = cas_list
        self.opt = opt

ifunc= 'tPBE'

x2_aminobenzonitrile_12e11o_0 = Molecule('x2_aminobenzonitrile_12e11o',geom_x2_aminobenzonitrile, 0, 'C1', 'A',  0, 'julccpvdz', 2, 12,11, 0, [23,27,28,29,30,31, 35,38,39,49,50])
x3_aminobenzonitrile_12e11o_0 = Molecule('x3_aminobenzonitrile_12e11o',geom_x3_aminobenzonitrile, 0, 'C1', 'A',  0, 'julccpvdz', 2, 12,11, 0, [23,27,28,29,30,31, 34,35,38,39,49])
x4_aminobenzonitrile_12e11o_0 = Molecule('x4_aminobenzonitrile_12e11o',geom_x4_aminobenzonitrile, 0, 'C1', 'A',  0, 'julccpvdz', 2, 12,11, 0, [24,27,28,29,30,31, 35,36,38,39,49])
aniline_8e7o_opt_0            = Molecule('aniline_8e7o',           geom_aniline_opt,      0, 'C1', 'A',  0, 'julccpvdz', 2, 8,7,  0, [20,23,24,25,31,33,34], False)
aniline_8e7o_0                = Molecule('aniline_8e7o',           geom_aniline,          0, 'C1', 'A',  0, 'julccpvdz', 2, 8,7,  0, [20,23,24,25,31,33,34])
trans_aminophenol_10e8o_0     = Molecule('trans_aminophenol_10e8o',geom_trans_aminophenol,0, 'C1', 'A',  0, 'julccpvdz', 2, 10,8, 0, [24,25,27,28,29,35,47,49])
cis_aminophenol_10e8o_0       = Molecule('cis_aminophenol_10e8o',  geom_cis_aminophenol,  0, 'C1', 'A',  0, 'julccpvdz', 2, 10,8, 0, [22,24,27,28,29,35,37,38])
aminobenzoic_10e8o_0          = Molecule('aminobenzoic_10e8o',     geom_aminobenzoic,     0, 'C1', 'A',  0, 'julccpvdz', 2, 10,8, 0, [22,24,27,28,29,35,37,38])
h2co_6e6o_0                   = Molecule('h2co_6e6o',              geom_h2co,             0, 'C1', 'A',  0, 'julccpvdz', 2, 6,6,  0, [6,7,8,9,10,12])
phenol_8e7o_0                 = Molecule('phenol_8e7o',            geom_phenol,           0, 'C1', 'A',  0, 'julccpvdz', 2, 8,7,  0, [19,23,24,25,31,33,34])
#!!!
OH_phenol_10e9o_0             = Molecule('OH_phenol_10e9o',            geom_OH_phenol,           0, 'C1', 'A',  0, 'julccpvdz', 2, 10,9,  0, [19,20,23,24,25,26,31,33,34])
benzonitrile_10e10o_0         = Molecule('benzonitrile_10e10o',    geom_benzonitrile,     0, 'C1', 'A',  0, 'julccpvdz', 2, 10,10,0, [21,24,25,26,27,29,32,43,45,46])
fluorobenzene_8e7o_0          = Molecule('fluorobenzene_8e7o',     geom_fluorobenzene,    0, 'C1', 'A',  0, 'julccpvdz', 2, 8,7,  0, [16,23,24,25,31,32,34])
fluorobenzene_8e7o_1          = Molecule('fluorobenzene_8e7o',     geom_fluorobenzene,    0, 'C1', 'A',  0, 'julccpvdz', 2, 8,7,  1, [16,23,24,25,31,32,34])
fluorobenzene_6e6o_0_tz          = Molecule('fluorobenzene_6e6o',     geom_fluorobenzene,    0, 'C1', 'A',  0, 'julccpvtz', 2, 6,6,  0, [23,24,25,31,32,34])
fluorobenzene_6e6o_1_tz          = Molecule('fluorobenzene_6e6o',     geom_fluorobenzene,    0, 'C1', 'A',  0, 'julccpvtz', 2, 6,6,  1, [23,24,25,31,32,34])
fluorobenzene_6e6o_0          = Molecule('fluorobenzene_6e6o',     geom_fluorobenzene,    0, 'C1', 'A',  0, 'julccpvdz', 2, 6,6,  0, [23,24,25,31,32,34])
fluorobenzene_6e6o_1          = Molecule('fluorobenzene_6e6o',     geom_fluorobenzene,    0, 'C1', 'A',  0, 'julccpvdz', 2, 6,6,  1, [23,24,25,31,32,34])
x4_fluoroindole_10e9o_0       = Molecule('x4_fluoroindole_10e9o',  geom_x4_fluoroindole,  0, 'C1', 'A',  0, 'julccpvdz', 2, 10,9, 0, [27,32,33,34,35,41,43,46,47])
x4_fluoroindole_10e9o_1       = Molecule('x4_fluoroindole_10e9o',  geom_x4_fluoroindole,  0, 'C1', 'A',  0, 'julccpvdz', 2, 10,9, 1, [27,32,33,34,35,41,43,46,47])
x5_fluoroindole_10e9o_0       = Molecule('x5_fluoroindole_10e9o',  geom_x5_fluoroindole,  0, 'C1', 'A',  0, 'julccpvdz', 2, 10,9, 0, [27,32,33,34,35,40,44,45,51])
x5_fluoroindole_10e9o_1       = Molecule('x5_fluoroindole_10e9o',  geom_x5_fluoroindole,  0, 'C1', 'A',  0, 'julccpvdz', 2, 10,9, 1, [27,32,33,34,35,40,44,45,51])
x6_fluoroindole_10e9o_0       = Molecule('x6_fluoroindole_10e9o',  geom_x6_fluoroindole,  0, 'C1', 'A',  0, 'julccpvdz', 2, 10,9, 0, [28,32,33,34,35,40,44,45,48])
x6_fluoroindole_10e9o_1       = Molecule('x6_fluoroindole_10e9o',  geom_x6_fluoroindole,  0, 'C1', 'A',  0, 'julccpvdz', 2, 10,9, 1, [28,32,33,34,35,40,44,45,48])
x7_azaindole_10e9o_0          = Molecule('x7_azaindole_10e9o',     geom_x7_azaindole,     0, 'C1', 'A',  0, 'julccpvdz', 2, 10,9, 0, [22,27,29,30,31,35,38,43,44])
x7_azaindole_10e9o_1          = Molecule('x7_azaindole_10e9o',     geom_x7_azaindole,     0, 'C1', 'A',  0, 'julccpvdz', 2, 10,9, 1, [22,27,29,30,31,35,38,43,44])
x5_cyanoindole_14e13o_0       = Molecule('x5_cyanoindole_14e13o',  geom_x5_cyanoindole,   0, 'C1', 'A',  0, 'julccpvdz', 2, 14,13,0, [26,31,33,34,35,36,37, 41,44,45,46,49,56])
x5_cyanoindole_14e13o_1       = Molecule('x5_cyanoindole_14e13o',  geom_x5_cyanoindole,   0, 'C1', 'A',  0, 'julccpvdz', 2, 14,13,1, [26,31,33,34,35,36,37, 41,44,45,46,49,56])
dimethoxybenzene_10e8o_0      = Molecule('dimethoxybenzene_10e8o', geom_dimethoxybenzene, 0, 'C1', 'A',  0, 'julccpvdz', 2, 10,8, 0, [24,25,35,36,37,47,50,56])
dimethoxybenzene_10e8o_1      = Molecule('dimethoxybenzene_10e8o', geom_dimethoxybenzene, 0, 'C1', 'A',  0, 'julccpvdz', 2, 10,8, 1, [24,25,35,36,37,47,50,56])
aniline_8e7o_Cs_0 = Molecule('aniline_8e7o_Cs_0',geom_aniline_Cs_0,0, 'C1',   'A',  0, 'julccpvdz', 2, 8,7, 0, [20,23,24,25,31,33,34], False)
OH_phenol3_10e9o_0= Molecule('OH_phenol3_10e9o', geom_OH_phenol3,0, 'C1', 'A',  0, 'julccpvdz', 3, 10,9,  0, [19,20,23,24,25,26,31,33,34])
# OH_phenol3_10e9o_0= Molecule('OH_phenol3_10e9o', geom_OH_phenol3,0, 'C1', 'A',  0, 'julccpvdz', 3, 10,9,  0, [19,20,23,24,25,26,31,33,34]) #old
# species=[h2co_6e6o, aniline_8e7o, aminobenzoic_10e8o, trans_aminophenol_10e8o, cis_aminophenol_10e8o, fluorobenzene_6e6o, benzonitrile_10e10o ]
# species=[fluorobenzene_6e6o, x4_fluoroindole_10e9o, x5_fluoroindole_10e9o, x6_fluoroindole_10e9o, x7_azaindole_11e9o, x5_cyanoindole_14e13o]
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
species=[aniline_8e7o_Cs_0]
species=[aniline_8e7o_opt_0]
species=[OH_phenol3_10e9o_0]
species=[h2co_6e6o_0]
for x in species:
    cms_dip(x, ifunc)