import numpy as np
#import cupy as np
import time

inputtrk = {'trk_cov12': 1.5431574240665213e-07, 'trk_theta': 0.35594117641448975, 'trk_phi': -2.613372802734375, 'trk_cov11': 7.526661534029699e-07, 'trk_cov14': 4.3124190760040646e-07, 'trk_cov15': 1.68539375883925e-06, 'trk_q': 1, 'trk_cov25': 6.676875566525437e-08, 'trk_cov24': 3.1068903991780678e-09, 'trk_cov23': 2.649119409845118e-07, 'trk_cov22': 9.626245400795597e-08, 'trk_cov45': 7.356584799406111e-05, 'trk_cov44': 0.00040678296006807003, 'trk_hitidx': [1, 0, 17, 16, 36, 35, 33, 34, 59, 58, 70, 85, 101, 102, 116, 117, 132, 133, 152, 169, 187, 202], 'trk_cov01': 4.1375109560704004e-08, 'trk_cov00': 6.290299552347278e-07, 'trk_cov03': -2.804026640189443e-06, 'trk_cov02': 2.0973730840978533e-07, 'trk_cov05': -7.755406890332818e-07, 'trk_cov04': -2.419662877381737e-07, 'trk_ipt': 0.23732035065189902, 'trk_y': -7.723824977874756, 'trk_x': -12.806846618652344, 'trk_z': 38.13014221191406, 'trk_cov13': 6.219111130687595e-06, 'trk_cov55': 0.0002306247719158348, 'trk_cov33': 0.00253512163402557, 'trk_cov34': 0.000923913115050627, 'trk_cov35': 0.0008420574605423793}
inputhit = {'hit_z': 57.8067626953125, 'hit_y': -12.24150276184082, 'hit_x': -20.7824649810791, 'hit_yz': 0.00012282167153898627, 'hit_yy': 2.8030024168401724e-06, 'hit_xx': 2.545517190810642e-06, 'hit_xy': -2.6680759219743777e-06, 'hit_zz': 11.385087966918945, 'hit_zx': 0.00014160551654640585}

def PosInMtrx(i, j, D):
    return i*D+j

def SymOffsets33(i):
    offs = [ 0, 1, 3, 1, 2, 4, 3, 4, 5]
    return offs[i]

def SymOffsets66(i):
    offs = [0, 1, 3, 6, 10, 15, 1, 2, 4, 7, 11, 16, 3, 4, 5, 8, 12, 17, 6, 7, 8, 9, 13, 18, 10, 11, 12, 13, 14, 19, 15, 16, 17, 18, 19, 20]
    return offs[i]

# for now this is just the same as list, so it's useless... but maybe it will be expanded in the future
class MatriplNP:
    def __init__(self, elem_array):
        self.m_ = elem_array
    def __getitem__(self, key):
        return self.m_[key]
    def __setitem__(self, key, value):
        self.m_[key] = value

def prepareTracks():
    na_trk_x = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_x"]
    na_trk_y = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_y"]
    na_trk_z = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_z"]
    na_trk_ipt = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_ipt"]
    na_trk_theta = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_theta"]
    na_trk_phi = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_phi"]
    na_trk_q = np.random.choice(a=[1, -1], size=(nevts,ntrks), p=[0.5, 0.5])*inputtrk["trk_q"]
    na_trk_cov = MatriplNP([None] * 21)
    na_trk_cov[SymOffsets66(PosInMtrx(0,0,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov00"]
    na_trk_cov[SymOffsets66(PosInMtrx(0,1,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov01"]
    na_trk_cov[SymOffsets66(PosInMtrx(0,2,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov02"]
    na_trk_cov[SymOffsets66(PosInMtrx(0,3,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov03"]
    na_trk_cov[SymOffsets66(PosInMtrx(0,4,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov04"]
    na_trk_cov[SymOffsets66(PosInMtrx(0,5,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov05"]
    na_trk_cov[SymOffsets66(PosInMtrx(1,1,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov11"]
    na_trk_cov[SymOffsets66(PosInMtrx(1,2,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov12"]
    na_trk_cov[SymOffsets66(PosInMtrx(1,3,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov13"]
    na_trk_cov[SymOffsets66(PosInMtrx(1,4,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov14"]
    na_trk_cov[SymOffsets66(PosInMtrx(1,5,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov15"]
    na_trk_cov[SymOffsets66(PosInMtrx(2,2,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov22"]
    na_trk_cov[SymOffsets66(PosInMtrx(2,3,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov23"]
    na_trk_cov[SymOffsets66(PosInMtrx(2,4,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov24"]
    na_trk_cov[SymOffsets66(PosInMtrx(2,5,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov25"]
    na_trk_cov[SymOffsets66(PosInMtrx(3,3,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov33"]
    na_trk_cov[SymOffsets66(PosInMtrx(3,4,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov34"]
    na_trk_cov[SymOffsets66(PosInMtrx(3,5,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov35"]
    na_trk_cov[SymOffsets66(PosInMtrx(4,4,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov44"]
    na_trk_cov[SymOffsets66(PosInMtrx(4,5,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov45"]
    na_trk_cov[SymOffsets66(PosInMtrx(5,5,6))] = np.random.normal(1,smear,(nevts,ntrks))*inputtrk["trk_cov55"]
    trk = {'trk_x': na_trk_x, 'trk_y': na_trk_y, 'trk_z': na_trk_z,
           'trk_ipt': na_trk_ipt, 'trk_theta': na_trk_theta, 'trk_phi': na_trk_phi,
           'trk_cov': na_trk_cov,'trk_q': na_trk_q
    }
    return trk

def prepareHits():
    na_hit_x = np.random.normal(1,smear,(nevts,ntrks))*inputhit["hit_x"]
    na_hit_y = np.random.normal(1,smear,(nevts,ntrks))*inputhit["hit_y"]
    na_hit_z = np.random.normal(1,smear,(nevts,ntrks))*inputhit["hit_z"]
    na_hit_cov = MatriplNP([None] * 6)
    na_hit_cov[SymOffsets33(PosInMtrx(0,0,3))] = np.random.normal(1,smear,(nevts,ntrks))*inputhit["hit_xx"]
    na_hit_cov[SymOffsets33(PosInMtrx(1,0,3))] = np.random.normal(1,smear,(nevts,ntrks))*inputhit["hit_xy"]
    na_hit_cov[SymOffsets33(PosInMtrx(1,1,3))] = np.random.normal(1,smear,(nevts,ntrks))*inputhit["hit_yy"]
    na_hit_cov[SymOffsets33(PosInMtrx(1,2,3))] = np.random.normal(1,smear,(nevts,ntrks))*inputhit["hit_yz"]
    na_hit_cov[SymOffsets33(PosInMtrx(2,0,3))] = np.random.normal(1,smear,(nevts,ntrks))*inputhit["hit_zx"]
    na_hit_cov[SymOffsets33(PosInMtrx(2,2,3))] = np.random.normal(1,smear,(nevts,ntrks))*inputhit["hit_zz"]
    hit = { 'hit_x': na_hit_x,'hit_y': na_hit_y,'hit_z': na_hit_z, 'hit_cov': na_hit_cov }
    return hit

def MultHelixPropEndcap(a, b, c):
    c[ 0] = b[ 0] + a[ 2]*b[ 3] + a[ 3]*b[ 6] + a[ 4]*b[10] + a[ 5]*b[15]
    c[ 1] = b[ 1] + a[ 2]*b[ 4] + a[ 3]*b[ 7] + a[ 4]*b[11] + a[ 5]*b[16]
    c[ 2] = b[ 3] + a[ 2]*b[ 5] + a[ 3]*b[ 8] + a[ 4]*b[12] + a[ 5]*b[17]
    c[ 3] = b[ 6] + a[ 2]*b[ 8] + a[ 3]*b[ 9] + a[ 4]*b[13] + a[ 5]*b[18]
    c[ 4] = b[10] + a[ 2]*b[12] + a[ 3]*b[13] + a[ 4]*b[14] + a[ 5]*b[19]
    c[ 5] = b[15] + a[ 2]*b[17] + a[ 3]*b[18] + a[ 4]*b[19] + a[ 5]*b[20]
    c[ 6] = b[ 1] + a[ 8]*b[ 3] + a[ 9]*b[ 6] + a[10]*b[10] + a[11]*b[15]
    c[ 7] = b[ 2] + a[ 8]*b[ 4] + a[ 9]*b[ 7] + a[10]*b[11] + a[11]*b[16]
    c[ 8] = b[ 4] + a[ 8]*b[ 5] + a[ 9]*b[ 8] + a[10]*b[12] + a[11]*b[17]
    c[ 9] = b[ 7] + a[ 8]*b[ 8] + a[ 9]*b[ 9] + a[10]*b[13] + a[11]*b[18]
    c[10] = b[11] + a[ 8]*b[12] + a[ 9]*b[13] + a[10]*b[14] + a[11]*b[19]
    c[11] = b[16] + a[ 8]*b[17] + a[ 9]*b[18] + a[10]*b[19] + a[11]*b[20]
    c[12] = np.zeros_like(b[0])
    c[13] = np.zeros_like(b[0])
    c[14] = np.zeros_like(b[0])
    c[15] = np.zeros_like(b[0])
    c[16] = np.zeros_like(b[0])
    c[17] = np.zeros_like(b[0])
    c[18] = b[ 6]
    c[19] = b[ 7]
    c[20] = b[ 8]
    c[21] = b[ 9]
    c[22] = b[13]
    c[23] = b[18]
    c[24] = a[26]*b[ 3] + a[27]*b[ 6] + b[10] + a[29]*b[15]
    c[25] = a[26]*b[ 4] + a[27]*b[ 7] + b[11] + a[29]*b[16]
    c[26] = a[26]*b[ 5] + a[27]*b[ 8] + b[12] + a[29]*b[17]
    c[27] = a[26]*b[ 8] + a[27]*b[ 9] + b[13] + a[29]*b[18]
    c[28] = a[26]*b[12] + a[27]*b[13] + b[14] + a[29]*b[19]
    c[29] = a[26]*b[17] + a[27]*b[18] + b[19] + a[29]*b[20]
    c[30] = b[15]
    c[31] = b[16]
    c[32] = b[17]
    c[33] = b[18]
    c[34] = b[19]
    c[35] = b[20]
    
def MultHelixPropTranspEndcap(a, b, c):
    c[ 0] = b[ 0] + b[ 2]*a[ 2] + b[ 3]*a[ 3] + b[ 4]*a[ 4] + b[ 5]*a[ 5]
    c[ 1] = b[ 6] + b[ 8]*a[ 2] + b[ 9]*a[ 3] + b[10]*a[ 4] + b[11]*a[ 5]
    c[ 2] = b[ 7] + b[ 8]*a[ 8] + b[ 9]*a[ 9] + b[10]*a[10] + b[11]*a[11]
    c[ 3] = b[12] + b[14]*a[ 2] + b[15]*a[ 3] + b[16]*a[ 4] + b[17]*a[ 5]
    c[ 4] = b[13] + b[14]*a[ 8] + b[15]*a[ 9] + b[16]*a[10] + b[17]*a[11]
    c[ 5] = np.zeros_like(b[0])
    c[ 6] = b[18] + b[20]*a[ 2] + b[21]*a[ 3] + b[22]*a[ 4] + b[23]*a[ 5]
    c[ 7] = b[19] + b[20]*a[ 8] + b[21]*a[ 9] + b[22]*a[10] + b[23]*a[11]
    c[ 8] = np.zeros_like(b[0])
    c[ 9] = b[21]
    c[10] = b[24] + b[26]*a[ 2] + b[27]*a[ 3] + b[28]*a[ 4] + b[29]*a[ 5]
    c[11] = b[25] + b[26]*a[ 8] + b[27]*a[ 9] + b[28]*a[10] + b[29]*a[11]
    c[12] = np.zeros_like(b[0])
    c[13] = b[27]
    c[14] = b[26]*a[26] + b[27]*a[27] + b[28] + b[29]*a[29]
    c[15] = b[30] + b[32]*a[ 2] + b[33]*a[ 3] + b[34]*a[ 4] + b[35]*a[ 5]
    c[16] = b[31] + b[32]*a[ 8] + b[33]*a[ 9] + b[34]*a[10] + b[35]*a[11]
    c[17] = np.zeros_like(b[0])
    c[18] = b[33]
    c[19] = b[32]*a[26] + b[33]*a[27] + b[34] + b[35]*a[29]
    c[20] = b[35]

def helixAtZ(tracks, zout, outtracks):
    k = tracks["trk_q"]*100/3.8
    deltaZ = zout - tracks["trk_z"]
    pt = 1./tracks["trk_ipt"]
    cosP = np.cos(tracks["trk_phi"])
    sinP = np.sin(tracks["trk_phi"])
    cosT = np.cos(tracks["trk_theta"])
    sinT = np.sin(tracks["trk_theta"])
    pxin = cosP*pt
    pyin = sinP*pt
    alpha  = deltaZ*sinT*tracks["trk_ipt"]/(cosT*k)
    sina = np.sin(alpha) # this can be approximated
    cosa = np.cos(alpha) # this can be approximated
    outtracks["trk_x"] = tracks["trk_x"] + k*(pxin*sina - pyin*(1.-cosa))
    outtracks["trk_y"] = tracks["trk_y"] + k*(pyin*sina + pxin*(1.-cosa))
    outtracks["trk_z"] = zout
    outtracks["trk_ipt"] = tracks["trk_ipt"]
    outtracks["trk_phi"] = tracks["trk_phi"]+alpha
    outtracks["trk_theta"] = tracks["trk_theta"]
    
    sCosPsina = np.sin(cosP*sina)
    cCosPsina = np.cos(cosP*sina)

    errorProp = MatriplNP([None]*36)
    for i in range(0,6): errorProp[PosInMtrx(i,i,6)] = np.ones_like(tracks["trk_x"])
    errorProp[PosInMtrx(0,2,6)] = cosP*sinT*(sinP*cosa*sCosPsina - cosa)/cosT
    errorProp[PosInMtrx(0,3,6)] = cosP*sinT*deltaZ*cosa*( 1. - sinP*sCosPsina )/(cosT*tracks["trk_ipt"]) - k*(cosP*sina - sinP*(1.-cCosPsina))/(tracks["trk_ipt"]*tracks["trk_ipt"])
    errorProp[PosInMtrx(0,4,6)] = (k/tracks["trk_ipt"])*( -sinP*sina + sinP*sinP*sina*sCosPsina - cosP*(1. - cCosPsina ) )
    errorProp[PosInMtrx(0,5,6)] = cosP*deltaZ*cosa*( 1. - sinP*sCosPsina )/(cosT*cosT)
    errorProp[PosInMtrx(1,2,6)] = cosa*sinT*(cosP*cosP*sCosPsina - sinP)/cosT
    errorProp[PosInMtrx(1,3,6)] = sinT*deltaZ*cosa*( cosP*cosP*sCosPsina + sinP )/(cosT*tracks["trk_ipt"]) - k*(sinP*sina + cosP*(1.-cCosPsina))/(tracks["trk_ipt"]*tracks["trk_ipt"])
    errorProp[PosInMtrx(1,4,6)] = (k/tracks["trk_ipt"])*( -sinP*(1. - cCosPsina) - sinP*cosP*sina*sCosPsina + cosP*sina )
    errorProp[PosInMtrx(1,5,6)] = deltaZ*cosa*( cosP*cosP*sCosPsina + sinP )/(cosT*cosT)
    errorProp[PosInMtrx(4,2,6)] = -tracks["trk_ipt"]*sinT/(cosT*k)
    errorProp[PosInMtrx(4,3,6)] = sinT*deltaZ/(cosT*k)
    errorProp[PosInMtrx(4,5,6)] = tracks["trk_ipt"]*deltaZ/(cosT*cosT*k)
         
    temp = MatriplNP([None]*36)
    MultHelixPropEndcap(errorProp, tracks["trk_cov"], temp)
    MultHelixPropTranspEndcap(errorProp, temp, tracks["trk_cov"])


nevts = 1000
ntrks = 9600
smear = 0.1

print('track in pos: ', inputtrk["trk_x"], inputtrk["trk_y"], inputtrk["trk_z"])
print('track in cov: ', inputtrk["trk_cov00"], inputtrk["trk_cov11"], inputtrk["trk_cov22"])
print('hit in pos: ', inputhit["hit_x"], inputhit["hit_y"], inputhit["hit_z"])

print('produce nevts=', nevts, ' ntrks=', ntrks, ' smearing by=', smear)

trk = prepareTracks()
hit = prepareHits()

#print 'track in pos: ', trkflt["trk_x"], trkflt["trk_y"], trkflt["trk_z"]
#print 'track in cov: ', trkflt["trk_cov00"], trkflt["trk_cov11"], trkflt["trk_cov22"]
#print 'hit: ', hitflt["hit_x"], hitflt["hit_y"], hitflt["hit_z"]

outtrk = trk.copy()
t0 = time.time()
helixAtZ(trk,hit["hit_z"],outtrk)
t1 = time.time()

#print 'track out pos: ', outtrk["trk_x"], outtrk["trk_y"], outtrk["trk_z"]
#print 'track out cov: ', outtrk["trk_cov00"], outtrk["trk_cov11"], outtrk["trk_cov22"]
print('done ntracks=', nevts*ntrks, 'tot time=', t1-t0, 'time/trk=', (t1-t0)/(nevts*ntrks))

print('track x avg=', outtrk["trk_x"].mean(), ' std/avg=', np.abs(outtrk["trk_x"].std()/outtrk["trk_x"].mean()))
print('track y avg=', outtrk["trk_y"].mean(), ' std/avg=', np.abs(outtrk["trk_y"].std()/outtrk["trk_y"].mean()))
print('track z avg=', outtrk["trk_z"].mean(), ' std/avg=', np.abs(outtrk["trk_z"].std()/outtrk["trk_z"].mean()))

print('(trk_x-hit_x)/trk_x avg=', np.abs(((outtrk["trk_x"]-hit["hit_x"])/outtrk["trk_x"]).mean()), ' std=', ((outtrk["trk_x"]-hit["hit_x"])/outtrk["trk_x"]).std())
print('(trk_y-hit_y)/trk_y avg=', np.abs(((outtrk["trk_y"]-hit["hit_y"])/outtrk["trk_y"]).mean()), ' std=', ((outtrk["trk_y"]-hit["hit_y"])/outtrk["trk_y"]).std())
print('(trk_z-hit_z)/trk_z avg=', np.abs(((outtrk["trk_z"]-hit["hit_z"])/outtrk["trk_z"]).mean()), ' std=', ((outtrk["trk_z"]-hit["hit_z"])/outtrk["trk_z"]).std())
