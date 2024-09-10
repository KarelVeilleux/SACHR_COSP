import numpy       as     np
from   scipy.stats import pearsonr
NaN = np.nan

def spatial_correlation_coeffcient(DATA1, DATA2, samplingbox, alpha, ignore_nan='no'):

    JI=samplingbox[0]
    JF=samplingbox[1]+1
    II=samplingbox[2]
    IF=samplingbox[3]+1

    boxsize = (IF - II) * (JF - JI)
    

    NI = DATA1.shape[0]
    NJ = DATA1.shape[1]
    print(NI,NJ) 
    r = np.empty([NI, NJ])
    p = np.empty([NI, NJ])
    for I in range(NI):
        for J in range(NJ):
            data1 = np.empty([boxsize])
            data2 = np.empty([boxsize])
            k = 0
            for i in range(I+II, I+IF):
                for j in range(J+JI, J+JF):
                    if   i < 0 or i >= NI or j < 0 or j >= NJ:
                        data = NaN
                        data1[k] = NaN
                        data2[k] = NaN
                        print('a',I,J, DATA1[I,J],'%3d %3d    ' % (i,j) ,data)
                    else:
                        data = DATA1[i,j]
                        print('b',I,J, DATA1[I,J],'%3d %3d    ' % (i,j) ,data)
                        data1[k] = DATA1[i,j]
                        data2[k] = DATA2[i,j]
                    k = k + 1
            print(data1)
            print('')
            
            if np.isnan(np.sum(data1 + data2)):
                r[I,J] = NaN;
                p[I,J] = NaN;
            else:
                r[I,J], p[I,J] = pearsonr(data1, data2)

    return r, p




