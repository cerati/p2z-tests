import numpy as np
from scipy import loadtxt
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

events = []
z = open('compare_output.txt')
for line in z.readlines():
  col = line.rstrip().split(' ')
#  events.append({"fullname": col[0], "compiler": col[1], "mode": col[2], "i":int(col[3]), "events":int(col[4]), "compute":float(col[5]), "computepertrk":float(col[6]), "region":float(col[7]),"regionpertrk":float(col[7])/float(col[4]), "mem":float(col[8]),"mempertrk":float(col[8])/float(col[4]), "setup":float(col[9]),"setuppertrk":float(col[9])/float(col[4])})
  events.append({"fullname": col[0], "compiler": col[1], "mode": col[2], "i":int(col[3]),"iter":int(col[4]), "events":int(col[5]),"tracks":int(col[6]),"bsize":int(col[7]),"nb":int(col[8]), "compute":float(col[9]), "region":float(col[10]), "mem":float(col[11]), "setup":float(col[12]),"threads":int(col[13])})




#comp = .001199188
#mem = .1938896 #+ .1170012
#setup = 12.351428+1.673311
#kokkos_gpu = {'fullname': 'Kokkos(nvcc)','compiler':'nvcc','i':0,'events':960000,'compute': comp, 'computepertrk': comp/960000,'region': comp+mem,'regionpertrk':(comp+mem)/960000,'mem':mem,'mempertrk':mem/960000,'setup':setup,'setuppertrk':(setup)/960000}
#mem = 0
#setup = 0
#comp = 1.45* 0.000001*960000
#kokkos_cpu1 = {'fullname': 'Kokkos(gcc)','compiler':'gcc','i':0,'events':960000,'compute': comp, 'computepertrk': comp/960000,'region': comp+mem,'regionpertrk':(comp+mem)/960000,'mem':mem,'mempertrk':mem/960000,'setup':setup,'setuppertrk':(setup)/960000}
#comp = 5.78* 0.0000001*960000
#kokkos_cpu2 = {'fullname': 'Kokkos(icc)','compiler':'icc','i':0,'events':960000,'compute': comp, 'computepertrk': comp/960000,'region': comp+mem,'regionpertrk':(comp+mem)/960000,'mem':mem,'mempertrk':mem/960000,'setup':setup,'setuppertrk':(setup)/960000}
#comp = 2.40* 0.000001*960000
#mkl_1 = {'fullname': 'MKL(icc)','compiler':'icc','i':0,'events':960000,'compute': comp, 'computepertrk': comp/960000,'region': comp+mem,'regionpertrk':(comp+mem)/960000,'mem':mem,'mempertrk':mem/960000,'setup':setup,'setuppertrk':(setup)/960000}
#
#events.append(kokkos_cpu1)
#events.append(kokkos_cpu2)
#events.append(kokkos_gpu)
#events.append(mkl_1)
df = pd.DataFrame(events)
df['gpu'] = (df['compiler'] == 'nvcc') | (df['mode']== 'acc')
dfx = df.groupby(['fullname','events','gpu']).mean()
dfx = dfx.reset_index()
dfx['regionthroughput']  = (dfx['iter']*dfx['tracks']*dfx['events'])/dfx['region']
dfx['computethroughput'] = (dfx['iter']*dfx['tracks']*dfx['events'])/dfx['compute']
dfx['memthoughtput']     = (dfx['iter']*dfx['tracks']*dfx['events'])/dfx['mem']
dfx = dfx.iloc[dfx['gpu'].argsort()]
#event2 = [kokkos_cpu,kokkos_gpu]
#df2 = pd.DataFrame(event2)
#dfx.append(df2)
#print(df2)
print(dfx)
#df2 = pd.DataFrame(events)
#df2["threadtot"]=df2['threadx']*df2['thready']
#dfx2 = df2.groupby(['fullname','stream','block','threadtot','events']).mean()
#dfx2 = dfx2.reset_index()
#print(df)
#print(dfx)

#import seaborn as sns
#sub = dfx[dfx['threadx']==1]
#test = dfx['thready'].unique()
def getsub(dfx,n1,n3,n4,xname,yname,xscale,ver):
  pp = PdfPages('throughput_%s_vs_%s_by_%s_v%i.pdf'%(yname,xname,n4,ver))
  for x1 in dfx[n1].unique():
    df1 = dfx[dfx[n1]==x1]
    for x2 in df1[n1].unique():
      df2 = df1[df1[n1]==x2]
      for x3 in df2[n3].unique():
        df3 = df2[df2[n3]==x3]
        fig, ax = plt.subplots()
        count = 0
        for x4 in df3[n4].unique():
          count = count+1
          if (ver ==0 and count >0.5*len(df3[n4].unique())) or (ver ==1 and count <=0.5*len(df3[n4].unique())):
            continue
          df4 = df3[df3[n4]==x4]
          print(df4)
          df4.plot(x=xname,y=yname,style='.-', ax=ax,label=x4)
        #print(dfx['thready'].unique())
        ax.set_title("%s;%s=%s"%(x1,n3,x3))
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        ax.set_xscale(xscale)
        ax.set_yscale('log')
        ax.set_xticks(dfx[xname].unique())
        ax.set_xticklabels(dfx[xname].unique())
        ax.legend(title=n4,loc='center right')
#        plt.ticklabel_format(axis='y',style="sci",scilimits(0,3))
        #plt.show()
        pp.savefig()
          #for x5 in df4[n5].unique():
          #  df5 = df4[df4[n5]==x5]
  pp.close()    

#getsub(dfx,'fullname','stream','thready','threadx','block','regionpertrk','linear')
#getsub(dfx,'fullname','stream','thready','threadx','block','computepertrk','linear')
#getsub(dfx,'fullname','stream','thready','threadx','block','mempertrk','linear')
#getsub(dfx,'fullname','stream','block','threadx','regionpertrk','log',0)
#getsub(dfx,'fullname','stream','block','threadx','computepertrk','log',0)
#getsub(dfx,'fullname','stream','block','threadx','mempertrk','log',0)
#getsub(dfx,'fullname','threadx','stream','block','regionpertrk','linear',0)
#getsub(dfx,'fullname','threadx','stream','block','computepertrk','linear',0)
#getsub(dfx,'fullname','threadx','stream','block','mempertrk','linear',0)
#getsub(dfx,'fullname','block','threadx','stream','regionpertrk','linear',0)
#getsub(dfx,'fullname','block','threadx','stream','computepertrk','linear',0)
#getsub(dfx,'fullname','block','threadx','stream','mempertrk','linear',0)
#
#getsub(dfx,'fullname','stream','block','threadx','regionpertrk','log',1)
#getsub(dfx,'fullname','stream','block','threadx','computepertrk','log',1)
#getsub(dfx,'fullname','stream','block','threadx','mempertrk','log',1)
#getsub(dfx,'fullname','threadx','stream','block','regionpertrk','linear',1)
#getsub(dfx,'fullname','threadx','stream','block','computepertrk','linear',1)
#getsub(dfx,'fullname','threadx','stream','block','mempertrk','linear',1)
#getsub(dfx,'fullname','block','threadx','stream','regionpertrk','linear',1)
#getsub(dfx,'fullname','block','threadx','stream','computepertrk','linear',1)
#getsub(dfx,'fullname','block','threadx','stream','mempertrk','linear',1)

#getsub(dfx,'fullname','threadx','block','stream','regionpertrk','linear',0)
#getsub(dfx,'fullname','threadx','block','stream','computepertrk','linear',0)
#getsub(dfx,'fullname','threadx','block','stream','mempertrk','linear',0)
#getsub(dfx,'fullname','threadx','block','stream','regionpertrk','linear',1)
#getsub(dfx,'fullname','threadx','block','stream','computepertrk','linear',1)
#getsub(dfx,'fullname','threadx','block','stream','mempertrk','linear',1)
def getbar(dfx,yscale):
  #for name in dfx.fullname.unique():
   # df2 = dfx[dfx['fullname'] == name]
    #for bsize in dfx.bsize.unique():
      #df3 = df2[df2['bsize'] == bsize]
  #ax = dfx.plot.bar(y='computepertrk',x='fullname')
  #ax = dfx[['fullname','computepertrk','mempertrk']].plot.bar(x='fullname',stacked=True)
  ax = dfx[['fullname','regionthroughput']].plot.bar(x='fullname',stacked=True)
  ax.set_ylabel('computation throughput (trks/s)')
  ax.set_title("computation throughput by implementation")
  ax.set_yscale(yscale)
  ax.tick_params(axis="x", labelsize=15)
  ax.tick_params(axis="y", labelsize=15)
  plt.setp(ax.xaxis.get_majorticklabels(),rotation=-45)
  for p in ax.patches:
    if p.get_height() == 0: continue
    ax.annotate("%.4e"%p.get_height(),(p.get_x()*1.005,p.get_height()*1.005))
  plt.ticklabel_format(axis='y',style="sci",scilimits=(0,3))
  plt.show()
  #plt.savefig("comparisons.pdf")
#def getbarr(dfx,yscale):
#  #for name in dfx.fullname.unique():
#   # df2 = dfx[dfx['fullname'] == name]
#    #for bsize in dfx.bsize.unique():
#      #df3 = df2[df2['bsize'] == bsize]
#  #ax = dfx.plot.bar(y='computepertrk',x='fullname')
#  ax = dfx.plot.bar(y='regionpertrk',x='fullname')
#  ax.set_ylabel('region time per track')
#  ax.set_title("region time")
#  ax.set_yscale(yscale)
#  plt.setp(ax.xaxis.get_majorticklabels(),rotation=-45)
#  for p in ax.patches:
#    ax.annotate("%.4e"%p.get_height(),(p.get_x()*1.005,p.get_height()*1.005))
#  plt.ticklabel_format(axis='y',style="sci",scilimits=(0,3))
#  #plt.savefig("comparisons.png")

getbar(dfx,'linear')
#getbarr(dfx,'linear')
#getbar(dfx,'log')
#getbarr(dfx,'log')
#print(dfx.nsmallest(30,'computepertrk'))









