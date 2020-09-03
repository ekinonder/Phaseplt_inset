%matplotlib notebook
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.animation import ArtistAnimation
import numpy as np
from numpy import diff
from scipy.integrate import odeint
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

plt.style.use('classic')
#f=open("circuit2.txt","r")
f=open('../circuit57_0.txt','r')

data=np.loadtxt(f)
t=data[:,0]
x=data[:,1]
trans=500000


x1=x[trans:]
t1=t[trans:]
n11=750000
n22=800000
#plt.show()
x=x[x>10.]

n1=1000
n2=len(x)
x=x[-50000:]
t=t[-50000:]
x=x[range(0,len(x),5)]
t=t[range(0,len(t),5)]
dxdt=np.diff(x)/np.diff(t)
dxdt1=np.diff(x1)/np.diff(t1)
ddxt=np.diff(dxdt)/np.diff(t[0:-1])
#ax.plot(x[0:len(x)-1],dxdt)
fig,ax=plt.subplots()
#fig5=plt.figure()
#ax5=plt.axes(projection='3d')
#ax5.plot3D(x[0:-2],dxdt[0:-1],ddxt)

x=x[0:-1]
y=dxdt
x1=x1[0:-1]
ax.plot(x1,dxdt1,alpha=0.2)

ax.set_xlim([9,13.5])
ax.set_ylim([-0.15,0.175])


n1=554600-trans
n2=554815-trans
n3=554830-trans
n4=554908-trans
n5=554990-trans
n6=555020-trans
n7=555085-trans
n8=555220-trans
n9=555285-trans
n10=555380-trans


###Second path####3
v1=555600-trans
v2=556010-trans
v3=556530-trans
v4=556920-trans
v5=556985-trans
v6=557015-trans
v7=557100-trans
v8=557150-trans
v9=557285-trans
v10=557580-trans
###########################33

marker=20
markerx=13

xpath=x1[554600-trans:557800-trans]
dxdt1path=dxdt1[554600-trans:557800-trans]
ax.plot(xpath,dxdt1path,"b")



ax.plot(x1[n1],dxdt1[n1],'r+', markersize=marker, markeredgewidth=2)
ax.plot(x1[n2],dxdt1[n2],'r+', markersize=marker, markeredgewidth=2)
ax.plot(x1[n3],dxdt1[n3],'g+', markersize=marker, markeredgewidth=2)
ax.plot(x1[n4],dxdt1[n4],'g+', markersize=marker, markeredgewidth=2)
ax.plot(x1[n5],dxdt1[n5],'c+', markersize=marker, markeredgewidth=2)
ax.plot(x1[n6],dxdt1[n6],'c+', markersize=marker, markeredgewidth=2)
ax.plot(x1[n7],dxdt1[n7],'m+', markersize=marker, markeredgewidth=2)
ax.plot(x1[n8],dxdt1[n8],'m+', markersize=marker, markeredgewidth=2)
ax.plot(x1[n9],dxdt1[n9],'y+', markersize=marker, markeredgewidth=2)
ax.plot(x1[n10],dxdt1[n10],'y+', markersize=marker, markeredgewidth=2)
#ax.plot(x1[n11],dxdt1[n11],'ok')
#ax.plot(x1[n12],dxdt1[n12],'ok')

ax.text(9.78,0.012,"u1",color="r",fontsize=11)
ax.text(10.24,0.03477,"u2",color="r",fontsize=11)
ax.text(10.84,0.04,"u3",color="g",fontsize=11)
ax.text(11.0,0.012,"u4",color="g",fontsize=11)
ax.text(11.54,0.041,"u5",color="c",fontsize=11)
ax.text(13.05,0.026,"u6",color="c",fontsize=11)
ax.text(11.2569,0.0225,"u7",color="m",fontsize=11)
ax.text(11.07,-0.026,"u8",color="m",fontsize=11)
ax.text(10.7917,-0.052,"u9",color="y",fontsize=11)
ax.text(10.073,-0.017,"u10",color="y",fontsize=11)
#ax.text(10.71,-0.049,"11")
#ax.text(9.97,0.007,"12")
ly=np.linspace(-0.15,0.15,250)
#l1x=9.899*np.ones(len(ly))
#l2x=10.314*np.ones(len(ly))
#l3x=10.8127*np.ones(len(ly))
#l4x=11.1791*np.ones(len(ly))
#l5x=11.5487*np.ones(len(ly))
#l6x=13.0002*np.ones(len(ly))
#l7x=11.3297*np.ones(len(ly))
#l8x=11.1697*np.ones(len(ly))
#l9x=10.6917*np.ones(len(ly))
#l10x=10.0163*np.ones(len(ly))
#l11x=9.899*np.ones(len(ly))
#l12x=9.899*np.ones(len(ly))
#ax.plot(l1x,ly,"r",alpha=0.1)
#ax.plot(l2x,ly,"r",alpha=0.1)
#ax.plot(l3x,ly,"g",alpha=0.1)
#ax.plot(l4x,ly,"g",alpha=0.1)
#ax.plot(l5x,ly,"c",alpha=0.1)
#ax.plot(l6x,ly,"c",alpha=0.1)
#ax.plot(l7x,ly,"m",alpha=0.1)
#ax.plot(l8x,ly,"m",alpha=0.1)
#ax.plot(l9x,ly,"y",alpha=0.1)
#ax.plot(l10x,ly,"y",alpha=0.1)



####ARROW PLOTS####3
xarw=xpath
arwstep=50
yarw=dxdt1path
xarw0 = xarw[range(len(xarw)-1)]
xarw1 = xarw[range(1,len(xarw))]
yarw0 = yarw[range(len(yarw)-1)]
yarw1 = yarw[range(1,len(yarw))]
xpos = (xarw0[range(0,len(xarw0),arwstep)]+xarw1[range(0,len(xarw1),arwstep)])/2
ypos = (yarw0[range(0,len(yarw0),arwstep)]+yarw1[range(0,len(yarw1),arwstep)])/2
xdir = xarw1[range(0,len(xarw1),arwstep)]-xarw0[range(0,len(xarw0),arwstep)]
ydir = yarw1[range(0,len(yarw1),arwstep)]-yarw0[range(0,len(yarw0),arwstep)]
#ax.scatter(xarw,yarw)
#ax.plot(xarw,yarw)
# plot arrow on each line:
for X,Y,dX,dY in zip(xpos, ypos, xdir, ydir):
    ax.annotate("", xytext=(X,Y),xy=(X+0.1*dX,Y+0.1*dY), 
    arrowprops=dict(arrowstyle="->", color='k'), size = 20)





#####INSET PLOT######
axins = zoomed_inset_axes(ax, 5, loc=3) # zoom-factor: 2.5, location: upper-left
axins.plot(xpath, dxdt1path,"b")
axins.plot(x1[n1],dxdt1[n1],'r+', markersize=marker, markeredgewidth=2)
axins.plot(x1[n10],dxdt1[n10],'y+', markersize=marker, markeredgewidth=2)
axins.plot(x1[v1],dxdt1[v1],'rx', markersize=markerx, markeredgewidth=2)
axins.plot(x1[v2],dxdt1[v2],'rx', markersize=markerx, markeredgewidth=2)
axins.text(10.13,-0.00017,"v2",color="r",fontsize=11)
axins.text(10.01,-0.0058,"u10",color="y",fontsize=11)
axins.text(9.93,-0.0058,"v1",color="r",fontsize=11)
axins.text(9.89,-0.0058,"u1",color="r",fontsize=11)
axins.plot(x1[v3],dxdt1[v3],'gx', markersize=markerx, markeredgewidth=2)
axins.text(9.88,0.0040,"v3",color="g",fontsize=11)




xarwin=xpath
arwstepin=50
yarwin=dxdt1path
xarw0in = xarwin[range(len(xarwin)-1)]
xarw1in = xarwin[range(1,len(xarwin))]
yarw0in = yarwin[range(len(yarwin)-1)]
yarw1in = yarwin[range(1,len(yarwin))]
xposin = (xarw0in[range(0,len(xarw0in),arwstepin)]+xarw1in[range(0,len(xarw1in),arwstepin)])/2
yposin = (yarw0in[range(0,len(yarw0in),arwstepin)]+yarw1in[range(0,len(yarw1in),arwstepin)])/2
xdirin = xarw1in[range(0,len(xarw1in),arwstepin)]-xarw0in[range(0,len(xarw0in),arwstepin)]
ydirin = yarw1in[range(0,len(yarw1in),arwstepin)]-yarw0in[range(0,len(yarw0in),arwstepin)]

for X,Y,dX,dY in zip(xposin, yposin, xdirin, ydirin):
    axins.annotate("", xytext=(X,Y),xy=(X+0.1*dX,Y+0.1*dY), 
    arrowprops=dict(arrowstyle="->", color='k'), size = 10)

xloc1, xloc2, yloc1, yloc2 = 9.85, 10.20, -0.008, 0.0081 # specify the limits
axins.set_xlim(xloc1, xloc2) # apply the x-limits
axins.set_ylim(yloc1, yloc2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins, loc1=2, loc2=1, fc="none", ec="0.5")


#####SADDLE INSET PLOT#####
axins1 = zoomed_inset_axes(ax, 2.5, loc="upper center") # zoom-factor: 2.5, location: upper-left
axins1.plot(xpath, dxdt1path,"b")
axins1.plot(x1[n4],dxdt1[n4],'g+', markersize=marker, markeredgewidth=2)
axins1.plot(x1[n7],dxdt1[n7],'m+', markersize=marker, markeredgewidth=2)
axins1.plot(x1[n8],dxdt1[n8],'m+', markersize=marker, markeredgewidth=2)
axins1.text(11.1395,-0.0096,"u8",color="m",fontsize=11)
axins1.text(11.2937,0.013,"u7",color="m",fontsize=11)
axins1.text(11.12,0.005,"u4",color="g",fontsize=11)
axins1.plot(x1[v4],dxdt1[v4],'gx', markersize=markerx, markeredgewidth=2)
axins1.text(11.23,-0.009,"v4",color="g",fontsize=11)
axins1.plot(x1[v7],dxdt1[v7],'mx', markersize=markerx, markeredgewidth=2)
axins1.text(11.32,-0.009,"v7",color="m",fontsize=11)

xarwin=xpath
arwstepin=20
yarwin=dxdt1path
xarw0in = xarwin[range(len(xarwin)-1)]
xarw1in = xarwin[range(1,len(xarwin))]
yarw0in = yarwin[range(len(yarwin)-1)]
yarw1in = yarwin[range(1,len(yarwin))]
xposin = (xarw0in[range(0,len(xarw0in),arwstepin)]+xarw1in[range(0,len(xarw1in),arwstepin)])/2
yposin = (yarw0in[range(0,len(yarw0in),arwstepin)]+yarw1in[range(0,len(yarw1in),arwstepin)])/2
xdirin = xarw1in[range(0,len(xarw1in),arwstepin)]-xarw0in[range(0,len(xarw0in),arwstepin)]
ydirin = yarw1in[range(0,len(yarw1in),arwstepin)]-yarw0in[range(0,len(yarw0in),arwstepin)]

for X,Y,dX,dY in zip(xposin, yposin, xdirin, ydirin):
    axins1.annotate("", xytext=(X,Y),xy=(X+0.1*dX,Y+0.1*dY), 
    arrowprops=dict(arrowstyle="->", color='k'), size = 10)

x1loc1, x1loc2, y1loc1, y1loc2 = 11.02, 11.42, -0.0171, 0.018 # specify the limits
axins1.set_xlim(x1loc1, x1loc2) # apply the x-limits
axins1.set_ylim(y1loc1, y1loc2) # apply the y-limits
plt.yticks(visible=False)
plt.xticks(visible=False)
mark_inset(ax, axins1, loc1=3, loc2=4, fc="none", ec="0.5")

plt.show()


###########################SECOND PATH PLOTS#############################


ax.plot(x1[v1],dxdt1[v1],'rx', markersize=markerx, markeredgewidth=2)
ax.plot(x1[v2],dxdt1[v2],'rx', markersize=markerx, markeredgewidth=2)
ax.plot(x1[v3],dxdt1[v3],'gx', markersize=markerx, markeredgewidth=2)
ax.plot(x1[v4],dxdt1[v4],'gx', markersize=markerx, markeredgewidth=2)
ax.plot(x1[v5],dxdt1[v5],'cx', markersize=markerx, markeredgewidth=2)
ax.plot(x1[v6],dxdt1[v6],'cx', markersize=markerx, markeredgewidth=2)
ax.plot(x1[v7],dxdt1[v7],'mx', markersize=markerx, markeredgewidth=2)
ax.plot(x1[v8],dxdt1[v8],'mx', markersize=markerx, markeredgewidth=2)
#ax.plot(x1[v9],dxdt1[v9],'yx', markersize=markerx, markeredgewidth=2)
#ax.plot(x1[v10],dxdt1[v10],'yx', markersize=markerx, markeredgewidth=2)




ax.text(9.89281,-0.0199,"v1",color="r",fontsize=11)
ax.text(10.1,0.017,"v2",color="r",fontsize=11)
ax.text(9.7676,-0.018,"v3",color="g",fontsize=11)
ax.text(11.15,0.0116,"v4",color="g",fontsize=11)
ax.text(11.738,0.02714,"v5",color="c",fontsize=11)
ax.text(13.105,0.00223,"v6",color="c",fontsize=11)
ax.text(11.42,-0.0165,"v7",color="m",fontsize=11)
#ax.text(11.07,-0.026,"u8",color="m",fontsize=11)
#ax.text(10.7917,-0.052,"u9",color="y",fontsize=11)
#ax.text(10.103,-0.017,"u10",color="y",fontsize=11)
