
import numpy as np 
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import Utils
import matplotlib.pyplot as plt 
from sunpy.time import parse_time




# animation settings 

fsize=13
fadeind = 200*24 #if s/c positions are given in hourly resolution

symsize_planet=110
symsize_spacecraft=80

#for parker spiral   
theta=np.arange(0,np.deg2rad(180),0.01)
cme_color='#8C99FD'

def angle_to_coord_line(angle,x0,y0,x1,y1):
    #rotate by 4 deg for HI1 FOV
    ang=np.deg2rad(angle)
    rot=np.array([[np.cos(ang), -np.sin(ang)], 
                  [np.sin(ang), np.cos(ang)]
                  ])    
    [x2,y2]=np.dot(rot,[x1,y1])

    #add to sta position
    x2f=x0+x2
    y2f=y0+y2    
    
    return Utils.cart2sphere(x2f,y2f,0.0)    

def draw_punch_fov(pos, time_num, timeind,ax):
    
    lcolor='green'

    #sta position
    x0=pos.x[timeind]
    y0=pos.y[timeind]
    z0=0

    x1=-pos.x[timeind]
    y1=-pos.y[timeind]
    z1=0

    
    r2,t2,lon2=angle_to_coord_line(45,x0,y0,x1,y1)
    r3,t3,lon3=angle_to_coord_line(-45,x0,y0,x1,y1)


    r4,t4,lon4=angle_to_coord_line(1.5,x0,y0,x1,y1)
    r5,t5,lon5=angle_to_coord_line(-1.5,x0,y0,x1,y1)

    r6,t6,lon6=angle_to_coord_line(4.4,x0,y0,x1,y1)
    r7,t7,lon7=angle_to_coord_line(-4.4,x0,y0,x1,y1)

    r8,t8,lon8=angle_to_coord_line(7.4,x0,y0,x1,y1)
    r9,t9,lon9=angle_to_coord_line(-7.4,x0,y0,x1,y1)
    # r5,t5,lon5=angle_to_coord_line(ang4d,x0,y0,x1,y1)



    r0,t0,lon0 =Utils.cart2sphere(x0,y0,z0)   
    ax.plot([lon0,lon2],[r0,r2],linestyle='-',color=lcolor,alpha=0.5, lw=1.2)
    ax.plot([lon0,lon3],[r0,r3],linestyle='-',color=lcolor,alpha=0.5, lw=1.2)
    ax.plot([lon0,lon6],[r0,r6],linestyle='-',color=lcolor,alpha=0.5, lw=1.2)
    ax.plot([lon0,lon7],[r0,r7],linestyle='-',color=lcolor,alpha=0.5, lw=1.2,label="PUNCH WFI Field of View")

    # ax.plot([lon0,lon4],[r0,r4],linestyle='-',color=lcolor,alpha=0.45, lw=1)
    # ax.plot([lon0,lon5],[r0,r5],linestyle='-',color=lcolor,alpha=0.45, lw=1)
    # ax.plot([lon0,lon8],[r0,r8],linestyle='-',color=lcolor,alpha=0.45, lw=1)
    ax.fill([lon0,lon9,lon5],[r0,r9,r5],color=lcolor,alpha=0.3)
    ax.fill([lon0,lon8,lon4],[r0,r8,r4],color=lcolor,alpha=0.3,label="PUNCH NFI Field of View")

    



def plot_stereo_hi_fov(pos, time_num, timeind,ax,sc,label_display=False):    
    
    #plots the STA FOV HI1 HI2
    
    #STB never flipped the camera:    
    if sc=='B': 
        ang1d=-4
        ang2d=-24
        ang3d=-18
        ang4d=-88
        lcolor='blue'
    
    if sc=='A': 
        ang1d=4
        ang2d=24
        ang3d=18
        ang4d=88
        lcolor='red'

        #STA flipped during conjunction
        if mdates.date2num(datetime(2015,11,1))<time_num<mdates.date2num(datetime(2023,8,12)):  
            ang1d=-4
            ang2d=-24
            ang3d=-18
            ang4d=-88

    #calculate endpoints
    
    #sta position
    x0=pos.x[timeind]
    y0=pos.y[timeind]
    z0=0
    
    #sta position 180° rotated    
    x1=-pos.x[timeind]
    y1=-pos.y[timeind]
    z1=0
    
    r2,t2,lon2=angle_to_coord_line(ang1d,x0,y0,x1,y1)
    r3,t3,lon3=angle_to_coord_line(ang2d,x0,y0,x1,y1)   
    r4,t4,lon4=angle_to_coord_line(ang3d,x0,y0,x1,y1)

    r5,t5,lon5=angle_to_coord_line(ang4d,x0,y0,x1,y1)
    
    #convert to polar coordinates and plot
    [r0,t0,lon0]=Utils.cart2sphere(x0,y0,z0)    
    #[r1,t1,lon1]=hd.cart2sphere(x1,y1,z1)    
    
    
     
    


    rc11,tc21,lonc11=angle_to_coord_line(0.7,x0,y0,x1,y1)
    rc21,tc21,lonc21=angle_to_coord_line(4.0,x0,y0,x1,y1)

    rc12,tc22,lonc12=angle_to_coord_line(-0.7,x0,y0,x1,y1)
    rc22,tc22,lonc22=angle_to_coord_line(-4.0,x0,y0,x1,y1)
    # r2,t2,lon2=angle_to_coord_line(45,x0,y0,x1,y1) 

    #ax.plot([lon0,lon1],[r0,r1],'--r',alpha=0.5)

    if(label_display):
        label1="STEREO-A/HI1 Field of View"
        label2="STEREO-A/COR2 Field of View"
    else:
        label1=""
        label2=""


    ax.plot([lon0,lon2],[r0,r2],linestyle='-',color=lcolor,alpha=0.5, lw=1.2)
    ax.plot([lon0,lon3],[r0,r3],linestyle='-',color=lcolor,alpha=0.5, lw=1.2,label=label1)

    ax.fill([lon0,lonc11,lonc21],[r0,rc11,rc21],color=lcolor,alpha=0.3)
    ax.fill([lon0,lonc12,lonc22],[r0,rc12,rc22],color=lcolor,alpha=0.3,label=label2)
    # ax.plot([lon0,lon4],[r0,r4],linestyle='--',color=lcolor,alpha=0.3, lw=0.8)
    # ax.plot([lon0,lon5],[r0,r5],linestyle='--',color=lcolor,alpha=0.3, lw=0.8)


def make_frame_trajectories(positions,start_end=True,cmes=None,punch=False,trajectories=False):
    psp = positions['psp']
    solo = positions['solo']
    sta = positions['sta']
    earth = positions['l1'] #earth is l1!
    bepi = positions['bepi']
    mercury = positions['mercury']
    venus = positions['venus']
    mars = positions['mars']
    
    res_in_days=1/144.
    t_start = earth["time"][0][0]
    t_end  =  earth["time"][-1][0]
    frame_time_num=t_start


    
    backcolor='#052E37' #xkcd:black' '#052E37'
    psp_color='#052E37' #'xkcd:black' '#052E37'
    bepi_color='#5833FE'
    solo_color='#F29707' #'xkcd:orange' '#F29707'
    earth_color='#75CC41'
    sta_color='#E75C13'#
    mercury_color='#9dabae'
    venus_color='#8C11AA'
    mars_color='#E75C13'
    cme_color='#8C99FD'
    
    red = '#CC2C01' #'xkcd:magenta'
    green = earth_color #'#BFCE40' #'xkcd:green'
    blue = '#5833FE' #'xkcd:azure'


    fig,ax=plt.subplots(1,1,figsize = (10,10),dpi=100,subplot_kw={'projection': 'polar'}) #full hd

    if(start_end):
        ks = [0,-1]
        alphas = [1.0,0.6]
    else:
        ks = np.arange(0,len(positions["l1"]))
        alphas = np.ones((len(ks)))
    for k in ks:
        #plot all positions including text R lon lat for some 


        # ax.scatter(venus[k].lon, venus[k].r*np.cos(venus[k].lat), s=symsize_planet, c=venus_color, alpha=alphas[k],lw=0,zorder=3)
        # ax.scatter(mercury[k].lon, mercury[k].r*np.cos(mercury[k].lat), s=symsize_planet, c=mercury_color, alpha=alphas[k],lw=0,zorder=3)
        ax.scatter(earth[k].lon, earth[k].r*np.cos(earth[k].lat), s=symsize_planet, c=earth_color, alpha=alphas[k],lw=0,zorder=3)
        ax.scatter(sta[k].lon, sta[k].r*np.cos(sta[k].lat), s=symsize_spacecraft, c=sta_color, marker='s', alpha=alphas[k],lw=0,zorder=3,label="STEREO-A:  "+ mdates.num2date(sta[k].time[0]).strftime('%Y-%m-%d') )
        # ax.scatter(mars[k].lon, mars[k].r*np.cos(mars[k].lat), s=symsize_planet, c='#E75C13', alpha=0.7,lw=0,zorder=3)


        #plot stereoa fov hi1/2    
        plot_stereo_hi_fov(sta,frame_time_num, k, ax,'A',(True if k == -1 else False))
        ax.set_theta_zero_location('E')

        plt.thetagrids(range(0,360,45),(u'0\u00b0',u'45\u00b0',u'90\u00b0',u'135\u00b0',u'+/- 180\u00b0       ',u'- 135\u00b0',u'- 90\u00b0',u'- 45\u00b0'), ha='center', fmt='%d',fontsize=fsize-1,color=backcolor, alpha=0.9,zorder=4)
        plt.rgrids((0.1,0.3,0.5,0.7,1.0),('0.10','0.3','0.5','0.7','1.0 AU'),angle=180, fontsize=fsize-3,alpha=0.5, color=backcolor)
        plot_cmes(ax,cmes,k,frame_time_num,res_in_days)

        #ax.set_ylim(0, 1.75) #with Mars
        ax.set_ylim(0, 1.2)
        #Sun
        ax.scatter(0,0,s=100,c='#F9F200',alpha=1, edgecolors='black', linewidth=0.3)
        plt.show()
        plt.close(fig)
        if (not start_end):
            fig,ax=plt.subplots(1,1,figsize = (10,10),dpi=100,subplot_kw={'projection': 'polar'})    
    
    
    ax.plot(sta.lon[:], sta.r[:]*np.cos(sta.lat[:]), c='#CC2C01', linestyle='--', alpha=0.6,lw=1,zorder=3)
    ax.scatter(np.deg2rad(60.0),1.0,s=100,c='pink',alpha=1,  linewidth=0.3,marker='s',label="L4")
    # ax.plot(mercury.lon[:], mercury.r[:]*np.cos(mercury.lat[:]), c=mercury_color, linestyle='--', alpha=0.6,lw=1,zorder=3)
    # ax.plot(venus.lon[:], venus.r[:]*np.cos(venus.lat[:]), c=venus_color, linestyle='--', alpha=0.6,lw=1,zorder=3)
    draw_punch_fov(earth,frame_time_num,-1,ax)
    angle = np.deg2rad(67.5)
    ax.legend(loc="lower left",
            bbox_to_anchor=(.5 + np.cos(angle)/2, .5 + np.sin(angle)/2))

    plt.show()
    plt.close(fig)



def plot_cmes(ax,cmes,k,frame_time_num,res_in_days):
    
    date_frame = mdates.num2date((frame_time_num+float(k)*res_in_days))
    diffs = [np.abs((mdates.num2date(cmes["hc_time_num1"][i])- date_frame).total_seconds()) for i in range(0,len(cmes["hc_time_num1"]))]
    cmeind1=np.where(np.array(diffs)<60.0)[0]

    
    for p in range(0,np.size(cmeind1)):
        
        t = ((np.arange(201)-10)*np.pi/180)-(cmes["hc_lon1"][cmeind1[p]]*np.pi/180)
        t1 = ((np.arange(201)-10)*np.pi/180)
        
        longcirc1 = []
        rcirc1 = []
        for i in range(3):

            xc1 = cmes["c1_ell"][i][cmeind1[p]]*np.cos(cmes["hc_lon1"][cmeind1[p]]*np.pi/180)+((cmes["a1_ell"][i][cmeind1[p]]*cmes["b1_ell"][i][cmeind1[p]])/np.sqrt((cmes["b1_ell"][i][cmeind1[p]]*np.cos(t1))**2+(cmes["a1_ell"][i][cmeind1[p]]*np.sin(t1))**2))*np.sin(t)
            yc1 = cmes["c1_ell"][i][cmeind1[p]]*np.sin(cmes["hc_lon1"][cmeind1[p]]*np.pi/180)+((cmes["a1_ell"][i][cmeind1[p]]*cmes["b1_ell"][i][cmeind1[p]])/np.sqrt((cmes["b1_ell"][i][cmeind1[p]]*np.cos(t1))**2+(cmes["a1_ell"][i][cmeind1[p]]*np.sin(t1))**2))*np.cos(t)

            longcirc1.append(np.arctan2(yc1, xc1))
            rcirc1.append(np.sqrt(xc1**2+yc1**2))

        ax.plot(longcirc1[0],rcirc1[0], color=cme_color, ls='-', alpha=1-abs(cmes["hc_lat1"][cmeind1[p]]/100), lw=1.5) 
        ax.fill_between(longcirc1[2], rcirc1[2], rcirc1[1], color=cme_color, alpha=.05)




