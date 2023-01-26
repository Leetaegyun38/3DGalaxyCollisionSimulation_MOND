import numpy as np 
import matplotlib.pyplot as plt 
import imageio
import os
import datetime
from tqdm import tqdm
from mpl_toolkits.mplot3d import Axes3D

GRAV = 6.67384e-8
D_L = 3.08567758e21 # kilo parsec [cm <- kpc]
D_M = 1.9891e33 # solar mass [g]
D_T = 24.0 * 3600 * 365.2526 * 1e6 # Myr [년->초]

N_TOT=500 #1280
T_END=1000 #1500
time_interval = 5
ax_max = 4e1
ax_min = -4e1

#Galaxy1
Mass1 = 2e10
coordinate_x1 = 0
coordinate_y1 = 0
coordinate_z1 = 0
velocity_x1 = 0
velocity_y1 = 0
velocity_z1 = 0
alpha1 = 0
beta1 = 0
min_radi = 2
max_radi = 16
Schwarzschild_radi = 1e-1

#Galaxy2
Mass2 = 2e10
coordinate_x2 = 3e1
coordinate_y2 = 0
coordinate_z2 = 0
velocity_x2 = -4e-2
velocity_y2 = 2e-2
velocity_z2 = 2e-2
alpha2 = 0
beta2 = 20*np.pi/180
min_radii = 2
max_radii = 8
Schwarzschild_radii = 1e-1


M = Mass1, Mass2 #1230e9, 2e12 #2e10, 0.7e10 / 은하1, 은하2 질량
R0 = np.array([coordinate_x1,coordinate_y1,coordinate_z1,velocity_x1,velocity_y1,velocity_z1])
R1 = np.array([coordinate_x2,coordinate_y2,coordinate_z2,velocity_x2,velocity_y2,velocity_z2]) #76e1,0,-1.1e-1,0 / 두번째 은하


def fsmall(t, r):#별이 두 블랙홀에 받는 힘
    '''
      R1 ; massive object 1 [solar mass]
      M  ; masses of massive objects (1,2) [parsec] 
    '''
    global R1 #global : 전역변수

    x, y, z, vx, vy, vz = r
    
    GM0, GM1 = M 
    GX1, GY1, GZ1 = R1[0], R1[1], R1[2]

    C = (GRAV * D_M / D_L**3) * (D_T**2) #G랑 같은 차원 - 단위환산 상수
    
    rc0 = np.sqrt(x**2 + y**2 + z**2)**3 #원점 거리 세제곱
    rc1 = np.sqrt((x-GX1)**2 + (y-GY1)**2 + (z-GZ1)**2)**3 #두번째 은하 거리 세제곱
    
    xdot, ydot, zdot = vx, vy, vz
    
    if rc0 < Schwarzschild_radi: rc0 = Schwarzschild_radi
    if rc1 < Schwarzschild_radii: rc1 = Schwarzschild_radii #사건의 지평선 안쪽으로 들어가는 물체 - 거리1에 고정 -> 외부관측자에게는 정지한 것처럼 보임
    xddot = -C*GM0*x/rc0 -C*GM1*(x-GX1)/rc1 #가속도 a_x
    yddot = -C*GM0*y/rc0 -C*GM1*(y-GY1)/rc1 # 가속도 a_y
    zddot = -C*GM0*z/rc0 -C*GM1*(z-GZ1)/rc1 # 가속도 a_z
    return np.array([xdot, ydot, zdot, xddot, yddot, zddot])      

def fbig(t, r): #두번째은하 블랙홀이 첫번째 은하 블랙홀에 받는 힘
    '''
      M [solar mass] 
      R1 [parsec]
    '''
    GX1, GY1, GZ1, GVX1, GVY1, GVZ1 = r
    
    GM0, GM1 = M 
    GMR = GM0*GM1/(GM0+GM1) #환산질량

    C = (GRAV * D_M / D_L**3) * (D_T**2)
    Rc = np.sqrt(GX1**2 + GY1**2 + GZ1**2)**3 #두 은하 사이 거리 세제곱
    if Rc < max(Schwarzschild_radi, Schwarzschild_radii): Rc = max(Schwarzschild_radi, Schwarzschild_radii) #두 은하 충돌 1이 한계 -> 관측자 기준 멈춰있음을 표현
    xdot, ydot, zdot = GVX1, GVY1, GVZ1
    xddot = -C*GMR*GX1/Rc
    yddot = -C*GMR*GY1/Rc
    zddot = -C*GMR*GZ1/Rc
    
    return np.array([xdot, ydot, zdot, xddot, yddot, zddot])      


def rk4(f, t, h, r): #Runge Kutta Method
    k1 = f(t, r) #f는 함수다 (fbig, fsmall이었다.)
    k2 = f(t + h/2, r + h/2 * k1)
    k3 = f(t + h/2, r + h/2 * k2)
    k4 = f(t + h, r + h * k3)
    return (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def gen_stars1(n,R0,M0,rmin=2.,rmax=12.):
    theta = np.random.random(int(n)) * np.pi *2.0
    r = rmin + np.random.random(int(n)) * (rmax-rmin)
    sxr, syr, szr = r*np.cos(theta), r*np.sin(theta), 0
    sx, sy, sz = \
        sxr*np.cos(beta1)+syr*np.sin(alpha1)*np.sin(beta1)+szr*np.cos(alpha1)*np.sin(beta1)+R0[0], \
        syr*np.cos(alpha1)-szr*np.sin(alpha1)+R0[1], \
        -sxr*np.sin(beta1)+syr*np.sin(alpha1)*np.cos(beta1)+szr*np.cos(alpha1)*np.cos(beta1)+R0[2]
    v0 = np.sqrt(GRAV * (D_M * M0) / (D_L * r)) * (D_T / D_L)
    svxr, svyr, svzr = v0*np.cos(theta+np.pi/2.0), v0*np.sin(theta+np.pi/2.0), 0
    svx, svy, svz = \
        svxr*np.cos(beta1)+svyr*np.sin(alpha1)*np.sin(beta1)+svzr*np.cos(alpha1)*np.sin(beta1), \
        svyr*np.cos(alpha1)-svzr*np.sin(alpha1), \
        -svxr*np.sin(beta1)+svyr*np.sin(alpha1)*np.cos(beta1)+svzr*np.cos(alpha1)*np.cos(beta1)
    return sx, sy, sz, svx, svy, svz 
    
def gen_stars2(n,R0,M0,rmin=2.,rmax=12.):
    theta = np.random.random(int(n)) * np.pi *2.0     
    r = rmin + np.random.random(int(n)) * (rmax-rmin)
    sxr, syr, szr = r*np.cos(theta), r*np.sin(theta), 0
    sx, sy, sz = \
        sxr*np.cos(beta2)+syr*np.sin(alpha2)*np.sin(beta2)+szr*np.cos(alpha2)*np.sin(beta2)+R0[0], \
        syr*np.cos(alpha2)-szr*np.sin(alpha2)+R0[1], \
        -sxr*np.sin(beta2)+syr*np.sin(alpha2)*np.cos(beta2)+szr*np.cos(alpha2)*np.cos(beta2)+R0[2]
    v0 = np.sqrt(GRAV * (D_M * M0) / (D_L * r)) * (D_T / D_L)
    svxr, svyr, svzr = v0*np.cos(theta+np.pi/2.0), v0*np.sin(theta+np.pi/2.0), 0
    svx, svy, svz = \
        svxr*np.cos(beta2)+svyr*np.sin(alpha2)*np.sin(beta2)+svzr*np.cos(alpha2)*np.sin(beta2), \
        svyr*np.cos(alpha2)-svzr*np.sin(alpha2), \
        -svxr*np.sin(beta2)+svyr*np.sin(alpha2)*np.cos(beta2)+svzr*np.cos(alpha2)*np.cos(beta2)
    return sx, sy, sz, svx, svy, svz 

          
def integration(h, tend=1, n=100): 
    global R1

    sx1, sy1, sz1, svx1, svy1, svz1 = gen_stars1(n*0.5, R0, M[0], rmin=min_radi, rmax=max_radi) 
    sx2, sy2, sz2, svx2, svy2, svz2 = gen_stars2(n*0.5, R1, M[1], rmin=min_radii, rmax=max_radii) #별 개수비
    sx = np.concatenate((sx1, sx2))#, sx3, sx4, sx5, sx6, sx7, sx8), axis=0)
    sy = np.concatenate((sy1, sy2))#, sy3, sy4, sy5, sy6, sy7, sy8), axis=0)
    sz = np.concatenate((sz1, sz2))#, sy3, sy4, sy5, sy6, sy7, sy8), axis=0)
    svx = np.concatenate((svx1, svx2+R1[3]))#, svx3, svx4, svx5, svx6, svx7, svx8), axis=0)
    svy = np.concatenate((svy1, svy2+R1[4]))#, svy3, svy4, svy5, svy6, svy7, svy8), axis=0)
    svz = np.concatenate((svz1, svz2+R1[5]))#, svy3, svy4, svy5, svy6, svy7, svy8), axis=0)    

    px, py, pz, pvx, pvy, pvz = [], [], [], [], [], [] 
    pt, pG = [], [] 
    t = 0
    pbar = tqdm(total=tend)
    while t <= tend: 
        pbar.update(h)
        pt.append(t) 
        px.append(sx.copy()) #sx의 값을 복사해서 px에 추가한다.
        py.append(sy.copy())
        pz.append(sz.copy())
        pvx.append(svx.copy())
        pvy.append(svy.copy())
        pvz.append(svz.copy())
        pG.append(R1.copy()) 

        R1 += rk4(fbig, t, h, R1) 
        for i in range(n): 
            Rin = np.array([sx[i], sy[i], sz[i], svx[i], svy[i], svz[i]]) 
            dr = rk4(fsmall, t, h, Rin) #vx vy vz ax ay az
            sx[i] += dr[0]
            sy[i] += dr[1]
            sz[i] += dr[2]
            svx[i] += dr[3]
            svy[i] += dr[4]  #vy +ay * t
            svz[i] += dr[5]
        t += h 
        #print("Processing... " + str(round(t/tend*100, 1)) + "%")
    pbar.close()

    #print("Total data:", len(pt))
        
    return np.array(pt), \
           np.array(px), np.array(py), np.array(pz), \
           np.array(pvx), np.array(pvy), np.array(pvz), \
           np.array(pG) 


# calculate the numerical integration     
pt, px, py, pz, pvx, pvy, pvz, pG = \
     integration(time_interval, tend=T_END, n=N_TOT) 

for k in tqdm(range(len(pt))):
    # generate figure
    fig = plt.figure(1)
    # make axis
    ax = fig.add_subplot(121, projection='3d')
    # draw GALAXY1
    ax.scatter(R0[0], R0[1], R0[2], color='black', marker='o', s=10, \
        label='M1 = %5.1E M_sun' % (M[0],)) #그래프그리기
    # draw STARS
    for i in range(N_TOT):
        if i < N_TOT*0.5: #별 개수 비율 주의
            ax.scatter(px[k,i], py[k,i], pz[k,i], color='red', marker='o', s=0.1)
        else:
            ax.scatter(px[k,i], py[k,i], pz[k,i], color='blue', marker='o', s=0.1)
    # draw GALAXY2
    ax.scatter(pG[k,0], pG[k,1], pG[k,2], color='black', marker='o', s=7, \
            label='M2 = %5.1E M_sun' % (M[1],))

    ax.set_xlabel('X [kpc]')
    ax.set_ylabel('Y [kpc]')
    ax.set_zlabel('Z [kpc]')
    plt.suptitle('Galactic Interaction with %dstars; %dMyr' %(N_TOT, (k*time_interval)), fontsize = 16)

    ax.set_xlim(ax_min,ax_max)
    ax.set_ylim(ax_min,ax_max)
    ax.set_zlim(ax_min,ax_max)
    #ax.view_init(30,60)
    # input TIME label
    #tlabel = ax.text(2'  ', fontsize=20)

    #ax = Axes3D(fig)
    X = np.arange(ax_min,ax_max,(ax_max-ax_min)/7)
    Y = np.arange(ax_min,ax_max,(ax_max-ax_min)/7)
    X, Y = np.meshgrid(X, Y)                   # X,Y값을 직교좌표계로 변환
    Z = X*0 + Y*0

    #ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='summer')   
    wire = ax.plot_wireframe(X, Y, Z, color='limegreen', linewidth=0.3) 

    ax = fig.add_subplot(222,aspect=100)
    ax.plot(0,0,'yo', ms=5, \
        label='M1 = %5.1E M_sun' % (M[0],)) 
    ax.plot(np.sqrt(pG[k,0]**2+pG[k,1]**2+pG[k,2]**2), np.sqrt(pG[k,3]**2+pG[k,4]**2+pG[k,5]**2), 'go', ms=5, \
        label='M2 = %5.1E M_sun' % (M[1],)) 
    plt.suptitle('Galaxy rotation curve; %dMyr' %((k*time_interval)), fontsize = 16)
    #labe1= 'velocity'
    pdots = [] 
    for j in range(N_TOT):
        if j < N_TOT*0.5:
            pdot, = ax.plot(np.sqrt(px[k,j]**2+py[k,j]**2+pz[k,j]**2), np.sqrt(pvx[k,j]**2+pvy[k,j]**2+pvz[k,j]**2), 'ro', ms=0.6, \
                    markeredgewidth=1, alpha=1)
        else:
            pdot, = ax.plot(np.sqrt(px[k,j]**2+py[k,j]**2+pz[k,j]**2), np.sqrt(pvx[k,j]**2+pvy[k,j]**2+pvz[k,j]**2), 'bo', ms=0.6, \
                    markeredgewidth=1, alpha=1)
    pdots.append(pdot)

    tlabe1 = ax.text(1.2e1,2e-1,'  ', fontsize=20)
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('V')
    plt.xlim(0,4e1)
    plt.ylim(0,2.5e-1)
    plt.legend(fontsize=6,loc='lower right', bbox_to_anchor=(1,-0.6))
    plt.draw()

    ax = fig.add_subplot(224,aspect=100)
    ax.plot(0,0,'go', ms=5, \
        label='M2 = %5.1E M_sun' % (M[1],)) 
    ax.plot(np.sqrt(pG[k,0]**2+pG[k,1]**2+pG[k,2]**2), np.sqrt(pG[k,3]**2+pG[k,4]**2+pG[k,5]**2), 'yo', ms=5, \
        label='M1 = %5.1E M_sun' % (M[0],)) 
    #labe1= 'velocity'
    pdots = [] 
    for j in range(N_TOT):
        if j < N_TOT*0.5:
            pdot, = ax.plot(np.sqrt((px[k,j]-pG[k,0])**2+(py[k,j]-pG[k,1])**2+(pz[k,j]-pG[k,2])**2), np.sqrt((pvx[k,j]-pG[k,3])**2+(pvy[k,j]-pG[k,4])**2+(pvz[k,j]-pG[k,5])**2), 'ro', ms=0.6, \
                    markeredgewidth=1, alpha=1)
        else:
            pdot, = ax.plot(np.sqrt((px[k,j]-pG[k,0])**2+(py[k,j]-pG[k,1])**2+(pz[k,j]-pG[k,2])**2), np.sqrt((pvx[k,j]-pG[k,3])**2+(pvy[k,j]-pG[k,4])**2+(pvz[k,j]-pG[k,5])**2), 'bo', ms=0.6, \
                    markeredgewidth=1, alpha=1)
    pdots.append(pdot)
    tlabe1 = ax.text(1.2e1,2e-1,'  ', fontsize=20)
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('V')
    plt.xlim(0,4e1)
    plt.ylim(0,2.5e-1)
    plt.legend(fontsize=6,loc='lower right', bbox_to_anchor=(1,-0.6))
    plt.draw()

    plt.subplots_adjust(wspace=0.5)

    #fig.tight_layout()

    plt.savefig(r'C:\Users\enter1\Desktop\chang\galaxychang\test\interaction-%05i' % (k,))
    plt.close()

print("Making gif file", "Please Wait...")


### 이미지 gif 변환
directory = r'C:\Users\enter1\Desktop\chang\galaxychang\test'
file_type = r'png'
save_gif_name = r'Galactic_interation'
time = 20/len(pt)
speed_sec = { 'duration':time}
images = []
for file_name in tqdm(os.listdir(directory)):
    if file_name.endswith('.{}'.format(file_type)):
        file_path = os.path.join(directory, file_name)
        images.append(imageio.imread(file_path))
imageio.mimsave('{}/{}.gif'.format(directory, save_gif_name), images, **speed_sec)

#완료 시간 및 문구 출력
now = datetime.datetime.now()
nowTime = now.strftime('%H:%M:%S')
print("Complete (",nowTime,"): DO NOT TOUCH THE COMPUTER")







'''
ax.plot(R0[0], R0[1], R0[2], color='dimgrey', marker='o', linestyle='', ms=3, \
        label='M1 = %5.1E M_sun' % (M[0],)) #그래프그리기
'''

'''
# draw STARS
sdots = [] 
for i in range(N_TOT):
    if i < N_TOT*0.5: #별 개수 비율 주의
        sdot, = ax.plot(px[0,i], py[0,i], pz[0,i], color='red', marker='o', ms=1, \
                    markeredgewidth=1, alpha=0.3) # alpha는 투명도
    else:
        sdot, = ax.plot(px[0,i], py[0,i], pz[0,i], color='blue', marker='o', ms=1, \
                    markeredgewidth=1, alpha=0.3)
    sdots.append(sdot)

# draw GALAXY2
cdot, = \
  ax.plot(pG[0,0], pG[0,1], pG[0,2], color='black', marker='o', linestyle='', ms=3, \
          label='M2 = %5.1E M_sun' % (M[1],))

# input TIME label
#tlabel = ax.text(-2e1,2.5e1,'  ', fontsize=20) 

# input X-Y label
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
ax.set_zlabel('Z [kpc]')
plt.suptitle('Galactic Interaction with stars', fontsize = 16)

# set X-Y limit
plt.xlim(-1e2,1e2)
plt.ylim(-1e2,1e2)


plt.legend(fontsize=12,loc='lower right') #범례
# draw plot for each time-step 


for k in range(len(pt)): 
    update(k, pt, px, py, pz, pvx, pvy, pvz, pG)
    plt.draw()
    plt.savefig(r'D:\galaxy_data\test-%05i' % (k,))
    print("Image saving... " + str(round((k+1)/len(pt)*100, 2)) + "%")
    
print("Making gif file", "Please Wait")



### 이미지 gif 변환
directory = r'D:\galaxy_data'
file_type = r'png'
save_gif_name = r'Galactic_interation'
time = 20/len(pt)
speed_sec = { 'duration':time}

images = []
for file_name in os.listdir(directory):
    if file_name.endswith('.{}'.format(file_type)):
        file_path = os.path.join(directory, file_name)
        images.append(imageio.imread(file_path))

imageio.mimsave('{}/{}.gif'.format(directory, save_gif_name), images, **speed_sec)


now = datetime.datetime.now()
nowTime = now.strftime('%H:%M:%S')
print("Complete (",nowTime,"): DO NOT TOUCH THE COMPUTER")
'''