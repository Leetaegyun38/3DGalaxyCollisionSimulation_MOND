import numpy as np 
import matplotlib.pyplot as plt 
import imageio
import os

GRAV = 6.67384e-8 # Gravity constant
D_L = 3.08567758e21 # kilo parsec (kpc -> cm)
D_M = 1.9891e33 # solar mass (solar mass -> g)
D_T = 24.0 * 3600 * 365.2526 * 1e6 # 1 Myr (year -> sec)
A0 = 1.2e-7 * 10 # 수정뉴턴역학 가속도 상수 (cm/s^2) 1.2e-10*10

N_TOT=1200 # 별 개수
T_END=5000 # 종료시간
M = 2e10, 1e10 # 은하1, 은하2 질량 2e10, 0.7e10
R0 = np.array([0,0,0,0]) # 첫 번째 은하(x, y, vx, vy)
R1 = np.array([4e1,0,0,2.0e-2]) # 두 번째 은하(x, y, vx, vy)
M_STAR = 1

def fsmall(t, r): # 별의 속도와 좌표 입력, 가속도 산출해서 array에 넣어줌 (by 블랙홀 두 개)
    '''
      R1 ; massive object 1 [solar mass]
      M  ; masses of massive objects (1,2) [parsec] 
    '''
    global R1 #global: 전역변수

    x, y, vx, vy = r
    
    GM0, GM1 = M 
    GX1, GY1 = R1[0], R1[1] # 두 번째 은하의 좌표

    C = (GRAV * D_M / D_L**3) * (D_T**2) # =G: 단위환산자(sm->g, kpc->cm, year->sec)

    rc0 = np.sqrt(x**2 + y**2) #원점 거리
    rc1 = np.sqrt((x-GX1)**2 + (y-GY1)**2) #두 번째 은하 거리
    
    xdot, ydot = vx, vy
    if rc0 < 1e0: rc0 = 1e0 #외부 관측자에게 사건의지평선에서 정지한 것처럼 보임
    if rc1 < 1e0: rc1 = 1e0
    xddot = -C*GM0*M_STAR*x/rc0**3 -C*GM1*M_STAR*(x-GX1)/rc1**3 #x 가속도 (x/r=cosw)
    yddot = -C*GM0*M_STAR*y/rc0**3 -C*GM1*M_STAR*(y-GY1)/rc1**3 #y 가속도

#    if  (xddot**2 + yddot**2) <= A0**2:
#        xddot = -x/rc0*(C*GM0*A0/rc0**2)**0.5 -(x-GX1)/rc1*(C*GM1*A0/rc1**2)**0.5
#        yddot = -y/rc0*(C*GM0*A0/rc0**2)**0.5 -(y-GY1)/rc1*(C*GM1*A0/rc1**2)**0.5
    return np.array([xdot, ydot, xddot, yddot])      

def fbig(t, r): # 두 번째 은하의 속도와 좌표 입력, 가속도 산출해서 array에 넣어줌(by 첫 번째 블랙홀)
    '''
      M [solar mass] 
      R1 [parsec]
    '''
    GX1, GY1, GVX1, GVY1 = r
    
    GM0, GM1 = M 
    GMR = GM0*GM1/(GM0+GM1) #환산질량

    C = (GRAV * D_M / D_L**3) * (D_T**2)
    Rc = np.sqrt(GX1**2 + GY1**2) #두 은하 사이의 거리
    if Rc < 1e0: Rc = 1e0 #두 은하 충돌x(관측자 입장에서 정지)
    xdot, ydot = GVX1, GVY1
    xddot = -C*GMR*GX1/Rc**3
    yddot = -C*GMR*GY1/Rc**3

    if  (xddot**2 + yddot**2) <= A0**2:
        xddot = -GX1/Rc*(C*GM0*A0/Rc**2)**0.5
        yddot = -GY1/Rc*(C*GM0*A0/Rc**2)**0.5   
    
    return np.array([xdot, ydot, xddot, yddot])      


def rk4(f, t, h, r): #4차 룽게-쿠타 방법
    k1 = f(t, r) #fbig, fsmall
    k2 = f(t + h/2, r + h/2 * k1)
    k3 = f(t + h/2, r + h/2 * k2)
    k4 = f(t + h, r + h * k3)
    return (h/6) * (k1 + 2*k2 + 2*k3 + k4)

def gen_stars(n,r,R0,M0):
    theta = np.random.random(int(n)) * np.pi *2.0     
    sx, sy = r*np.cos(theta)+R0[0], r*np.sin(theta)+R0[1]
    v0 = np.sqrt(GRAV * (D_M * M0) / (D_L * r)) * (D_T / D_L)
    svx, svy = v0*np.cos(theta+np.pi/2.0), v0*np.sin(theta+np.pi/2.0)
    return sx, sy, svx, svy 
    
def gen_stars2(n,R0,M0,rmin=2.,rmax=15.): #Mis 1,2 mis 2
    theta = np.random.random(int(n)) * np.pi *2.0     
    r = rmin + np.random.random(int(n)) * (rmax-rmin)
    sx, sy = r*np.cos(theta)+R0[0], r*np.sin(theta)+R0[1]
    v0 = np.sqrt(GRAV * (D_M * M0) / (D_L * r)) * (D_T / D_L)
    svx, svy = v0*np.cos(theta+np.pi/2.0), v0*np.sin(theta+np.pi/2.0)
    return sx, sy, svx, svy 

          
def integration(h, tend=1, n=100): 
    global R1

    sx1, sy1, svx1, svy1 = gen_stars2(n*0.9, R0, M[0], rmin=2., rmax=12.) 
    sx2, sy2, svx2, svy2 = gen_stars2(n*0.1, R1, M[1], rmin=2., rmax=4.) #Mis8
    sx = np.concatenate((sx1, sx2))#, sx3, sx4, sx5, sx6, sx7, sx8), axis=0)
    sy = np.concatenate((sy1, sy2))#, sy3, sy4, sy5, sy6, sy7, sy8), axis=0)
    svx = np.concatenate((svx1, svx2+R1[2]))#, svx3, svx4, svx5, svx6, svx7, svx8), axis=0)
    svy = np.concatenate((svy1, svy2+R1[3]))#, svy3, svy4, svy5, svy6, svy7, svy8), axis=0)    

    px, py, pvx, pvy = [], [], [], [] 
    pt, pG = [], [] 
    t = 0  
    while t <= tend: 
        pt.append(t) 
        px.append(sx.copy()) #sx의 값을 복사해서 px에 추가한다.
        py.append(sy.copy())
        pvx.append(svx.copy())
        pvy.append(svy.copy())
        pG.append(R1.copy()) 

        R1 += rk4(fbig, t, h, R1) 
        for i in range(n): 
            Rin = np.array([sx[i], sy[i], svx[i], svy[i]]) 
            dr = rk4(fsmall, t, h, Rin) 
            sx[i] += dr[0]
            sy[i] += dr[1]
            svx[i] += dr[2]
            svy[i] += dr[3]  
        t += h 
        print("Processing... " + str(round(t/tend*100, 1)) + "%")

    print("Total data:", len(pt))
        
    return np.array(pt), \
           np.array(px), np.array(py), \
           np.array(pvx), np.array(pvy), \
           np.array(pG) 

# animation routine 
def update(tidx,pt, px, py, pvx, pvy, pG):
    for i in range(N_TOT):
        sdots[i].set_data(px[tidx,i],py[tidx,i])
    cdot.set_data(pG[tidx,0], pG[tidx,1])
    tlabel.set_text("T = %d Myrs" % (pt[tidx],))
    return sdots, cdot

# calculate the numerical integration     
pt, px, py, pvx, pvy, pG = \
     integration(5, tend=T_END, n=N_TOT) 

# generate figure
fig = plt.figure()
# make axis
ax = fig.add_subplot(111,aspect=1) 
# draw GALAXY1
ax.plot(R0[0], R0[1], 'bo', ms=15, \
        label='M1 = %5.1E M_sun' % (M[0],)) #그래프그리기
# draw STARS
sdots = [] 
for i in range(N_TOT):
    if i < N_TOT*0.9:
        sdot, = ax.plot(px[0,i], py[0,i], 'bo', ms=0.7, \
                    markeredgewidth=1, alpha=1)
    else:
        sdot, = ax.plot(px[0,i], py[0,i], 'ro', ms=0.7, \
                    markeredgewidth=1, alpha=1)
    sdots.append(sdot)
# draw GALAXY2
cdot, = \
  ax.plot(pG[0,0], pG[0,1], 'ro', ms=10, \
          label='M2 = %5.1E M_sun' % (M[1],))
# input TIME label
tlabel = ax.text(-2e1,2.5e1,'  ', fontsize=20)

# input X-Y label
ax.set_xlabel('X [kpc]')
ax.set_ylabel('Y [kpc]')
# set X-Y limit
plt.xlim(-1.5e1,3.5e1)
plt.ylim(-1.4e1,1.4e1)
plt.xlim(-6e1,6e1)
plt.ylim(-6e1,6e1)
plt.legend(fontsize=12,loc='lower right')

# draw plot for each time-step 
for k in range(len(pt)):
    update(k, pt, px, py, pvx, pvy, pG)
    plt.draw()
    plt.savefig(r'C:\Users\enter1\Desktop\chang\galaxychang\test\galaxy-%05i' % (k,))
    print("Image saving... " + str(round((k+1)/len(pt)*100, 2)) + "%")

print("Making gif file", "Please Wait")

### 이미지 gif 변환
directory = r'C:\Users\enter1\Desktop\chang\galaxychang\test'
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
print("Complete:")