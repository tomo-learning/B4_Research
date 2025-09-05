import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1 , 0)

N=4096*2 # サンプリング数
wavelen=532*10**(-6) # 波長(mm)
dx=0.5e-3 # サンプリング間隔
print(dx)
f=1 # レンズの焦点距離
distance = np.arange(0, 8 + dx, dx) #レンズからの距離

x=np.linspace(-N/2,N/2-1,N)
xpos=6096
#物体面の振幅分布
u0=[0.0 for _ in range(len(x))]
for i in range(len(u0)):
    if i==xpos:
       u0[i]=1.0
a=2 #1枚目のレンズと物体の距離
b=4 #レンズ間の距離
c=2 #2枚目のレンズと像の距離

x_mm = (np.arange(N) - N//2) * dx
D1 = 1.5  # 1枚目のレンズの直径(mm)
D2 = 1.5  # 2枚目のレンズの直径(mm)

# ASMによるレンズ前面の波面の計算
nux=ffp.fftfreq(N,dx)
nu_sq=1/wavelen**2-nux**2
mask=nu_sq>0
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*a)
ux=ffp.ifft(ffp.fft(u0)*phase_func)



amplitude_map = np.zeros((len(distance), N))

for i in range(len(distance)):
    if distance[i]>a:
        break
    z=distance[i]
    nux=ffp.fftfreq(N,dx)
    nu_sq=1/wavelen**2-nux**2
    mask=nu_sq>0
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=ffp.ifft(ffp.fft(u0)*phase_func)
    amplitude_map[i, :] = np.abs(diffraction)**2


image=np.abs(ux)**2


Px1=rect(x_mm / D1) # 瞳関数
fx = np.zeros_like(Px1, dtype=np.complex128)

inputIntensity=0
outputIntensity=0

for i in range(len(ux)):
    if Px1[i]==1:
        inputIntensity+=np.abs(ux[i])**2

# レンズの変換を適用
for i in range(N):
    fx[i]=ux[i]*Px1[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

for i in range(N):
    fx[i]=ux[i]*Px1[i]*np.exp(-1j*2*np.pi*(np.sqrt(f**2+(x[i]*dx)**2)-f)/wavelen)

for i in range(len(ux)):
    if Px1[i]==1:
        outputIntensity+=np.abs(ux[i])**2

print("inputIntensity:",inputIntensity)
print("outputIntensity:",outputIntensity)

for i in range(len(distance)):
    if not a<distance[i]<a+b:
        continue
    z=distance[i]-a
    nux=ffp.fftfreq(N,dx)
    nu_sq=1/wavelen**2-nux**2
    mask=nu_sq>0
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=ffp.ifft(ffp.fft(fx)*phase_func)
    amplitude_map[i, :] = np.abs(diffraction)**2


# 各距離で角スペクトル法による伝搬計算
nux=ffp.fftfreq(N,dx)
nu_sq=1/wavelen**2-nux**2
mask=nu_sq>0
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*b)
gx=ffp.ifft(ffp.fft(fx)*phase_func)


Px2=rect(x_mm / D2) # 瞳関数
hx = np.zeros_like(Px2, dtype=np.complex128)

# レンズの変換を適用
for i in range(N):
    hx[i]=gx[i]*Px2[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

for i in range(len(distance)):
    if distance[i]<=a+b:
        continue
    z=distance[i]-a-b
    nux=ffp.fftfreq(N,dx)
    nu_sq=1/wavelen**2-nux**2
    mask=nu_sq>0
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=ffp.ifft(ffp.fft(hx)*phase_func)
    amplitude_map[i, :] = np.abs(diffraction)**2

nux=ffp.fftfreq(N,dx)
nu_sq=1/wavelen**2-nux**2
mask=nu_sq>0
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*c)
diffraction=ffp.ifft(ffp.fft(hx)*phase_func)
image=np.abs(diffraction)**2
print(sum(image))
x1=x*dx # 実空間の座標

fig,ax=plt.subplots(figsize=(5,4))
ax.plot(x,image)
ax.set_xlabel("x")
plt.show()

for i in range(len(distance)):
    if np.max(amplitude_map[i, :]) != 0:
        amplitude_map[i, :] /= np.max(amplitude_map[i, :])  # 各距離で正規化

renz1_pos = []
renz2_pos = []
for i in range(len(Px1)):
    if Px1[i]==1.0 and Px1[i-1]==0.0:
        renz1_pos.append(x[i])
    if Px1[i]==1.0 and Px1[i+1]==0.0:
        renz1_pos.append(x[i])
    if Px2[i]==1.0 and Px2[i-1]==0.0:
        renz2_pos.append(x[i])
    if Px2[i]==1.0 and Px2[i+1]==0.0:
        renz2_pos.append(x[i])
    
        
        
#振幅強度のマップを表示
fig1, ax1 = plt.subplots(figsize=(16, 8))
extent = [distance[0], distance[-1], x1[0], x1[-1]]
im = ax1.imshow(amplitude_map.T, extent=extent, origin='lower', aspect='auto', cmap='gray')  # カラーマップを 'gray' に変更
ax1.set_xlabel("$z$")
ax1.set_ylabel("$x$") 
ax1.plot([a, a], [renz1_pos[0]*dx,renz1_pos[1]*dx], color='red', linestyle='--', label='1st Lens Position')
ax1.plot([a+b, a+b], [renz2_pos[0]*dx,renz2_pos[1]*dx], color='blue', linestyle='--', label='2nd Lens Position')
fig1.colorbar(im, ax=ax1, label="Amplitude Intensity")
fig1.savefig("Erectimage_amplitude_map.png")
plt.show()

