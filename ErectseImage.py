import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1 , 0)

N=2048 # サンプリング数
wavelen=0.6328*10**(-3) # 波長(mm)
dx=0.5e-3 # サンプリング間隔
f=2 # レンズの焦点距離
distance = np.arange(0, 15 + dx, dx) #レンズからの距離

x=np.linspace(-N/2,N/2-1,N)

#物体面の振幅分布
u0=[0.0 for _ in range(len(x))]
for i in range(len(u0)):
    if 1023<i<1324:
       u0[i]=1.0
a=4 #1枚目のレンズと物体の距離
b=8 #レンズ間の距離
c=4 #2枚目のレンズと像の距離

# ASMによるレンズ前面の波面の計算
nux=ffp.fftfreq(N,dx)
nu_sq=1/wavelen**2-nux**2
mask=nu_sq>0
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*a)
ux=ffp.ifft(ffp.fft(u0)*phase_func)

Px1=rect((x)/1024) # 瞳関数
fx = np.zeros_like(Px1, dtype=np.complex128)

# レンズの変換を適用
for i in range(N):
    fx[i]=ux[i]*Px1[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

amplitude_map = np.zeros((len(distance), N))

for i in range(len(distance)):
    if distance[i]>b:
        break
    z=distance[i]
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

Px2=rect((x)/50000) # 瞳関数
hx = np.zeros_like(Px2, dtype=np.complex128)

# レンズの変換を適用
for i in range(N):
    hx[i]=gx[i]*Px2[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

for i in range(len(distance)):
    if distance[i]<=b:
        continue
    z=distance[i]-b
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

x1=x*dx # 実空間の座標

fig,ax=plt.subplots(figsize=(5,4))
ax.plot(x,image)
ax.set_xlabel("x")
plt.show()

for i in range(len(distance)):
    if np.max(amplitude_map[i, :]) != 0:
        amplitude_map[i, :] /= np.max(amplitude_map[i, :])  # 各距離で正規化

#振幅強度のマップを表示
fig1, ax1 = plt.subplots(figsize=(16, 8))
extent = [distance[0], distance[-1], x[0], x[-1]]
im = ax1.imshow(amplitude_map.T, extent=extent, origin='lower', aspect='auto', cmap='gray')  # カラーマップを 'gray' に変更
ax1.set_xlabel("$z$")
ax1.set_ylabel("$x$") 
fig1.colorbar(im, ax=ax1, label="Amplitude Intensity")
fig1.savefig("Erectimage_amplitude_map.png")
plt.show()

