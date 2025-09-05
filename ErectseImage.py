import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1 , 0)

#パディングを行う関数
def pad(u, m):
    up = np.zeros(len(u)+2*m, dtype=complex)
    up[m:m+len(u)] = u
    return up

#中央部分を切り出す関数
def crop_center(u, N):
    s = (len(u)-N)//2
    return u[s:s+N]

N=4096*8 # サンプリング数
Npad=4096*8 # パディングサイズ
wavelen=532*10**(-6) # 波長(mm)
dx=0.15e-3 # サンプリング間隔
print(dx)
f=1 # レンズの焦点距離
distance = np.arange(0, 8 + dx, dx) #レンズからの距離

x=np.linspace(-N/2,N/2-1,N)
xpos=N//2 +13500#物体点の分布位置 N//2は原点
#物体面の振幅分布
u0=[0.0 for _ in range(len(x))]
u0[xpos]=1
a=2 #1枚目のレンズと物体の距離
b=4 #レンズ間の距離
c=2 #2枚目のレンズと像の距離

x_mm = (np.arange(N) - N//2) * dx
D1 = 1.5  # 1枚目のレンズの直径(mm)
D2 = 1.5  # 2枚目のレンズの直径(mm)

upad=pad(u0,Npad)
fftu0=ffp.fft(upad)
# ASMによるレンズ前面の波面の計算
nux=ffp.fftfreq(N+2*Npad,dx)
nu_sq=1/wavelen**2-nux**2
mask=nu_sq>0
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*a)
ux=crop_center(ffp.ifft(fftu0*phase_func),N)

amplitude_map = np.zeros((len(distance), N))

#物体面と一枚目のレンズ面の間の波面を各距離で計算
for i in range(len(distance)):
    if distance[i]>a:
        break
    z=distance[i]
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=crop_center(ffp.ifft(fftu0*phase_func),N)
    amplitude_map[i, :] = np.abs(diffraction)**2


Px1=rect(x_mm / D1) # 瞳関数
fx = np.zeros_like(Px1, dtype=np.complex128)

# レンズの変換を適用
for i in range(N):
    fx[i]=ux[i]*Px1[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

fpad=pad(fx,Npad)
fftfx=ffp.fft(fpad)
#一枚目のレンズと二枚目のレンズの間の波面を各距離で計算
for i in range(len(distance)):
    if not a<distance[i]<a+b:
        continue
    z=distance[i]-a
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=crop_center(ffp.ifft(fftfx*phase_func),N)
    amplitude_map[i, :] = np.abs(diffraction)**2


# 各距離で角スペクトル法による伝搬計算
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*b)
gx=crop_center(ffp.ifft(fftfx*phase_func),N)


Px2=rect(x_mm / D2) # 瞳関数
hx = np.zeros_like(Px2, dtype=np.complex128)

# レンズの変換を適用
for i in range(N):
    hx[i]=gx[i]*Px2[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))

padhx=pad(hx,Npad)
ffthx=ffp.fft(padhx)

#二枚目のレンズと結像面の間の波面を各距離で計算
for i in range(len(distance)):
    if distance[i]<=a+b:
        continue
    z=distance[i]-a-b
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=crop_center(ffp.ifft(ffthx*phase_func),N)
    amplitude_map[i, :] = np.abs(diffraction)**2

#結像面での波面を計算
phase_func=np.zeros(len(nux),dtype=np.complex128)
phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*c)
diffraction=crop_center(ffp.ifft(ffthx*phase_func),N)
image=np.abs(diffraction)**2


x1=x*dx # 実空間の座標

#結像面の強度分布をプロット
fig,ax=plt.subplots(figsize=(5,4))
ax.plot(x,image)
ax.set_xlabel("x")
plt.show()

for i in range(len(distance)):
    if np.max(amplitude_map[i, :]) != 0:
        amplitude_map[i, :] /= np.max(amplitude_map[i, :])  # 各距離で正規化

        
#振幅強度のマップを表示
fig1, ax1 = plt.subplots(figsize=(16, 8))
extent = [distance[0], distance[-1], x1[0], x1[-1]]
im = ax1.imshow(amplitude_map.T, extent=extent, origin='lower', aspect='auto', cmap='gray')  # カラーマップを 'gray' に変更
ax1.set_xlabel("$z$")
ax1.set_ylabel("$x$") 
fig1.colorbar(im, ax=ax1, label="Amplitude Intensity")
fig1.savefig("Erectimage_amplitude_map.png")
plt.show()

