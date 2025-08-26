import numpy as np
from scipy import fftpack as ffp
import matplotlib.pyplot as plt

def rect(x):
    return np.where(np.abs(x) <= 0.5, 1 , 0)

fig,ax=plt.subplots(2,figsize=(5,4))
N=2048
wavelen=0.6328*10**(-3)
dx=0.5e-3
f=2
distance = np.arange(0, 5 + dx, dx) 

x=np.linspace(-N/2,N/2-1,N)
gx=rect((x)/512)
print(gx)
fx = np.zeros_like(gx, dtype=np.complex128)
for i in range(N):
    fx[i]=gx[i]*np.exp(-1j*np.pi*(x[i]*dx)**2/(wavelen*f))
ax[0].plot(x,np.abs(fx)**2,"-k")
ax[0].set_xlabel("$x$")

amplitude_map = np.zeros((len(distance), N))

for i in range(len(distance)):
    z=distance[i]
    nux=ffp.fftfreq(N,dx)
    nu_sq=1/wavelen**2-nux**2
    mask=nu_sq>0
    phase_func=np.zeros(len(nux),dtype=np.complex128)
    phase_func[mask]=np.exp(1j*2*np.pi*np.sqrt(nu_sq[mask])*z)
    diffraction=ffp.ifft(ffp.fft(fx)*phase_func)
    amplitude_map[i, :] = np.abs(diffraction)**2

# 2次元プロット

x1=x*dx
for i in range(len(distance)):
    amplitude_map[i, :] /= np.max(amplitude_map[i, :])  # 各距離で正規化
fig1, ax1 = plt.subplots(figsize=(8, 6))
extent = [distance[0], distance[-1], x1[0], x1[-1]]
im = ax1.imshow(amplitude_map.T, extent=extent, origin='lower', aspect='auto', cmap='gray')  # カラーマップを 'gray' に変更
ax1.set_xlabel("$z$")  # 横軸を z に変更
ax1.set_ylabel("$x$")  # 縦軸を x に変更
fig1.colorbar(im, ax=ax1, label="Amplitude Intensity")
fig1.savefig("ASM1d_amplitude_map.png")
plt.show()

