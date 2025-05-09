"""
import math
import cmath
import matplotlib.pyplot as plt
import numpy as np


def plot_signal_smth(signal, time_samples, title='signal', xlabel='n', ylabel='Amplitude'):
    plt.figure(figsize=(10, 4))
    plt.plot(time_samples, signal, color='blue', zorder=2)
    plt.axhline(y=0, color='black', linewidth=1.15, zorder=0)
    plt.axvline(x=0, color='black', linewidth=1.15, zorder=0)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_disct(signal, samples, title='Discrete signal', xlabel='n', ylabel='Amplitude'):
    plt.figure()
    plt.stem(samples, np.real(signal), linefmt='blue', markerfmt='bo', basefmt=" ")
    plt.axhline(y=0, color='black', linewidth=1.15)
    plt.axvline(x=0, color='black', linewidth=1.15)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_disct_zoom(signal, samples, title='Discrete signal (Zoom)', xlabel='n', ylabel='Amplitude'):
    plt.figure()
    plt.stem(samples, np.real(signal), linefmt='blue', markerfmt='bo', basefmt=" ")
    plt.axhline(y=0, color='black', linewidth=1.15)
    plt.axvline(x=0, color='black', linewidth=1.15)
    plt.xlim(-75, 75)
    plt.ylim(-0.03, 0.11)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


def plot_mag_and_phase(magnitude, signal_phase, samples, title='Mag & Phase of Signal', mag_xlabel='n',
                       mag_ylabel='Amplitude', p_xlabel='n', p_ylabel='Amplitude'):

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.stem(samples, magnitude, linefmt='blue', markerfmt='bo', basefmt=" ")
    plt.axhline(y=0, color='black', linewidth=1.15)
    plt.axvline(x=0, color='black', linewidth=1.15)
    plt.xlim(-100, 100)
    plt.title(title)
    plt.xlabel(mag_xlabel)
    plt.ylabel(mag_ylabel)
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.stem(samples, signal_phase, linefmt='blue', markerfmt='bo', basefmt=" ")
    plt.axhline(y=0, color='black', linewidth=1.15)
    plt.axvline(x=0, color='black', linewidth=1.15)
    plt.xlabel(p_xlabel)
    plt.ylabel(p_ylabel)
    plt.grid(True)

    plt.tight_layout()
    plt.show()


# creating the time vector n= -1000:1:1000.
# start value is -1000 and the stop value is 1001 (1001 not included).
n = np.arange(-1000, 1001)

# creating the Window signal as required.
# np.abs(n) < 100 make sure that the condition is only for what between -99 and 99 (99 and -99 included)
a = np.where(np.abs(n) < 100, 1, 0)
N = len(a)
# ----Plotting the signal and the axis----
# ====Q1=====
# plotting the signal, the higher zorder is means that the line is above the lower zorder lines
plot_signal_smth(a, n, title='Window signal', xlabel='n', ylabel='a(n)')

# ====Q2====


def fourier_series(signal):
    num = len(signal)
    coef = [0] * num
    for ik in range(-int(num / 2), int(num / 2)):
        for elem in range(-int(num / 2), int(num / 2)):
            coef[ik + int(num / 2)] += (1 / num) * a[elem + int(num / 2)] * cmath.exp(-1j * ik * 2 * math.pi * elem /
                                                                                      num)
    return coef


def re_transform(signal):
    num = len(signal)
    signal_t = np.zeros(num, dtype=complex)
    for rk in range(num):
        rk_val = rk - num // 2
        for m in range(num):
            m_val = m - num // 2
            signal_t[rk] += signal[m] * cmath.exp(1j * rk_val * 2 * math.pi * m_val / num)
    return np.real(signal_t)


def find_coef_and_plot(coef, signal, samples, sig_length, title='Coefficient', xlabel='n', ylabel='Amplitude'):
    idx = coef + sig_length//2
    sig_sp_coef = signal[idx]
    x_coef = sig_sp_coef * np.exp(1j * 2 * coef * samples * math.pi / sig_length)
    plot_signal_smth(np.real(x_coef), samples, title, xlabel, ylabel)


ak = fourier_series(a)
phase = np.angle(ak)
mag = np.abs(ak)

plot_disct(ak, n, title='Fourier Series Coefficients', xlabel='k', ylabel='ak')
plot_disct_zoom(ak, n, title='Fourier Series (Zoom)', xlabel='k', ylabel='ak')
plot_mag_and_phase(mag, phase, n, title='Magnitude and Phase of Fourier Coefficients', mag_xlabel='k',
                   mag_ylabel='|ak|', p_xlabel='k', p_ylabel='Phase')

# ---------------
# need to prove real and symmetric in the frequency domain
# ---------------

# ====Q3====
k = np.arange(-1000, 1001)
bk = np.zeros(N, dtype=complex)
for r in range(N):
    k_val = r - N//2
    bk[r] = ak[r] * cmath.exp(-1j * 2 * np.pi * k_val * 100 / N)

plot_disct(bk, k, title='Fourier Coefficients shifted', xlabel='k', ylabel='bk')
plot_disct_zoom(bk, k, title='Fourier Coefficients (Zoom)', xlabel='k', ylabel='bk')

b = re_transform(bk)
b_real = np.real(b)
plot_signal_smth(b_real, n, title='Signal shifted', xlabel='n', ylabel='b(n)')
print(np.max(b_real))

bk_mag = np.abs(bk)
bk_phase = np.angle(bk)
plot_mag_and_phase(bk_mag, bk_phase, k, title='Magnitude and Phase of shifted Fourier Coefficients', mag_xlabel='k',
                   mag_ylabel='|bk|', p_xlabel='k', p_ylabel='Phase')

find_coef_and_plot(10, bk, k, N, title='b_10 coefficient in time', xlabel='n', ylabel='b(10)')

# ===Q4===

ck = np.zeros(N, dtype=complex)
for i in range(N):
    k_val = i - N//2
    ck[i] = ak[i] * (1 - cmath.exp(-1j * 2 * np.pi * k_val / N))

c = re_transform(ck)
c_real = np.real(c)
plot_signal_smth(c_real, n, title='Signal derivative', xlabel='n', ylabel='c(n)')

# ====Q5====

dk = np.zeros(N, dtype=complex)
for i in range(N):
    k_val = i - N//2
    dk[i] = ak[i] * ak[i] * N

d = re_transform(dk)
d_real = np.real(d)
plot_signal_smth(d_real, n, title='Signal convolution', xlabel='n', ylabel='d(n)')
d_max = np.max(d_real)
print(d_max)
"""