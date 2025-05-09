# Lior Cohen 314818345

# importing required libraries
import math
import cmath
import matplotlib.pyplot as plt
import numpy as np


# creating functions to plot the graphs throughout the assignment

# function that plots smooth continuous graph of given signal
def plot_signal_smth(signal, time_samples, title='signal', xlabel='n', ylabel='Amplitude'):
    plt.figure(figsize=(10, 4))                                     # figure window & figure size formating
    plt.plot(time_samples, signal, color='blue', zorder=2)    # plotting the signal
    plt.axhline(y=0, color='black', linewidth=1.15, zorder=0)       # plotting the axis lines, x-axis first then y-axis
    plt.axvline(x=0, color='black', linewidth=1.15, zorder=0)       # zorder makes sure they are under the signal
    plt.title(title)                                                # plot title
    plt.xlabel(xlabel)                                              # x-axis title
    plt.ylabel(ylabel)                                              # y-axis title
    plt.grid(True)                                                  # adding grid
    plt.show()                                                      # showing the plot


# function that plots the discrete signal, here it is being used mainly for fourier series plots (ak, bk, ck, etc...)
def plot_disct(signal, samples, title='Discrete signal', xlabel='n', ylabel='Amplitude'):
    plt.figure(figsize=(8, 4))                                           # figure window & figure size formating
    plt.stem(samples, np.real(signal), linefmt='blue', markerfmt='bo', basefmt=" ")     # plot discrete signal
    plt.axhline(y=0, color='black', linewidth=1.15)                      # plotting the axis lines
    plt.axvline(x=0, color='black', linewidth=1.15)                      # x-axis first, then y-axis
    plt.title(title)                                                     # plot title
    plt.xlabel(xlabel)                                                   # x-axis title
    plt.ylabel(ylabel)                                                   # y-axis title
    plt.grid(True)                                                       # adding grid
    plt.show()                                                           # showing plot


# same as the discrete plot function but with one added line to zoom on the signal for better sight of its behavior
def plot_disct_zoom(signal, samples, zoom, title='Discrete signal (Zoom)', xlabel='n', ylabel='Amplitude'):
    plt.figure()
    plt.stem(samples, np.real(signal), linefmt='blue', markerfmt='bo', basefmt=" ")
    plt.axhline(y=0, color='black', linewidth=1.15)
    plt.axvline(x=0, color='black', linewidth=1.15)
    plt.xlim(-zoom, zoom)                   # zooms in on symmetric area on the x-axis
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.show()


# function to plot discrete plots of the magnitude and the phase of a signal in the same figure window
def plot_mag_and_phase(magnitude, signal_phase, samples, title='Mag & Phase of Signal', mag_xlabel='n',
                       mag_ylabel='Amplitude', p_xlabel='n', p_ylabel='Amplitude'):
    plt.figure(figsize=(10, 8))                                     # figure window & figure size formating
    plt.subplot(2, 1, 1)                                      # creating subplot 1 (top) for the magnitude
    plt.stem(samples, magnitude, linefmt='blue', markerfmt='bo', basefmt=" ")   # plotting the signal magnitude
    plt.axhline(y=0, color='black', linewidth=1.15)                 # axis lines formating
    plt.axvline(x=0, color='black', linewidth=1.15)
    plt.xlim(-100, 100)                                       # zooming in on the area between x = -100 to x =100
    plt.title(title)                                                # plot title
    plt.xlabel(mag_xlabel)                                          # magnitude subplot x-axis title
    plt.ylabel(mag_ylabel)                                          # magnitude subplot y-axis title
    plt.grid(True)                                                  # adding grid

    plt.subplot(2, 1, 2)                                      # creating subplot 2 (bottom) for the phase
    plt.stem(samples, signal_phase, linefmt='blue', markerfmt='bo', basefmt=" ")    # plotting the signal phase
    plt.axhline(y=0, color='black', linewidth=1.15)                 # axis lines formating
    plt.axvline(x=0, color='black', linewidth=1.15)
    plt.xlabel(p_xlabel)                                            # phase subplot x-axis title
    plt.ylabel(p_ylabel)                                            # phase subplot y-axis title
    plt.grid(True)                                                  # adding grid

    plt.tight_layout()                               # making sure the plots spaced right and not overlapping each over
    plt.show()                                       # showing plot


# ====Q1=====
# creating the time vector n= -1000:1:1000.

# start value is -1000 and the stop value is 1001 (1001 not included).
n = np.arange(-1000, 1001)

# creating the Window signal as required.
# np.abs(n) < 100 make sure that the condition is only for what between -99 and 99 (99 and -99 included)
a = np.where(np.abs(n) < 100, 1, 0)
N = len(a)          # calculating the signals length for future usage

plot_signal_smth(a, n, title='a(n) signal', xlabel='n', ylabel='a(n)')    # plotting the a(n) signal


# ====Q2====
# here we want to find out the Fourier series coefficients of a(n) and plotting it
# and also checking if the signal in the frequency domain is real and symmetric

# function that calculates the Fourier series and creating its array of coefficients, time domain -> frequency domain
def fourier_series(signal):
    num = len(signal)                                   # calculating the signals length
    coef = [0] * num                                    # creating the array of coefficients
    for ik in range(- num // 2, num//2 + 1):       # calculating the coefficients with the formula give on the task
        for elem in range(- num // 2, num//2 + 1):   # technically it also does the Fourier transform to the signal
            coef[ik + num // 2] += (1 / num) * signal[elem + num // 2] * np.exp(-1j * ik * 2 * math.pi *
                                                                                elem / num)
    return coef                                         # returning the coefficients array


# function that does the inverse Fourier transform, creates its array of coefficients, frequency domain -> time domain
def re_transform(signal):
    num = len(signal)                           # calculating the length of the given signal
    signal_t = np.zeros(num, dtype=complex)     # creating the array of zeroes for the inversed coefficients
    for rk in range(num):                       # rk indexes on the array
        rk_val = rk - num // 2                  # rk_val is the actual k index in Fourier math
        for mi in range(num):                   # mi indexes on the array
            m_val = mi - num // 2               # m_val is the actual m index in Fourier math
            signal_t[rk] += signal[mi] * cmath.exp(1j * rk_val * 2 * math.pi * m_val / num)  # inverse Fourier transform
    return np.real(signal_t)                    # returning the time domain signal


# function the finds specific coefficient and plots its signal in the time domain
def find_coef_and_plot(coef, signal, samples, sig_length, title='Coefficient', xlabel='n', ylabel='Amplitude'):
    idx = coef + sig_length // 2                    # finding the actual given index in the array
    sig_sp_coef = signal[idx]                       # saving the coefficient to a variable
    x_coef = sig_sp_coef * np.exp(1j * 2 * coef * samples * math.pi / sig_length)   # calculating it in the time domain
    plot_signal_smth(np.real(x_coef), samples, title, xlabel, ylabel)       # plotting the specific coefficient signal


ak = fourier_series(a)            # calling for the function to calculate and create the Fourier series of given signal
phase = np.angle(ak)              # calculating the phase of the Fourier series coefficients phase of ak
mag = np.abs(ak)                  # calculating the ak coefficients magnitude

plot_disct(ak, n, title='a(k), Fourier Series Coefficients', xlabel='k', ylabel='ak')     # plot discrete plot of ak
plot_disct_zoom(ak, n, 75, title='a(k), Fourier Series (Zoom)', xlabel='k', ylabel='ak')  # zooms on the discrete
                                                                                                # ak plot
plot_mag_and_phase(mag, phase, n, title='Magnitude and Phase of a(k)', mag_xlabel='k',
                   mag_ylabel='|ak|', p_xlabel='k', p_ylabel='Phase')               # plot magnitude and phase plot

# symmetry check

diffs = []                                  # creating array for the max difference between the sides if the signal
for i in range(1, N // 2 + 1):              # running on the positive indexes of the signal
    diff = abs(np.real(ak[i + N//2]) - np.real(ak[-i + N//2]))      # calculating the difference between the value
                                                                    # of the positive and negative indexes
    diffs.append(diff)                      # adding the value of the difference to the array

print("Max real difference:", max(diffs))       # finding the max difference value in the array and printing it

flag = True                             # declaring the flag before the loop
for i in range(1, N//2 + 1):            # running over the positive side of the array
    right = np.real(ak[i + N//2])       # index on the positive side
    left = np.real(ak[-i + N//2])       # index on the negative side
    if not np.isclose(left, right, atol=5e-4):  # checking if the difference is not greater than 5e-4
        flag = False                    # if the difference exceeds the tolerance value the flag will be updated

if flag:                                # checking if the flag is on True
    print('a_k is Symmetric')           # printing that the signal is symmetric
if not flag:                            # checking if the flag is on False
    print('a_k is Asymmetric')          # printing that the signal is asymmetric

# real check

max_imag = np.max(np.abs(np.imag(ak)))      # finding the max value of the imaginary part of the signal
print('Max imaginary value of ak:', max_imag)                             # printing the max value


# ====Q3====
# here we want to see the identity that says:
# multiplication by exponent in the frequency domain <==> shifting in the time domain

bk = np.zeros(N, dtype=complex)         # creating the array for the bk signal
for r in range(N):                      # running on all the 2001 indexes
    k_val = r - N//2                    # the actual index on the centered signal
    bk[r] = ak[r] * cmath.exp(-1j * 2 * np.pi * k_val * 100 / N)    # multiplying the index value with exponent

plot_disct(bk, n, title='b(k)', xlabel='k', ylabel='bk')    # plotting bk discrete signal
plot_disct_zoom(bk, n, 75, title='b(k), Fourier Coefficients (Zoom)', xlabel='k', ylabel='bk')    # zoom on the plot

b = re_transform(bk)                    # creating b(n) with the inverse Fourier transform function
b_real = np.real(b)                     # taking the real part of b(n)
plot_signal_smth(b_real, n, title='b(n) = a(n) shifted', xlabel='n', ylabel='b(n)')      # plotting b(n) (shifted a(n))
print(np.max(b_real))                   # printing the max value of b(n) to see if the signal transformed correctly

bk_mag = np.abs(bk)                     # creating the signal for the magnitude of b(k)
bk_phase = np.angle(bk)                 # creating the signal for the phase of b(k)
plot_mag_and_phase(bk_mag, bk_phase, n, title='Magnitude and Phase of b(k)', mag_xlabel='k',
                   mag_ylabel='|bk|', p_xlabel='k', p_ylabel='Phase')       # plotting the magnitude & phase signals

# finding a specific coefficient contribution and plotting it
find_coef_and_plot(10, bk, n, N, title='b_10 coefficient in time', xlabel='n', ylabel='b(10)')

# ===Q4===
# here we want to see the identity that says:
# "multiplication by k" in the frequency domain <==> derivative in the time domain

ck = np.zeros(N, dtype=complex)             # creating the array for c(k) signal
for i in range(N):                          # running on all the indexes of the signal
    k_val = i - N//2                        # the actual index of the centered signal
    ck[i] = ak[i] * (1 - cmath.exp(-1j * 2 * np.pi * k_val / N))   # implementing the formula given in the questions
                                                                   # sheet for multiplying by k in the frequency domain
                                                                   # to find the derivative in the time domain

c = re_transform(ck)                        # inverse transform on c(k) to find the c(n) signal
c_real = np.real(c)                         # taking the real part of the signal c(n)
plot_signal_smth(c_real, n, title='c(n) = a(n) derivative', xlabel='n', ylabel='c(n)')   # plotting c(n)

# ====Q5====
# here we want to see the identity that says:
# multiplication in the frequency domain <==> convolution in the time domain

dk = np.zeros(N, dtype=complex)     # creating the array for d(k) signal
for i in range(N):                  # running on all the indexes of the signal
    k_val = i - N//2                # the actual index on the centered signal
    dk[i] = ak[i] * ak[i] * N       # multiplying the signal by its length and itself for convolution in the time domain

d = re_transform(dk)                # inverse transform on d(k) to find the d(n) signal
d_real = np.real(d)                 # taking the real part of d(n)
plot_signal_smth(d_real, n, title='d(n) = a(n) convolution with itself', xlabel='n', ylabel='d(n)')      # plotting d(n)
d_max = np.max(d_real)              # finding the max value of d(n)
print('Max value of d(n):', d_max)                        # printing the max value

# ====Q6====
# here we want to see the parseval identity in the time and frequency domains ,and see if it gives the same value

par_dn = 0                          # initializing the variable for the Parseval identity in the time domain
par_dk = 0                          # initializing the variable for the Parseval identity in the frequency domain
for i in range(N):
    par_dk += np.abs(dk[i] * dk[i])    # calculating Parsevals sum for the frequency domain signal

for i in range(N):
    par_dn += (1 / N) * np.abs(d[i] * d[i])     # calculating Parsevals sum for the time domain signal

# printing the values
print('Parseval in the frequency domain over d(k):', par_dk)
print('Parseval in the time domain over d(n):', par_dn)

# ===Q7===
# here we want to see the identity that says:
# multiplication in the time domain <==> circular convolution in the frequency domain
# and checking how close the signals values to each other

e = np.zeros(N, dtype=complex)      # creating the array for e(n) signal
for i in range(N):
    e[i] = a[i] * b[i]              # calculating the e(n) signal by multiplying a(n) with b(n)

ek = fourier_series(e)              # finding the e(k) signal using the Fourier series function
e_real = np.real(e)                 # taking the real part of e(n)
plot_signal_smth(e_real, n, title='e(n) Signal', xlabel='n', ylabel='e(n)')     # plotting e(n)
plot_disct(ek, n, title='e(k), Regular Fourier Transform Samples', xlabel='k', ylabel='ek')   # plotting e(k)
plot_disct_zoom(ek, n, 75, title='e(k), Regular Fourier Transform Samples (Zoom)', xlabel='k', ylabel='ek')
# zoom on e(k)

ek_hat = np.zeros(N, dtype=complex)     # creating the array for e_hat(k) signal
for i in range(N):          # running on all the indexes of the signal
    k = i - N // 2          # the actual k index in the centered signal
    for j in range(N):      # running on all the indexes of the signal
        m = j - N // 2      # the actual m index in the centered signal
        km = (k - m) % N    # gives index between 0 to N-1
        idx_km = (km + N // 2) % N  # adjusting the index to centered with the signal
        ek_hat[i] += ak[j] * bk[idx_km]  # preforming the circular convolution

e_hat = re_transform(ek_hat)        # inverse transform for e_hat(k) -> e_hat(n)
e_hat_real = np.real(e_hat)         # taking the real part of e_hat(n)
plot_signal_smth(e_hat_real, n, title='e(n) Hat', xlabel='n', ylabel='e(n)')    # plotting e_hat(n)
plot_disct(ek_hat, n, title='ek hat coefficients', xlabel='k', ylabel='ek_hat')  # plotting e_hat(k)
plot_disct_zoom(ek_hat, n, 75, title='ek hat (Zoom)', xlabel='k', ylabel='ek_hat')  # zoom on e_hat(k)
max_error = np.max(np.abs(ek - ek_hat))     # finding the max difference between e(k) and e_hat(k)
print("max error:", max_error)              # printing the max difference

# ====Q8====
# here we want to see what happens to a(n) if we multiply it by cosine in the time domain, and see it also it the
# frequency domain

g = np.zeros(N, dtype=complex)              # creating the array for g(n) signal
for i in range(N):              # finding what happens to a(n) if we multiply it by cosine in the time domain
    g[i] = a[i] * math.cos(2 * np.pi * 500 * i / N)

g_real = np.real(g)             # taking the real part of g(n)
gk = fourier_series(g)          # finding g(k) with the inverse transform for g(n)
plot_signal_smth(g_real, n, title='g(n) Signal', xlabel='n', ylabel='g(n)')     # plotting g(n)
plot_disct(gk, n, title='g(k) coefficients', xlabel='k', ylabel='g(k)')         # plotting g(k)
plot_disct_zoom(gk, n, 600, title='g(k) (Zoom)', xlabel='k', ylabel='g(k)')  # zoom on g(k)
