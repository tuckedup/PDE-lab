"""Diagnose the 2D restriction."""

import numpy as np

n = 32
x = np.arange(n) / n
X, Y = np.meshgrid(x, x, indexing="ij")
u = np.cos(2 * np.pi * X) * np.cos(2 * np.pi * Y)

spec = np.fft.rfft2(u)
print("rfft2 shape:", spec.shape)
# Where is the energy?
mag = np.abs(spec)
idx = np.dstack(np.unravel_index(np.argsort(-mag, axis=None)[:6], mag.shape))[0]
for i, j in idx:
    print(f"  mode (kx={i if i <= n//2 else i-n}, ky={j if j <= n//2 else j-n})  |spec|={mag[i,j]:.3f}")

# Now the truncation used by restrict_2d
s = spec.copy()
s[9:, :] = 0.0
s[:, 9:] = 0.0
filt = np.fft.irfft2(s, s=(n, n))
print("\nmax |filt - u| =", np.max(np.abs(filt - u)))
print("max |filt| =", np.max(np.abs(filt)))

r = filt[::2, ::2]
xo = np.arange(16) / 16
Xo, Yo = np.meshgrid(xo, xo, indexing="ij")
expected = np.cos(2 * np.pi * Xo) * np.cos(2 * np.pi * Yo)
print("\nmax |r - expected| =", np.max(np.abs(r - expected)))
print("r[0, :4]  =", r[0, :4])
print("expected  =", expected[0, :4])
print("\nfilt[0, :6] =", filt[0, :6])
print("u[0, :6]    =", u[0, :6])
