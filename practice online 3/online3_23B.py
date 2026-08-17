import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
import sys


# =====================================================================
# Given classes — paste your Task 2 implementations where indicated
# =====================================================================

class ContinuousImage:
    """Represents a grayscale image as a continuous 2D spatial signal. (Given)"""

    def __init__(self, image_path):
        self.image = imread(image_path, mode='L').astype(float)
        self.image = self.image / np.max(self.image)
        self.x = np.linspace(-1, 1, self.image.shape[1])
        self.y = np.linspace(-1, 1, self.image.shape[0])


class CFT2D:
    """2D Continuous Fourier Transform. (Given — paste your Task 2 solution)"""

    def __init__(self, image_obj: ContinuousImage):
        self.I = image_obj.image
        self.x = image_obj.x
        self.y = image_obj.y
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]
        self.u = np.linspace(-1 / (2 * dx), 1 / (2 * dx), self.I.shape[1])
        self.v = np.linspace(-1 / (2 * dy), 1 / (2 * dy), self.I.shape[0])

    def compute_cft(self):
        I = self.I
        x, y = self.x, self.y # image er x, y coordinates
        u, v = self.u, self.v # frequencies

        angle_ux = 2 * np.pi * np.outer(u, x)
        cos_ux = np.cos(angle_ux)
        sin_ux = np.sin(angle_ux)
    
        A = np.trapezoid(I[:, None, :] * cos_ux[None, :, :], x=x, axis=-1)
        B = np.trapezoid(I[:, None, :] * sin_ux[None, :, :], x=x, axis=-1)
    
        angle_vy = 2 * np.pi * np.outer(v, y)
        cos_vy = np.cos(angle_vy)
        sin_vy = np.sin(angle_vy)
    
        A_T = A.T
        B_T = B.T
    
        real = np.trapezoid(
            cos_vy[:, None, :] * A_T[None, :, :] - sin_vy[:, None, :] * B_T[None, :, :],
            x=y, axis=-1,
        )
        imag = -np.trapezoid(
            cos_vy[:, None, :] * B_T[None, :, :] + sin_vy[:, None, :] * A_T[None, :, :],
            x=y, axis=-1,
        )
    
        return real, imag

    def plot_magnitude(self):
        real, imag = self.compute_cft()
        magnitude = np.sqrt(real ** 2 + imag ** 2)
        plt.imshow(np.log(1 + magnitude), cmap='gray')
        plt.title("Magnitude Spectrum (log scale)")
        plt.axis('off')
        plt.show()


class InverseCFT2D:
    """Inverse 2D-CFT. (Given — paste your Task 2 solution)"""

    def __init__(self, real, imag, u, v, x, y):
        self.real = real
        self.imag = imag
        self.u = u
        self.v = v
        self.x = x
        self.y = y

    def reconstruct(self):
        real, imag = self.real, self.imag
        u, v = self.u, self.v
        x, y = self.x, self.y

        angle_vy = 2 * np.pi * np.outer(v, y)
        cos_vy = np.cos(angle_vy)
        sin_vy = np.sin(angle_vy)
    
        C = np.trapezoid(real[:, None, :] * cos_vy[:, :, None], x=v, axis=0)
        S = np.trapezoid(real[:, None, :] * sin_vy[:, :, None], x=v, axis=0)
        D = np.trapezoid(imag[:, None, :] * cos_vy[:, :, None], x=v, axis=0)
        E = np.trapezoid(imag[:, None, :] * sin_vy[:, :, None], x=v, axis=0)
    
        CmE = C - E
        SpD = S + D
    
        angle_ux = 2 * np.pi * np.outer(u, x)
        cos_ux = np.cos(angle_ux)
        sin_ux = np.sin(angle_ux)
    
        image = np.trapezoid(
            CmE[:, :, None] * cos_ux[None, :, :] - SpD[:, :, None] * sin_ux[None, :, :],
            x=u, axis=1,
        )
    
        return image
        
# =====================================================================
# Task 1 — band_pass and band_stop filters
# =====================================================================

class FrequencyFilter:

    def high_pass(self, real, imag, cutoff):
        """Given."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                if np.sqrt((i - cx) ** 2 + (j - cy) ** 2) <= cutoff:
                    real[i, j] = 0
                    imag[i, j] = 0
        return real, imag

    def band_pass(self, real, imag, r_low, r_high):
        """TODO: retain entries with r_low < d(i,j) <= r_high, zero the rest."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i-cx)**2 + (j-cy)**2)
                if not (r_low < d) or not d <= r_high:
                    real[i, j] = 0
                    imag[i,j] = 0
        return real, imag


    def band_stop(self, real, imag, r_low, r_high):
        """TODO: zero entries with r_low < d(i,j) <= r_high, retain the rest."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        imag = imag.copy()
        for i in range(rows):
            for j in range(cols):
                d = np.sqrt((i-cx)**2 + (j-cy)**2)
                if (r_low < d) and d <= r_high:
                    real[i, j] = 0
                    imag[i,j] = 0
        return real, imag

    def shift_brightness(self, real, imag, shift_amount):
        """TODO: Task 3. Add shift_amount to the real component of the exact center pixel."""
        rows, cols = real.shape
        cx, cy = rows // 2, cols // 2
        real = real.copy()
        real[cx,cy] += shift_amount
        return real, imag

# =====================================================================
# Task 2 — complementarity check on raw spatial reconstructions
# =====================================================================

class ReconstructionValidator:

    def verify_complementarity(self, I_recon, I_bp, I_bs):
        """TODO: verify the complementarity property. Return (is_valid, delta)."""
        delta = np.max(np.abs(I_bp + I_bs - I_recon))
        is_valid = delta < 1e-9
        return (is_valid, delta)
    
# =====================================================================
# Entry point (given — do not modify)
# =====================================================================
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 cft_edge_detector.py <input_image>")
        sys.exit(1)

    input_path = sys.argv[1]
    r_low, r_high = 10, 50

    img   = ContinuousImage(input_path)
    cft2d = CFT2D(img)
    real, imag = cft2d.compute_cft()

    filt = FrequencyFilter()
    real_bp, imag_bp = filt.band_pass(real, imag, r_low, r_high)
    real_bs, imag_bs = filt.band_stop(real, imag, r_low, r_high)

    def reconstruct(r, im):
        return InverseCFT2D(r, im, cft2d.u, cft2d.v, img.x, img.y).reconstruct()

    I_recon = reconstruct(real,    imag)
    I_bp    = reconstruct(real_bp, imag_bp)
    I_bs    = reconstruct(real_bs, imag_bs)

    validator = ReconstructionValidator()
    is_valid, delta = validator.verify_complementarity(I_recon, I_bp, I_bs)
    print(f"Complementarity check: {is_valid} | max delta: {delta:.2e}")

    def save_edge_map(I_raw, path):
        edge_map = np.abs(I_raw)
        if edge_map.max() > 0:
            edge_map = edge_map / edge_map.max()
        plt.imsave(path, 1 - edge_map, cmap='gray')
        print(f"Saved {path}")

    save_edge_map(I_bp, "pikachu_bandpass.png")
    save_edge_map(I_bs, "pikachu_bandstop.png")

    # Task 3 execution
    real_shifted, imag_shifted = filt.shift_brightness(real, imag, shift_amount=2.0)
    I_brightened = reconstruct(real_shifted, imag_shifted)
    
    # Save brightened image (clip to [0,1], no edge-map inversion)
    I_brightened_clipped = np.clip(I_brightened, 0, 1)
    plt.imsave("pikachu_brightened.png", I_brightened_clipped, cmap='gray')
    print("Saved pikachu_brightened.png")
