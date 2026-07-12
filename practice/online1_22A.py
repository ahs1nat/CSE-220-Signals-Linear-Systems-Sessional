import numpy as np
import matplotlib.pyplot as plt

# ----------------------------
# Time axis
# ----------------------------
T_MIN, T_MAX, N = -4.0, 4.0, 4001

def x_of_t(t: np.ndarray) -> np.ndarray: # t is the passed argument which should be a n dimensional numpy array. the return type also has to be a numpy array
    """
    Base signal x(t): sinusoidal signal
    """
    return (
        np.sin(2 * np.pi * 0.5 * t)
        + 0.5 * np.sin(2 * np.pi * 1.5 * t)
    )


# ==========================================================
# ANSWER IMPLEMENTATION
# ==========================================================

# computer works with discrete data instead of continuous ones. so we interpolate the signal
# It looks at the nearest left neighbor and nearest right neighbor in the original data arrays and
# calculates a weighted average (linear interpolation) to seamlessly fill in those missing "in-between" values
def interpolate_signal(
    t_original: np.ndarray,
    x_original: np.ndarray,
    t_query: np.ndarray
) -> np.ndarray:
    """
    Interpolate using average of two neighboring samples.
    """
    # t_query is the new timeline after doing t/k
    return np.interp(t_query, t_original, x_original)


def time_scale(
    t: np.ndarray,
    x: np.ndarray,
    k: int
) -> np.ndarray:
    """
    Time sub-scaling:
        y(t) = x(t / k)
    """
    time_scaled = t/k
    return interpolate_signal(t, x, time_scaled)


def plot_pair(t: np.ndarray, x: np.ndarray, y: np.ndarray, title: str):
    """
    Plot graphs.
    """
    plt.figure(figsize=(10,6))
    plt.plot(t, x, label='$x(t)$ Original', color='blue',linewidth = 1.5)
    plt.plot(t,y, label='$y(t)$ (scaled)', color = 'red', linewidth=1.5)
    plt.title(title)
    plt.grid(True, linestyle=':',alpha=0.6)
    plt.xlim(T_MIN, T_MAX)
    plt.legend(fontsize=11, loc='upper right')


# ----------------------------
# Main
# ----------------------------
def main():
    t = np.linspace(T_MIN, T_MAX, N)
    x = x_of_t(t)

    k = 2   # sub-scaling factor
    y = time_scale(t, x, k)

    plot_pair(
        t,
        x,
        y,
        title=f"Time Sub-scaling: y(t) = x(t / {k})"
    )
    plt.show()


if __name__ == "__main__":
    main()