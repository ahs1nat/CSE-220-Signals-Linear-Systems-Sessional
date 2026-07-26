'''
Given

x(t)=cos(2t)

for

−π≤t≤π

Implement

y(t)=3x(−t)

Requirements:
Generate the time axis.
Implement transform_signal(t, x, alpha) where alpha=3.
Plot both signals.
Values outside [−π,π] should be zero.
Repeatedly ask the user for alpha until they enter q.
'''

import numpy as np
import matplotlib.pyplot as plt

T_MIN, T_MAX = -np.pi, np.pi


def base_signal(t):
    """
    Base signal:
        x(t) = cos(2t)
    Defined only on [-pi, pi].
    """
    x = np.cos(2 * t)
    x[(t < T_MIN) | (t > T_MAX)] = 0
    return x


def transform_signal(t, x, alpha):
    """
    Compute:
        y(t) = alpha * x(-t)

    TODO:
    - Perform time reversal.
    - Perform amplitude scaling.
    """
    x_rev = base_signal(-t)
    y = alpha * x_rev
    return y


def plot_signals(t, x, y, alpha):
    plt.figure(figsize=(8, 5))

    plt.plot(t, x, label="x(t)", linewidth=2)
    plt.plot(t, y, label=f"y(t) = {alpha}x(-t)", linewidth=2)

    plt.title("Time Reversal and Amplitude Scaling")
    plt.xlabel("t")
    plt.ylabel("Amplitude")

    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():

    # Generate time axis
    t = np.linspace(T_MIN, T_MAX, 1000)

    # Generate base signal
    x = base_signal(t)

    print("Transformation:")
    print("    y(t) = alpha * x(-t)")
    print("Enter 'q' to quit.\n")

    while True:

        # TODO:
        # 1. Ask the user for alpha.
        # 2. Quit if the user enters 'q'.
        # 3. Convert alpha to float.
        # 4. Call transform_signal().
        # 5. Plot the signals.
        alpha = input("Enter the value of alpha: " )
        if alpha.lower() == 'q':
            pass
        alpha = float(alpha)
        y = transform_signal(t,x, alpha)
        plot_signals(t,x,y, alpha)

    print("Program terminated.")


if __name__ == "__main__":
    main()