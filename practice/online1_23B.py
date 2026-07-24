import numpy as np
import matplotlib.pyplot as plt

DT = 0.05 # sampling interval for the time axis
T_MIN, T_MAX = -np.pi, np.pi # x(t) is defined only on this range

def generate_time_axis(t_min=T_MIN, t_max=T_MAX, dt=DT):
    return np.arange(t_min, t_max + dt / 2, dt)


def base_signal(t):
    x = np.sin(t)
    x[(t < T_MIN) | (t > T_MAX)] = 0
    return x

def interpolate_signal(t, x, query_t):
    # TODO: implement interpolation   
    y = np.zeros_like(query_t)
    for i, tq in enumerate(query_t):
        if tq < T_MIN or tq > T_MAX:
            y[i] = 0
        else:
            idx = np.searchsorted(t, tq)
            if idx < len(t) and t[idx]==tq:
                y[i] = x[idx]
            else:
                left = idx-1
                right = idx
                y[i] = (x[left] + x[right])/2
    return y

def transform_signal(t, x, alpha, beta):
    # TODO: implement transformation
    # Missing values will be calculated from the average of the nearest left and  right values.
    t_new = alpha*t + beta
    y = interpolate_signal(t,x,t_new)
    return y

def plot_signals(t, x, y, alpha, beta):
    plt.figure(figsize=(9, 5))
    plt.plot(t, x, label="x(t)", linewidth=2)
    plt.plot(t, y, label=f"y(t) = x({alpha}t + {beta})", linewidth=2, linestyle="--")
    plt.title("Time Scaling and Shifting of a Signal")
    plt.xlabel("t")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    t = generate_time_axis()
    x = base_signal(t)

    print("Enter alpha and beta to plot y(t) = x(alpha*t + beta).")
    print("Type 'q' at any prompt to quit.\n")

    while True:
        alpha_input = input("Enter alpha (q to quit): ")

        if alpha_input.lower()=="q":
            break

        beta_input = input("Enter beta (q to quit): ")

        if beta_input.lower()=="q":
            break

        alpha = float(alpha_input)
        beta = float(beta_input)

        y = transform_signal(t,x,alpha,beta)
        plot_signals(t,x,y,alpha,beta)

print("Exiting.")
if __name__ == "__main__":
    main()