import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import time

"""Точное решение"""
def exact_solution(x, y, z, t):
    spatial = np.sin(x) * np.sin(y) * np.sin(z)
    temporal = np.cos(np.sqrt(3) * t) + np.sin(np.sqrt(3) * t) + 0.5 * np.sin(2 * t)
    return spatial * temporal

"""Неоднородность f(x,y,z,t)"""
def source_term(X, Y, Z, t):
    return -0.5 * np.sin(2 * t) * np.sin(X) * np.sin(Y) * np.sin(Z)

"""Решение 3D волнового уравнения методом конечных разностей"""
def solve_wave_3d(N = 50, tau = None, T = 1.0, save_frames = False):
    L = np.pi
    h = L / N
    x = np.linspace(0, L, N + 1)
    y = np.linspace(0, L, N + 1)
    z = np.linspace(0, L, N + 1)
    
    if tau is None:
        tau = 1.5 * h / np.sqrt(3)
    
    Nt = int(np.ceil(T / tau))
    # tau = T / Nt  # Корректировка для точного попадания в T
    sigma = tau**2 / h**2
    
    if sigma > 1.0 / 3.0 + 1e-10:
        print(f"Нарушено условие Куранта: σ = {sigma:.4f} > 1/3")

    # Три временных слоя
    u_prev = np.zeros((N + 1, N + 1, N + 1))
    u_curr = np.zeros((N + 1, N + 1, N + 1))
    u_next = np.zeros((N + 1, N + 1, N + 1))
    
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Начальные условия
    u0 = np.sin(X) * np.sin(Y) * np.sin(Z)
    ut0 = (np.sqrt(3) + 1) * np.sin(X) * np.sin(Y) * np.sin(Z)
    utt0 = -3.0 * u0 + source_term(X, Y, Z, 0.0)
    
    u_prev = u0.copy()
    u_curr = u0 + tau * ut0 + 0.5 * tau**2 * utt0
    
    # Граничные условия
    u_curr[0,:,:] = u_curr[N,:,:] = 0
    u_curr[:,0,:] = u_curr[:,N,:] = 0
    u_curr[:,:,0] = u_curr[:,:,N] = 0
    
    frames = []
    times = []
    z_mid = N // 2
    
    if save_frames:
        frames.append(u_curr[:, :, z_mid].copy())
        times.append(tau)
    
    # Основной цикл по времени
    for n in range(1, Nt):
        t = n * tau
        
        # Векторизованный лапласиан
        laplacian = (u_curr[2:, 1:-1, 1:-1] + u_curr[:-2, 1:-1, 1:-1] +
                     u_curr[1:-1, 2:, 1:-1] + u_curr[1:-1, :-2, 1:-1] +
                     u_curr[1:-1, 1:-1, 2:] + u_curr[1:-1, 1:-1, :-2] -
                     6.0 * u_curr[1:-1, 1:-1, 1:-1])
        
        src = source_term(X[1:-1, 1:-1, 1:-1], Y[1:-1, 1:-1, 1:-1], Z[1:-1, 1:-1, 1:-1], t)
        
        u_next[1:-1, 1:-1, 1:-1] = (2.0 * u_curr[1:-1, 1:-1, 1:-1] - 
                                     u_prev[1:-1, 1:-1, 1:-1] + 
                                     sigma * laplacian + 
                                     tau**2 * src)
        
        # Граничные условия Дирихле
        u_next[0, :, :] = u_next[N, :, :] = 0
        u_next[:, 0, :] = u_next[:, N, :] = 0
        u_next[:, :, 0] = u_next[:, :, N] = 0
        
        u_prev, u_curr = u_curr.copy(), u_next.copy()
        
        if save_frames and (n % max(1, Nt // 100) == 0):
            frames.append(u_curr[:, :, z_mid].copy())
            times.append(t)
    
    if save_frames:
        return x, y, z, u_curr, np.array(times), np.array(frames), X, Y, Z
    else:
        return x, y, z, u_curr

"""Вычисление L2-нормы погрешности"""
def compute_error(u_num, x, y, z, t):
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u_ex = exact_solution(X, Y, Z, t)
    error = u_num - u_ex
    
    h = x[1] - x[0]
    l2 = np.sqrt(np.sum(error**2) * h**3)
    return l2, u_ex

"""Проверка сходимости схемы"""
def convergence_test():
    print(f"{'N':<6} {'h':<10} {'τ':<10} {'σ':<10} {'L2 error':<14} {'Order L2':<10}")
    
    results = []
    T = 0.5
    
    for N in [20, 40, 80, 160]:
        h = np.pi / N
        tau = 1.5 * h / np.sqrt(3)
        sigma = tau**2 / h**2
        
        start_time = time.time()
        x, y, z, u_num = solve_wave_3d(N = N, tau = tau, T = T, save_frames = False)
        elapsed = time.time() - start_time
        
        l2, _ = compute_error(u_num, x, y, z, T)
        results.append((N, h, tau, sigma, l2, elapsed))

    for i, item in enumerate(results):
        N, h, tau, sigma, l2, elapsed = item
        
        if i == 0:
            # Первая строка — порядка ещё нет
            print(f"{N:<6} {h:<10.4f} {tau:<10.4f} {sigma:<10.4f} {l2:<14.4e}")
        else:
            # Последующие строки — считаем порядок
            h_prev, l2_prev = results[i-1][1], results[i-1][4]
            order_l2 = np.log(l2_prev / l2) / np.log(h_prev / h)
            print(f"{N:<6} {h:<10.4f} {tau:<10.4f} {sigma:<10.4f} {l2:<14.4e} {order_l2:<10.3f}")
    
    return results

"""Построение графика сходимости"""
def plot_convergence(results):
    plt.rcParams.update({'font.size': 12})
    
    h_vals = np.array([r[1] for r in results])
    l2_vals = np.array([r[4] for r in results])
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    
    # L2-ошибка vs h
    ax.loglog(h_vals, l2_vals, 'bo-', label='L₂ ошибка', linewidth=2.5, 
              markersize=10, markerfacecolor='white', markeredgewidth=2)
    
    # Линия O(h²)
    h_ref = np.array([h_vals[0], h_vals[-1]])
    ax.loglog(h_ref, l2_vals[0] * (h_ref / h_vals[0])**2, 'm--', 
              label='O(h²) — второй порядок', alpha=0.8, linewidth=2.5)
    
    ax.set_xlabel('Шаг сетки $h$', fontsize=13, labelpad=10)
    ax.set_ylabel('Погрешность $L_2$', fontsize=13, labelpad=10)
    ax.set_title('Сходимость конечно-разностной схемы', fontsize=14, fontweight='bold', pad=15)
    ax.grid(True, which='both', alpha=0.4, linestyle='--')
    ax.legend(fontsize=11, framealpha=0.95)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.show()

"""Визуализация решения"""
def visualize_solution():
    N = 40
    T = 2.0

    x, y, z, u_num, times, frames, X, Y, Z = solve_wave_3d(N = N, T = T, save_frames = True)
    
    l2_err, u_ex = compute_error(u_num, x, y, z, T)
    print(f"\nПогрешность L₂ в момент T = {T}: {l2_err:.4e}")
    
    # Анимация
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    X2D, Y2D = np.meshgrid(x, y, indexing='ij')
    z_mid_val = z[len(z)//2]
    
    im1 = ax1.imshow(frames[0], extent=[0, np.pi, 0, np.pi], origin='lower', vmin=-2.5, vmax=2.5, cmap='RdBu_r')
    im2 = ax2.imshow(exact_solution(X2D, Y2D, z_mid_val, times[0]), extent=[0, np.pi, 0, np.pi], origin='lower', vmin=-2.5, vmax=2.5, cmap='RdBu_r')
    
    ax1.set_title('Численное решение (срез z=π/2)')
    ax2.set_title('Точное решение (срез z=π/2)')
    time_text = fig.suptitle(f'Время: t = {times[0]:.3f}', fontsize=13)
    
    def animate(i):
        im1.set_array(frames[i])
        im2.set_array(exact_solution(X2D, Y2D, z_mid_val, times[i]))
        time_text.set_text(f'Время: t = {times[i]:.3f}')
        return im1, im2, time_text
    
    anim = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=False)
    
    try:
        writer = PillowWriter(fps=20)
        anim.save('wave_animation.gif', writer=writer)
        print("Анимация сохранена как 'wave_animation.gif'")
    except Exception as e:
        print(f"Не удалось сохранить анимацию: {e}")
    
    plt.tight_layout()
    plt.show()
    
    # 3D графики
    fig = plt.figure(figsize=(16, 11))
    idx_z = len(z) // 2

    # Численное решение
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X[:, :, idx_z], Y[:, :, idx_z], u_num[:, :, idx_z], 
                            cmap='viridis', edgecolor='none', alpha=0.95)
    ax1.set_title(f'Численное решение (t={T})', pad=15)
    ax1.set_xlabel('x', labelpad=10)
    ax1.set_ylabel('y', labelpad=10)
    ax1.set_zlabel('u', labelpad=10)
    cbar1 = fig.colorbar(surf1, ax=ax1, shrink=0.6, pad=0.15, aspect=15)
    cbar1.set_label('u', rotation=270, labelpad=15)

    # Точное решение
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    u_ex_3d = exact_solution(X[:, :, idx_z], Y[:, :, idx_z], z[idx_z], T)
    surf2 = ax2.plot_surface(X[:, :, idx_z], Y[:, :, idx_z], u_ex_3d, 
                            cmap='viridis', edgecolor='none', alpha=0.95)
    ax2.set_title(f'Точное решение (t={T})', pad=15)
    ax2.set_xlabel('x', labelpad=10)
    ax2.set_ylabel('y', labelpad=10)
    ax2.set_zlabel('u', labelpad=10)
    cbar2 = fig.colorbar(surf2, ax=ax2, shrink=0.6, pad=0.15, aspect=15)
    cbar2.set_label('u', rotation=270, labelpad=15)

    # Погрешность
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    error_3d = u_num[:, :, idx_z] - u_ex_3d
    surf3 = ax3.plot_surface(X[:, :, idx_z], Y[:, :, idx_z], error_3d, 
                            cmap='coolwarm', edgecolor='none', alpha=0.95)
    ax3.set_title(f'Погрешность (t={T})', pad=15)
    ax3.set_xlabel('x', labelpad=10)
    ax3.set_ylabel('y', labelpad=10)
    ax3.set_zlabel('error', labelpad=10)
    cbar3 = fig.colorbar(surf3, ax=ax3, shrink=0.6, pad=0.15, aspect=15)
    cbar3.set_label('error', rotation=270, labelpad=15)

    # Профиль
    ax4 = fig.add_subplot(2, 2, 4)
    idx_y = len(y) // 2
    ax4.plot(x, u_num[:, idx_y, idx_z], 'b-o', label='Численное', markersize=3, linewidth=1.5)
    ax4.plot(x, u_ex[:, idx_y, idx_z], 'r--', label='Точное', linewidth=2)
    ax4.set_xlabel('x', labelpad=10)
    ax4.set_ylabel('u', labelpad=10)
    ax4.set_title(f'Профиль при y=z=π/2', pad=10)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.suptitle('3D Визуализация', fontsize=15, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.0, w_pad=2.0)
    plt.show()
    
    return anim

if __name__ == "__main__":
    results = convergence_test()

    # start = time.perf_counter()
    # print(f"время {time.perf_counter() - start:.2f} с")
    # plot_convergence(results)
    # anim = visualize_solution()