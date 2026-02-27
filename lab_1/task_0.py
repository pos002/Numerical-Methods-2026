import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

"""
Точное решение
"""
def exact_solution(x, y, z, t):
    spatial = np.sin(x) * np.sin(y) * np.sin(z)
    temporal = np.cos(np.sqrt(3) * t) + np.sin(np.sqrt(3) * t) + 0.5 * np.sin(2 * t)
    return spatial * temporal

"""
Неоднородность - f(x, y, z, t)
"""
def source_term(x, y, z, t):
    return -0.5 * np.sin(2 * t) * np.sin(x) * np.sin(y) * np.sin(z)

def solve_wave_3d(N=50, tau=None, T=1.0, save_frames=False):
    L = np.pi
    h = L / N
    x = np.linspace(0, L, N + 1)
    y = np.linspace(0, L, N + 1)
    z = np.linspace(0, L, N + 1)
    
    if tau is None:
        tau = 0.9 * h / np.sqrt(3)
    sigma = tau**2 / h**2
    
    if sigma > 1.0 / 3.0:
        print(f"Нарушено условие Куранта: σ = {sigma:.4f} > 1/3")
    
    # Временная сетка
    Nt = int(np.ceil(T / tau))
    
    # Инициализация слоёв
    u_prev = np.zeros((N + 1, N + 1, N + 1))  # n-1
    u_curr = np.zeros((N + 1, N + 1, N + 1))  # n
    u_next = np.zeros((N + 1, N + 1, N + 1))  # n+1
    
    # Создание сетки для вычислений
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    """
    Начальные условия u(x, y, z, 0) = sin(x)sin(y)sin(z) и u_t(x, y, z, 0) = (sqrt(3) + 1)sin(x)sin(y)sin(z)
    """
    u_curr = np.sin(X) * np.sin(Y) * np.sin(Z)
    ut_curr = (np.sqrt(3) + 1) * np.sin(X) * np.sin(Y) * np.sin(Z)

    # Вторая производная по времени из уравнения: u_tt = Δu + f
    # Для начального момента: Δu = -3·sin(x)sin(y)sin(z)
    utt_curr = -3.0 * u_curr + source_term(X, Y, Z, 0.0)
    # Сохраняем u^0
    u_prev = u_curr.copy()
    # Вычисляем u^1 по формуле Тейлора
    u_curr = u_curr + tau * ut_curr + 0.5 * tau**2 * utt_curr
    
    # Подготовка для анимации (срез при z = π/2)
    frames = []
    times = []
    z_mid = N // 2
    
    if save_frames:
        frames.append(u_curr[:, :, z_mid].copy())
        times.append(tau)
    
    # Основной цикл по времени
    t = tau
    for n in range(1, Nt):
        t += tau
        
        for i in range(1, N):
            for j in range(1, N):
                for k in range(1, N):
                    laplacian = (u_curr[i+1, j, k] + u_curr[i-1, j, k] + u_curr[i, j+1, k] + u_curr[i, j-1, k] + u_curr[i, j, k+1] + u_curr[i, j, k-1] - 6.0 * u_curr[i, j, k])
                    u_next[i, j, k] = (2.0 * u_curr[i, j, k] - u_prev[i, j, k] + sigma * laplacian + tau**2 * source_term(X[i, j, k], Y[i, j, k], Z[i, j, k], t))
        
        # Граничные условия Дирихле
        u_next[0, :, :] = u_next[N, :, :] = 0
        u_next[:, 0, :] = u_next[:, N, :] = 0
        u_next[:, :, 0] = u_next[:, :, N] = 0
        
        # Сдвиг слоёв
        u_prev, u_curr = u_curr.copy(), u_next.copy()
        
        if save_frames and (n % max(1, Nt // 100) == 0):
            frames.append(u_curr[:, :, z_mid].copy())
            times.append(t)
    
    if save_frames:
        return x, y, z, u_curr, np.array(times), np.array(frames), X, Y, Z
    else:
        return x, y, z, u_curr  # Для теста сходимости возвращаем только необходимое

"""
Вычисление L2 и L∞ погрешностей
"""
def compute_error(u_num, x, y, z, t):
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    u_ex = exact_solution(X, Y, Z, t)
    error = u_num - u_ex
    
    linf = np.max(np.abs(error))
    l2 = np.sqrt(np.sum(error**2) * (x[1]-x[0])**3)
    
    return l2, linf, u_ex

"""
Проверка сходимости
"""
def convergence_test():
    print(f"{'N':<6} {'h':<10} {'τ':<10} {'σ':<10} {'L2 error':<12} {'L∞ error':<12}")
    print()
    
    results = []
    T = 0.5
    
    for N in [20, 40, 80]:
        h = np.pi / N
        tau = 0.9 * h / np.sqrt(3)
        sigma = tau**2 / h**2
        
        x, y, z, u_num = solve_wave_3d(N=N, tau=tau, T=T, save_frames=False)
        l2, linf, _ = compute_error(u_num, x, y, z, T)
        
        results.append((N, h, tau, sigma, l2, linf))
        
        if len(results) == 1:
            print(f"{N:<6} {h:<10.4f} {tau:<10.4f} {sigma:<10.4f} {l2:<12.4e} {linf:<12.4e}")
        else:
            print(f"{N:<6} {h:<10.4f} {tau:<10.4f} {sigma:<10.4f} {l2:<12.4e} {linf:<12.4e}")

"""
Визуализация решения с анимацией и 3D графиком
"""
def visualize_solution():
    N = 40
    T = 2.0
    x, y, z, u_num, times, frames, X, Y, Z = solve_wave_3d(N=N, T=T, save_frames=True)
    
    l2_err, linf_err, u_ex = compute_error(u_num, x, y, z, T)
    print(f"\nПогрешность в момент T = {T}:")
    print(f"L2  = {l2_err:.4e}")
    print(f"L∞  = {linf_err:.4e}")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    X2D, Y2D = np.meshgrid(x, y, indexing='ij')
    z_mid_val = z[len(z)//2]
    
    vmin = -2.5
    vmax = 2.5
    im1 = ax1.imshow(frames[0], extent=[0, np.pi, 0, np.pi], 
                     origin='lower', vmin=vmin, vmax=vmax, cmap='RdBu_r')
    im2 = ax2.imshow(exact_solution(X2D, Y2D, z_mid_val, times[0]), 
                     extent=[0, np.pi, 0, np.pi], 
                     origin='lower', vmin=vmin, vmax=vmax, cmap='RdBu_r')
    
    ax1.set_title('Численное решение (срез z=π/2)')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    ax2.set_title('Точное решение (срез z=π/2)')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    time_text = fig.suptitle(f'Время: t = {times[0]:.3f}', fontsize=14)
    
    def animate(i):
        im1.set_array(frames[i])
        im2.set_array(exact_solution(X2D, Y2D, z_mid_val, times[i]))
        time_text.set_text(f'Время: t = {times[i]:.3f}')
        return im1, im2, time_text
    
    anim = FuncAnimation(fig, animate, frames=len(times), interval=50, blit=False)
    
    # Сохранение анимации в GIF
    try:
        writer = PillowWriter(fps=20)
        anim.save('wave_animation.gif', writer=writer)
        from IPython.display import Image, display
        try:
            display(Image('wave_animation.gif'))
        except:
            pass
    except Exception as e:
        print(f"Ошибка при сохранении анимации: {e}")
    
    plt.tight_layout()
    plt.show()
    
    fig = plt.figure(figsize=(14, 10))
    
    t_3d = T
    idx_z = len(z) // 2
    
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    surf1 = ax1.plot_surface(X[:, :, idx_z], Y[:, :, idx_z], u_num[:, :, idx_z],
                            cmap='viridis', edgecolor='none', alpha=0.95, linewidth=0)
    ax1.set_title(f'Численное решение\n(z=π/2, t={t_3d:.1f})', fontsize=11, pad=10)
    ax1.set_xlabel('x', labelpad=8)
    ax1.set_ylabel('y', labelpad=8)
    ax1.set_zlabel('u', labelpad=8)
    ax1.set_zlim(-2.5, 2.5)
    fig.colorbar(surf1, ax=ax1, shrink=0.7, pad=0.1)
    ax1.view_init(elev=30, azim=45)
    
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    u_exact_3d = exact_solution(X[:, :, idx_z], Y[:, :, idx_z], z[idx_z], t_3d)
    surf2 = ax2.plot_surface(X[:, :, idx_z], Y[:, :, idx_z], u_exact_3d,
                            cmap='viridis', edgecolor='none', alpha=0.95, linewidth=0)
    ax2.set_title(f'Точное решение\n(z=π/2, t={t_3d:.1f})', fontsize=11, pad=10)
    ax2.set_xlabel('x', labelpad=8)
    ax2.set_ylabel('y', labelpad=8)
    ax2.set_zlabel('u', labelpad=8)
    ax2.set_zlim(-2.5, 2.5)
    fig.colorbar(surf2, ax=ax2, shrink=0.7, pad=0.1)
    ax2.view_init(elev=30, azim=45)
    
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    error_3d = u_num[:, :, idx_z] - u_exact_3d
    surf3 = ax3.plot_surface(X[:, :, idx_z], Y[:, :, idx_z], error_3d,
                            cmap='coolwarm', edgecolor='none', alpha=0.95, linewidth=0)
    ax3.set_title(f'Погрешность\n(z=π/2, t={t_3d:.1f})', fontsize=11, pad=10)
    ax3.set_xlabel('x', labelpad=8)
    ax3.set_ylabel('y', labelpad=8)
    ax3.set_zlabel('error', labelpad=8)
    ax3.set_zlim(-0.3, 0.3)
    fig.colorbar(surf3, ax=ax3, shrink=0.7, pad=0.1)
    ax3.view_init(elev=30, azim=45)
    
    ax4 = fig.add_subplot(2, 2, 4)
    idx_y = len(y) // 2
    ax4.plot(x, u_num[:, idx_y, idx_z], 'b-o', label='Численное', markersize=4, linewidth=2, alpha=0.8)
    ax4.plot(x, u_ex[:, idx_y, idx_z], 'r--', label='Точное', linewidth=2.5)
    ax4.fill_between(x, u_num[:, idx_y, idx_z], u_ex[:, idx_y, idx_z], 
                     alpha=0.2, color='gray', label='Разница')
    ax4.set_xlabel('x', fontsize=11)
    ax4.set_ylabel('u(x, π/2, π/2, T)', fontsize=11)
    ax4.set_title(f'Профиль решения при T = {T}', fontsize=11, pad=10)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(-2.5, 2.5)
    
    plt.suptitle('3D Визуализация решения волнового уравнения', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()
    
    fig2 = plt.figure(figsize=(10, 8))
    ax = fig2.add_subplot(111, projection='3d')
    
    surf = ax.plot_surface(X[:, :, idx_z], Y[:, :, idx_z], u_num[:, :, idx_z],
                          cmap='plasma', edgecolor='none', alpha=0.95,
                          linewidth=0, antialiased=True)
    
    ax.set_title(f'3D поверхность решения (z=π/2, t={T:.1f})', fontsize=14, pad=20)
    ax.set_xlabel('x', fontsize=12, labelpad=10)
    ax.set_ylabel('y', fontsize=12, labelpad=10)
    ax.set_zlabel('u', fontsize=12, labelpad=10)
    ax.set_zlim(-2.5, 2.5)
    ax.view_init(elev=25, azim=50)
    
    cbar = fig2.colorbar(surf, ax=ax, shrink=0.6, pad=0.1, aspect=15)
    cbar.set_label('Амплитуда волны', rotation=270, labelpad=20, fontsize=11)
    
    plt.tight_layout()
    plt.show()
    
    return anim

if __name__ == "__main__":
    convergence_test()
    anim = visualize_solution()