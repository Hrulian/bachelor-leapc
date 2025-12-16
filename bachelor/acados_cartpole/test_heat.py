from bachelor.acados_cartpole.my_utils_plot import plot_policy_heatmap

# Basic usage - generates 2 gridded plots:
# 1. Gridded theta_ref heatmap (3x3 grid across v and thetadot)
# 2. Gridded force heatmap (3x3 grid across v and thetadot)
plot_policy_heatmap(
    actor_path="bachelor/acados_cartpole/real_sac_zop/checkpoints/actor.pth",
    v_fixed=0.0,           # Cart velocity for initial grid evaluation
    thetadot_fixed=0.0,    # Pole angular velocity for initial grid evaluation
    resolution=50,         # Grid resolution (50x50 per heatmap)
    plt_show=True,         # Display plots
    save_path=None         # Optional: save plots to file
)


