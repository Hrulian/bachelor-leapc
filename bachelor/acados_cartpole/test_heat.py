from bachelor.acados_cartpole.my_utils_plot import plot_policy_heatmap

# Basic usage
plot_policy_heatmap(
    actor_path="bachelor/acados_cartpole/real_sac_zop/checkpoints/actor.pth",
    v_fixed=0.0,           # Cart velocity
    thetadot_fixed=0.0,    # Pole angular velocity
    resolution=50          # Grid resolution
)

# # Save to file
# plot_policy_heatmap(
#     actor_path="path/to/actor.pth",
#     save_path="policy_heatmap.png"
# )


