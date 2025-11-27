import sys
import csv
import numpy as np
import imageio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from my_env import CartPoleEnv, CartPoleEnvConfig


def read_log(path):
    t, x, theta, v, thetadot, u = [], [], [], [], [], []
    with open(path, newline='') as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                t.append(float(row.get('t', '')))
            except Exception:
                t.append(None)
            def g(n):
                try:
                    return float(row.get(n, '') or 0.0)
                except Exception:
                    return 0.0
            x.append(g('x_m'))
            theta.append(g('theta'))
            v.append(g('v_m_s'))
            thetadot.append(g('thetadot'))
            u.append(g('u_pwm'))
    return np.array(t), np.array(x), np.array(theta), np.array(v), np.array(thetadot), np.array(u)


def render_video_from_csv(csv_path, out_path, fps=30):
    # parse CSV using the robust CSV reader which handles headers
    t, x, theta, v, thetadot, u = read_log(csv_path)
    if len(x) == 0:
        raise SystemExit('No rows found in CSV')

    # estimate dt (fallback to 1/fps)
    if np.any(np.isnan(t)) or len(t) < 2:
        dt = 1.0 / fps
    else:
        diffs = np.diff(t)
        dt = float(np.median(diffs)) if np.all(diffs > 0) else 1.0 / fps

    cfg = CartPoleEnvConfig(dt=dt)
    env = CartPoleEnv(render_mode='rgb_array', cfg=cfg)

    # Only render the environment frame (no trailing plot)
    fig, ax_img = plt.subplots(1, 1, figsize=(6, 4))
    canvas = FigureCanvas(fig)

    img_handle = None
    ax_img.axis('off')

    writer = imageio.get_writer(out_path, fps=fps)

    N = len(x)
    for i in range(N):
        state = np.array([x[i], theta[i], v[i], thetadot[i]], dtype=np.float32)
        env.x = state
        frame = env.render()  # rgb_array as numpy HxWx3

        if img_handle is None:
            img_handle = ax_img.imshow(frame)
            ax_img.axis('off')
        else:
            img_handle.set_data(frame)

        # no trailing plot to update; only update the image panel

        canvas.draw()
        w, h = fig.canvas.get_width_height()

        try:
            buf = canvas.tostring_rgb()
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        except AttributeError:
            if hasattr(canvas, "buffer_rgba"):
                buf = canvas.buffer_rgba()
                img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)[:, :, :3]
            else:
                buf = canvas.tostring_argb()
                arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
                img = arr[:, :, 1:4]  # take R,G,B

        writer.append_data(img)

    writer.close()
    print(f'Wrote video to {out_path}')


if __name__ == '__main__':
    from pathlib import Path

    script_dir = Path(__file__).parent

    if len(sys.argv) == 1:
        candidate = script_dir / 'real_cartpole_log.csv'
        if not candidate.exists():
            print(f'No arguments provided and {candidate} not found.')
            print('Usage: python render_from_csv.py <pwm_log.csv> <out.mp4> [fps]')
            raise SystemExit(1)
        csv_path = str(candidate)
        out_path = str(candidate.with_suffix('.mp4'))
        fps = 20
    else:
        csv_path = sys.argv[1]
        if len(sys.argv) >= 3:
            out_path = sys.argv[2]
        else:
            out_path = str(Path(csv_path).with_suffix('.mp4'))
        fps = int(sys.argv[3]) if len(sys.argv) >= 4 else 20

    render_video_from_csv(csv_path, out_path, fps=fps)
