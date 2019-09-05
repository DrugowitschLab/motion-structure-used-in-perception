import numpy as np

c = ( np.array([0,2]*2 + [0, 3, 5], dtype='float') + 0.5 ) / 12

fps = 50.

config = dict(
    # # #  Parameters for the simulation of dot motion  # # #
    sim = dict(
        dt = 0.001,                             # Integration time step (Euler integrator)
        # # #  The next four values control stochastic radii (optional feature)  # # #
        tau_vr = 0.001,                         # OU time constant for radial velocities
        tau_r = 0.001,                          # Time constant for stabilizing orbits' radii
        radial_sigma = 0.001,                   # Diffusion of radii
        radial_mean = 1., # np.array([1.]*61 + [0.8]*0 + [0.9]*0)                        # Avg radius of dot orbits
        ),
    # # #  Parameters for the plotting of dots on screen or video  # # #
    display = dict(
        fps = fps,                              # Target frames per second (for live preview and video)
        backend_interactive = "Qt5Agg",         # matplotlib backend for preview (Use "Qt4Agg", "macosx" or "Qt5Agg" if you encounter wrong frame timing with "TkAgg")
        backend_noninteractive = "Agg",         # matplotlib backend for video rendering
        axes_radius = 2.0,                     # Range of the display
        monitor_dpi = 109.,                     # Monitor resolution (dots per inch), required for correct figure and font size
        figsize = (6 * 16/9, 6),                # Figure size (width, height) in inches
        bg_color = "w",                         # Background color
        disc_color = c,                         # Dot color
        disc_radius = np.array([16]*5 + [20]*2),                       # Dot size (same units as fontsize)
        #disc_radius = np.array([16]*2 + [20]*2 + [16]*3),                       # Dot size (same units as fontsize)
        show_labels = False,                    # Identify the dots with numbers?
        label_color = "w",                      # Label color
        label_fontsize = 6,                     # Label fontsize
        show_grid = True,                      # Show a grid in polar coordinates?
        ),
    # experiment phases
    experiment = dict(
        order = ("still", "present", "fade", "track", "predict", "after"),
        still = dict(numFrames = int(round(1. * fps))),
        present = dict(numFrames = int(round(4.5 * fps))),
        fade = dict(numFrames = int(round(0.5 * fps))),
        track = dict(numFrames = int(round(1.5 * fps))),
        predict = dict(numFrames = 0),      # numFrames must be 0
        after = dict(numFrames = int(round(0.5 * fps))),
        f_points = lambda delta: np.round(10 * np.clip(1 - np.abs(delta/(np.pi/2)), 0, 1))
        ),
    # # #  Parameters for video rendering   # # #
    video = dict(
        dpi = 150,                              # Resolution of rendered frames for video (high values: high quality, slow rendering)
        bitrate=1024,                           # Target video size per second (our simple video may not use all of it)
        renderer = 'ffmpeg',                    # External library for rendering (see: https://matplotlib.org/api/animation_api.html#writer-classes)
        codec='libx264',                        # Video codec used for rendering (name may be renderer dependent)
        ),
    DEV = False,                                # Developer mode (show ticks, additional output,...)
    )
