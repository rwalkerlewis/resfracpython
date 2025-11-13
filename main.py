import numpy as np
import matplotlib.pyplot as plt
import random_fractures

centers, thetas = random_fractures.generate_fracture_data(
        Nf=30,
        domain=(-800, 800, -800, 800, 0, 100),
        seed=42
    )

L, H = 200.0, 80.0

    # Plot top-down view
random_fractures.plot_fractures_topdown(centers, thetas, L, H, title="300 Random Fractures (Top View)")