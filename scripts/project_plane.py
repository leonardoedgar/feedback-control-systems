#!/usr/bin/python3
from utils import create_figure
import matplotlib.pyplot as plt
import control as ctrl
import numpy as np
import sympy as sp
import os


def project_plane(plane_var, plane_eq, s_coord):
	# Substitute and simplify s=sigma+j*omega
	sigma, omega = sp.symbols('sigma omega', real=True)
	plane_eq_sub = plane_eq.subs({plane_var: sigma + sp.I*omega})
	plane_eq_real = sp.re(plane_eq_sub)
	plane_eq_im = sp.im(plane_eq_sub)

	# Project points
	projected_coord = [(plane_eq_real.subs({sigma: a, omega: b}),
		plane_eq_im.subs({sigma: a, omega: b})) for a,b in s_coord]
	x, y = zip(*projected_coord)

	# Plot the projected points
	plt.plot(x, y, 'b-')

	# Plot additional coordinates with labels
	coord_with_labels = [(-1, 0, "A"), (-1, 1, "B"), (0, 1, "C"),
		(1, 1, "D"), (1, 0, "E"), (1, -1, "F"), (0, -1, "G"),
		(-1, -1, "H")]
	for x, y, label in coord_with_labels: 
		proj_x = plane_eq_real.subs({sigma: x, omega: y})
		proj_y = plane_eq_im.subs({sigma: x, omega: y})
		plt.scatter(proj_x, proj_y, color='red', s=100, marker='o')
		plt.text(proj_x + 0.01, proj_y - 0.01, label, fontsize=18, color='red')

	plt.grid(True, linestyle=':', linewidth=0.8, color='gray')
	plt.xticks(fontsize=14) 
	plt.yticks(fontsize=14) 
	plt.xlabel("Real", fontsize=18)
	plt.ylabel("Imag", fontsize=18)


def get_s_coordinates(center, edge_length, num_points_per_edge=20):
    cx, cy = center
    half = edge_length / 2
    
    # Discretization along edge
    lin = np.linspace(-half, half, num_points_per_edge)
    
    # Four edges relative to center
    bottom = np.column_stack((lin + cx, np.full_like(lin + cx, -half + cy)))
    top    = np.column_stack((lin + cx,  np.full_like(lin + cx, half + cy)))
    left   = np.column_stack((np.full_like(lin + cy, -half + cx), lin + cy))
    right  = np.column_stack((np.full_like(lin + cy, half + cx), lin + cy))
    
    # Stack edges, avoid duplicate corners
    points = np.vstack([bottom, right[1:], top[::-1][1:], left[::-1][1:]])
    
    return points


def T5Q5(dir_to_save):
	s = sp.symbols('s')
	figure_name = "T5Q5"
	center = (0.0, 0.0)
	edge_length = 2.0
	delta_s = 1 + 1/(s+2)

	create_figure(figure_name)
	project_plane(s, delta_s, get_s_coordinates(center, edge_length, 50))

	filename = os.path.join(dir_to_save, figure_name + ".png")
	plt.savefig(filename)


def main():
	dir_to_save = "figures"
	if not os.path.exists(dir_to_save):
		os.makedirs(dir_to_save)
	T5Q5(dir_to_save)
	plt.show()


if __name__ == '__main__':
	main()	
