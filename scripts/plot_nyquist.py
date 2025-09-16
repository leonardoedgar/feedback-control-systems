#!/usr/bin/python3
from utils import create_figure
import matplotlib.pyplot as plt
import control as ctrl
import numpy as np
import sympy as sp
import os


def plot_nyquist(tf):
	s = sp.symbols('s')

	num, den = sp.fraction(sp.simplify(tf))
	num_coeffs = [float(c) for c in sp.Poly(num, s).all_coeffs()]
	den_coeffs = [float(c) for c in sp.Poly(den, s).all_coeffs()]

	# Create control TransferFunction
	G = ctrl.TransferFunction(num_coeffs, den_coeffs)

	# Frequency range for positive ω
	omega = np.logspace(-3, 3, 1000)

	mag, phase, omega = ctrl.freqresp(G, omega)
	
	# Cartesian coords
	x_pos = mag * np.cos(phase)
	y_pos = mag * np.sin(phase)
	x_neg = mag * np.cos(-phase)
	y_neg = mag * np.sin(-phase)

	# Plot Nyquist curves
	plt.plot(x_pos, y_pos, '-', color="b", label='$+\omega$')
	plt.plot(x_neg, y_neg, '--', color="b", label='$-\omega$')
	plt.plot(-1, 0, 'rx', markersize=10, label='(-1, 0)')

	# ---- Center of Cartesian coordinates ----
	cx = (x_pos.min() + x_pos.max()) / 2
	cy = (y_pos.min() + y_pos.max()) / 2

	# ---- Find curve point closest to (cx, cy) ----
	dist2 = (x_pos - cx)**2 + (y_pos - cy)**2
	mid_idx = np.argmin(dist2)

	# compute displacement vector between two points
	x0_pos, y0_pos = x_pos[mid_idx-1], y_pos[mid_idx-1]
	x1_pos, y1_pos = x_pos[mid_idx+1], y_pos[mid_idx+1]
	x0_neg, y0_neg = x_neg[mid_idx-1], y_neg[mid_idx-1]
	x1_neg, y1_neg = x_neg[mid_idx+1], y_neg[mid_idx+1]
	dx_pos, dy_pos = x1_pos-x0_pos, y1_pos-y0_pos
	dx_neg, dy_neg = x0_neg-x1_neg, y0_neg-y1_neg

	# Arrow for +ω
	plt.arrow(
		x0_pos, y0_pos, dx_pos, dy_pos,
		length_includes_head=True,
		head_width=0.02,
		head_length=0.05,
		fc="b", ec="b", lw=1.5
	)

	# Arrow for -ω
	plt.arrow(
		x0_neg, y0_neg, dx_neg, dy_neg,
		length_includes_head=True,
		head_width=0.02,
		head_length=0.05,
		fc="b", ec="b", lw=1.5
	)

	plt.title("Nyquist Plot", fontsize=16)
	plt.xlabel('Re[G(j$\omega$)]', fontsize=16)
	plt.ylabel('Im[G(j$\omega$)]', fontsize=16)
	plt.grid(True, linestyle=':')


def plot_nyquist_built_in(tf):
	s = sp.symbols('s')

	num, den = sp.fraction(sp.simplify(tf))
	num_coeffs = [float(c) for c in sp.Poly(num, s).all_coeffs()]
	den_coeffs = [float(c) for c in sp.Poly(den, s).all_coeffs()]

	# Create control TransferFunction
	G = ctrl.TransferFunction(num_coeffs, den_coeffs)
	
	# Plot the Nyquist plot
	num_circle, res = ctrl.nyquist_plot(
		G, arrows=1, arrow_size=12, return_contour=True,
		color='b', start_marker_size=0, primary_style=['-','-'],
		mirror_style=['--', '--'])
	plt.plot(-1, 0, 'r+', markersize=10, label='(-1, 0)')

	# Display the plot
	plt.title("Nyquist Plot", fontsize=16)
	plt.xlabel('Re[G(j$\omega$)]', fontsize=16)
	plt.ylabel('Im[G(j$\omega$)]', fontsize=16)
	plt.grid(True, linestyle=':')


def T5Q1(dir_to_save):
	s = sp.symbols('s')
	tf = 1/((s/2 + 1)*(2*s + 1))
	figure_name = "T5Q1"

	create_figure(figure_name)	
	plot_nyquist(tf)

	plt.legend(fontsize=16)
	filename = os.path.join(dir_to_save, figure_name + ".png")
	plt.savefig(filename)


def T5Q2(dir_to_save):
	s = sp.symbols('s')
	K = sp.symbols('K', real=True)
	tf = (K*(s+2))/(s**2*(s+4))
	tf = tf.subs(K, 1)
	figure_name = "T5Q2"

	create_figure(figure_name)	
	plot_nyquist_built_in(tf)

	plt.legend(fontsize=16)
	filename = os.path.join(dir_to_save, figure_name + ".png")
	plt.savefig(filename)


def main():
	dir_to_save = "figures"
	if not os.path.exists(dir_to_save):
		os.makedirs(dir_to_save)
	T5Q1(dir_to_save)
	T5Q2(dir_to_save)
	plt.show()


if __name__ == '__main__':
	main()
