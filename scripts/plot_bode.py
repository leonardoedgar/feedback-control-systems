#!/usr/bin/python3
from utils import create_figure
import matplotlib.pyplot as plt
import control as ctrl
import numpy as np
import sympy as sp
import os


def plot_bode(tf):
	s = sp.symbols('s')

	# Convert sympy expression to control TransferFunction
	num, den = sp.fraction(sp.simplify(tf))  # get numerator and denominator
	num_coeffs = sp.Poly(num, s).all_coeffs()
	den_coeffs = sp.Poly(den, s).all_coeffs()

	# Convert to float
	num_coeffs = [float(c) for c in num_coeffs]
	den_coeffs = [float(c) for c in den_coeffs]

	# Create control TransferFunction
	G = ctrl.TransferFunction(num_coeffs, den_coeffs)

	# Bode plot
	mag, phase, omega = ctrl.bode(G, dB=True, deg=True)
	
	# Get the current figure and axes
	fig = plt.gcf()
	ax_mag = fig.axes[0]   # magnitude plot
	ax_phase = fig.axes[1] # phase plot

	# Increase label font size without changing the text
	ax_mag.xaxis.label.set_size(16)
	ax_mag.yaxis.label.set_size(16)
	ax_phase.xaxis.label.set_size(16)
	ax_phase.yaxis.label.set_size(16)
	ax_mag.tick_params(axis='both', which='major', labelsize=14)
	ax_phase.tick_params(axis='both', which='major', labelsize=14)


def T4Q3(dir_to_save):
	s = sp.symbols('s')
	figure_name = "T4Q3"

	create_figure(figure_name)
	tf = (30 * (s+8)) / (s*(s+2)*(s+4))
	plot_bode(tf)

	filename = os.path.join(dir_to_save, figure_name + ".png")
	plt.savefig(filename)


def main():
	dir_to_save = "figures"
	if not os.path.exists(dir_to_save):
		os.makedirs(dir_to_save)
	T4Q3(dir_to_save)
	plt.show()


if __name__ == '__main__':
	main()	
