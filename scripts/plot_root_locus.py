#!/usr/bin/python3
from utils import create_figure
import matplotlib.pyplot as plt
import control as ctrl
import numpy as np
import sympy as sp
import os


def plot_root_locus(oltf, kvect=None, figure_name=None):
	s = sp.symbols('s')
	K = sp.symbols('K', real=True)
	numer_expr, denom_expr = oltf.as_numer_denom()
	num = [float(c) for c in sp.Poly(numer_expr.subs({K: 1}), s).all_coeffs()]
	den = [float(c) for c in sp.Poly(denom_expr, s).all_coeffs()]

	# Create transfer function
	G = ctrl.TransferFunction(num, den)

	# Compute the root locus
	rlist, klist = ctrl.root_locus(G, Plot=False, kvect=kvect)

	# Plot the root locus
	num_poles = rlist.shape[1]
	for i in range(num_poles):
		plt.plot(rlist[:, i].real, rlist[:, i].imag,
			'purple', label='root locus' if i==0 else '')

	# Plot the root locus arrow
	for i in range(num_poles):  # for each pole
		x_real = np.real(rlist[:, i])
		y_imag = np.imag(rlist[:, i])

		# Compute min and max along both axes
		x_min, x_max = x_real.min(), x_real.max()
		y_min, y_max = y_imag.min(), y_imag.max()

		# Midpoint
		x_mid = (x_min + x_max) / 2
		y_mid = (y_min + y_max) / 2

		# Clip midpoint between -10 and 10
		if x_mid > 10:
			x_mid = (x_min + 10) / 2
		elif x_mid < -10:
			x_mid = (x_max - 10) / 2

		if y_mid > 10:
			y_mid = (y_min + 10) / 2
		elif y_mid < -10:
			y_mid = (y_max - 10) / 2

		# Find index closest to this midpoint (Euclidean distance)
		distances = np.sqrt((x_real - x_mid)**2 + (y_imag - y_mid)**2)
		idx = np.argmin(distances)

		# Compute arrow vector (small step)
		prev_idx = max(idx - 1, 0)
		dx = x_real[idx] - x_real[prev_idx]
		dy = y_imag[idx] - y_imag[prev_idx]

		# Plot arrow
		plt.arrow(x_real[idx] - dx/2, y_imag[idx] - dy/2, dx, dy, shape='full',
			lw=0, length_includes_head=True, head_width=0.2, color='purple')

	# Plot the poles
	poles = ctrl.pole(G)  # poles of open-loop system
	plt.plot(poles.real, poles.imag, 'x', markersize=12, color='red', label='K=0 (poles)')
	for p in poles:
		if figure_name == "Q7":
			x_offset, y_offset = -1.1, -0.15
		else:
			x_offset, y_offset = -0.35, -0.8
		plt.text(p.real + x_offset, p.imag + y_offset, f"({p.real:.0f}, {p.imag:.0f})",
			fontsize=12, color='red')

	# Plot zeros
	zeros = ctrl.zero(G)
	if zeros.size != 0:
		plt.plot(zeros.real, zeros.imag, 'o', markersize=12, color="g", label='K=$\infty$ (zeros)')
		for z in zeros:
			plt.text(z.real - 0.35, z.imag - 0.8, f"({z.real:.0f}, {z.imag:.0f})",
				fontsize=12, color='green')

	# Plot y-intercepts
	y_intercepts = get_y_intercepts(oltf)
	y_intercepts = np.array(y_intercepts)[:, :2].tolist()
	for i in range(len(y_intercepts)):
		plt.plot(y_intercepts[i][0], y_intercepts[i][1], 'o', color='black', markersize=4,
			label='y-intercepts (s=j$\omega$)' if i==0 else '')
		plt.text(y_intercepts[i][0] + 0.3, y_intercepts[i][1],
			f"({y_intercepts[i][0]}, {y_intercepts[i][1]})", color='black', fontsize=12)

	# Draw axes (x=0 and y=0) as solid black lines
	plt.axhline(0, color='gray', linewidth=0.8)  # real axis
	plt.axvline(0, color='gray', linewidth=0.8)  # imaginary axis

	# Turn on grid with dotted lines
	plt.grid(True, linestyle=':', linewidth=0.8, color='gray')

	plt.xlim(-10, 10)
	plt.ylim(-10, 10)
	plt.xticks(fontsize=14) 
	plt.yticks(fontsize=14) 
	plt.xlabel("Real", fontsize=18)
	plt.ylabel("Imag", fontsize=18)
	return rlist, klist


def get_oltf(G, H):
	return sp.together(G*H)


def get_char_eq(oltf):
	return sp.together(1 + oltf)


def get_y_intercepts(oltf):
	s = sp.symbols('s')
	K, w = sp.symbols('K w', real=True)
	char_expr = get_char_eq(oltf)
	expr = char_expr.subs({s: sp.I*w})

	# Separate numerator into real and imaginary parts
	num, den = sp.together(expr).as_numer_denom()
	num = num.expand()

	# Separate numerator into real and imaginary parts
	num_real = sp.simplify(sp.re(num))
	num_imag = sp.simplify(sp.im(num))

	# Solve numerator real=0, imag=0 (since denominator ≠ 0)
	solutions = sp.solve([num_real, num_imag], [K, w], dict=True)
	return [(0, sol[w], sol[K]) for sol in solutions]


def plot_damping_ratio(damping_ratio, line_length=5):
	x0, y0 = 0, 0
	theta = np.arctan(np.sqrt(1-damping_ratio**2) / damping_ratio)
	dx, dy = np.cos(theta), np.sin(theta)
	x1, y1 = x0 + line_length*dx, y0 + line_length*dy
	x2, y2 = x0 - line_length*dx, y0 - line_length*dy
	plt.plot([x1, x2], [y1, y2], 'g-', linewidth=1, label='damping ratio')


def plot_root_locus_and_damping_ratio(oltf, damping_ratio, kvect=None,
	figure_name=None):
	rlist, klist = plot_root_locus(oltf, figure_name=figure_name,
		kvect=np.arange(0, 1000, 0.1))
	plot_damping_ratio(damping_ratio, 15)

	tol = 0.01  # tolerance

	# Flatten rlist (it’s npoints x npoles)
	s_vals = rlist.flatten()

	# Exclude s = 0 + 0j
	mask_nonzero = ~(np.isclose(np.real(s_vals), 0) & np.isclose(np.imag(s_vals), 0))
	s_vals = s_vals[mask_nonzero]

	# Compute damping ratios
	sigmas = np.real(s_vals)
	omegas = np.imag(s_vals)
	zetas = -sigmas / np.sqrt(sigmas**2 + omegas**2)

	# Find the closest match
	idx_closest = np.argmin(np.abs(zetas - damping_ratio))
	s_closest = s_vals[idx_closest]

	# Plot the closest match
	plt.plot(s_closest.real, s_closest.imag, marker='s', color='blue',
		linestyle="None", markersize=8, label="$s_{desired}$")
	plt.text(s_closest.real - 0.8, s_closest.imag - 1.5,
		f"({s_closest.real:.1f}, {s_closest.imag:.3f})",
		color='b', fontsize=12)


def T3Q5a(dir_to_save):
	s = sp.symbols('s')
	K = sp.symbols('K', real=True)
	G = K * 10 / (s*(s+1)*(s+4))
	H = 1
	oltf = get_oltf(G, H)

	figure_name = "T3Q5a"
	create_figure(figure_name)
	plot_root_locus(oltf, figure_name=figure_name)
	plt.legend(fontsize=16)
	filename = os.path.join(dir_to_save, figure_name + ".png")
	plt.savefig(filename)


def T3Q6(dir_to_save):
	s = sp.symbols('s')
	K = sp.symbols('K', real=True)
	G = K * (s+1) / ((s+5)*s**2*(s+2))
	H = 1
	oltf = get_oltf(G, H)

	figure_name = "T3Q6"
	create_figure(figure_name)
	plot_root_locus(oltf, figure_name=figure_name)
	plt.legend(fontsize=16)
	filename = os.path.join(dir_to_save, figure_name + ".png")
	plt.savefig(filename)


def T3Q7(dir_to_save):
	s = sp.symbols('s')
	K = sp.symbols('K', real=True)
	G = 2 * K / (s*(s**2+2*s+2))
	H = 1
	oltf = get_oltf(G, H)
	damping_ratio = 0.5

	figure_name = "T3Q7"
	create_figure(figure_name)
	plot_root_locus_and_damping_ratio(oltf, damping_ratio,
		figure_name=figure_name)
	plt.legend(fontsize=16)
	filename = os.path.join(dir_to_save, figure_name + ".png")
	plt.savefig(filename)


def main():
	dir_to_save = "figures"
	if not os.path.exists(dir_to_save):
		os.makedirs(dir_to_save)
	T3Q5a(dir_to_save)
	T3Q6(dir_to_save)
	T3Q7(dir_to_save)
	plt.show()


if __name__ == '__main__':
	main()
