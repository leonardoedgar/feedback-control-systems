#!/usr/bin/python3
import matplotlib.pyplot as plt


def create_figure(figure_name=None, dpi=100):
	# Desired size in pixels
	width_px = 1848
	height_px = 1094

	# Convert pixels to inches
	width_in = width_px / dpi
	height_in = height_px / dpi

	return plt.figure(figure_name, figsize=(width_in, height_in), dpi=dpi)
