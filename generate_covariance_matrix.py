# -*- coding: utf-8 -*-
"""
Created on Sat Aug 26 02:45:21 2023

@author: ashysky
"""

import numpy as np

def generate_covariance_matrix(g, p_data, p_noise, sigma_data, sigma_noise, rho_data, rho_noise, Rho):
    sigma_matrix_data = np.ones((p_data, p_data)) * sigma_data
    matrix = np.zeros((p_data, p_data))
    np.fill_diagonal(matrix, 1)
    matrix += rho_data * (1 - np.eye(p_data))
    cov_matrix_basis_data = matrix * sigma_matrix_data

    sigma_matrix_noise = np.ones((p_noise, p_noise)) * sigma_noise
    matrix = np.zeros((p_noise, p_noise))
    np.fill_diagonal(matrix, 1)
    matrix += rho_noise * (1 - np.eye(p_noise))
    cov_matrix_basis_noise = matrix * sigma_matrix_noise

    cov_matrix_basis = np.block([[cov_matrix_basis_data, np.zeros((p_data, p_noise))],
                                 [np.zeros((p_noise, p_data)), cov_matrix_basis_noise]])

    matrix = np.empty((0, 0))  # Initialize the block matrix
    for i in range(g):
        matrix = np.block([[matrix, np.zeros((matrix.shape[0], cov_matrix_basis.shape[1]))],
                           [np.zeros((cov_matrix_basis.shape[0], matrix.shape[1])), cov_matrix_basis]])

    block = np.ones((p_data + p_noise, p_data + p_noise))  # Example block matrix
    matrix1 = np.empty((0, 0))  # Initialize the block matrix
    for i in range(g):
        matrix1 = np.block([[matrix1, np.zeros((matrix1.shape[0], block.shape[1]))],
                           [np.zeros((block.shape[0], matrix1.shape[1])), block]])

    block = np.ones((p_data + p_noise, p_data + p_noise))  # Identity matrix
    matrix2 = np.tile(block, (g, g))
    matrix_temp = matrix2 - matrix1

    matrix3 = np.tile(cov_matrix_basis, (g, g))
    matrix = matrix + Rho * matrix_temp * matrix3

    return matrix

