import numpy as np
import cupy as cp
from multiprocessing import Pool
from qiskit import QuantumCircuit
from cuquantum import *

def build_qsvm_qc(bsp_qc,n_dim,y_t,x_t):
    qc_1 = bsp_qc.assign_parameters(y_t).to_gate()
    qc_2 = bsp_qc.assign_parameters(x_t).inverse().to_gate()
    kernel_qc = QuantumCircuit(n_dim)
    kernel_qc.append(qc_1,list(range(n_dim)))
    kernel_qc.append(qc_2,list(range(n_dim)))
    return kernel_qc

def renew_operand(n_dim,oper_tmp,y_t,x_t):
    oper = oper_tmp.copy()
    n_zg, n_zy_g = [], []
    for d1 in y_t:
        z_g  = np.array([[np.exp(-1j*0.5*d1),0],[0,np.exp(1j*0.5*d1)]])
        n_zg.append(z_g)
        y_g  = np.array([[np.cos(d1/2),-np.sin(d1/2)],[np.sin(d1/2),np.cos(d1/2)]])
        n_zy_g.append(z_g)
        n_zy_g.append(y_g)
    oper[n_dim*2:n_dim*4] = cp.array(n_zy_g)
    oper[n_dim*5-1:n_dim*6-1] = cp.array(n_zg)
    n_zgd, n_zy_gd = [], []
    for d2 in x_t[::-1]:       
        z_gd  = np.array([[np.exp(1j*0.5*d2),0],[0,np.exp(-1j*0.5*d2)]])
        n_zgd.append(z_gd)  
        y_gd  = np.array([[np.cos(d2/2),np.sin(d2/2)],[-np.sin(d2/2),np.cos(d2/2)]])
        n_zy_gd.append(y_gd)
        n_zy_gd.append(z_gd)
    oper[n_dim*6-1:n_dim*7-1] = cp.array(n_zgd)
    oper[n_dim*8-2:n_dim*10-2] = cp.array(n_zy_gd)
    return oper

def data_to_operand(n_dim,operand_tmp,data1,data2,indices_list):
    operand_list = []
    for i1, i2 in indices_list:
        n_op = renew_operand(n_dim,operand_tmp,data1[i1-1],data2[i2-1])
        operand_list.append(n_op) 
    return operand_list

def kernel_matrix_tnsm(y_t, x_t, exp, opers, indices_list, options, mode=None):
    """
    Calculate the kernel matrix using tensor network contraction accelerated by cuQuantum.

    Args:
        y_t (numpy.ndarray): The first set of input data.
        x_t (numpy.ndarray): The second set of input data.
        exp (str): The tensor network expression.
        opers (list): List of tensors.
        indices_list (list): List of index pairs for kernel matrix calculation.
        options (class): Options specified by cuquantum.NetworkOptions for the tensor network.
        mode (str, optional): If 'train', the function will make the kernel matrix symmetric and add a diagonal of ones.

    Returns:
        np.ndarray: The computed kernel matrix.
    """
    
    kernel_matrix = np.zeros((len(y_t),len(x_t)))
    i, oper = -1, opers[0]
    with Network(exp, *oper, options=options) as tn:
        tn.contract_path()
        for i1, i2 in indices_list:
            i += 1
            tn.reset_operands(*opers[i])
            amp_tn = abs(tn.contract()) ** 2
            kernel_matrix[i1-1][i2-1] = np.round(amp_tn,8)
        tn.free()
    if mode == 'train':
        kernel_matrix = kernel_matrix + kernel_matrix.T+np.diag(np.ones((len(x_t))))
        
    return kernel_matrix