import numpy as np
from IPython.display import Math

# ========================= Space Time Algebra Mathematical Operations ==================================


def wedge(q_bold, p_bold):
    '''
    Computing a wedge product between two spacetime vectors
    '''
    if len(q_bold)!= 4:
        raise ValueError('Spacetime Vector does not have length = 4')

    if len(p_bold)!= 4:
        raise ValueError('Spacetime Vector does not have length = 4')

    q = q_bold[0]
    Q = q_bold[1:]

    p = p_bold[0]
    P = p_bold[1:]

    S = np.zeros((6)) # A 6 Dimensional Vector
    S[:3] = np.cross(Q,P)              # 'Real' Part of the Spin Vector
    S[3:] = q*P-p*Q                    # 'Imaginary' Part of the Spin Vector

    return S

def Re(S):
    '''
    Real Part of the Spin Vector
    '''
    return S[:3]

def Im(S):
    '''
    Imaginary Part of the Spin Vector
    '''
    return S[3:]


def dot(q_bold, p_bold):
    '''
    Computing a Dot Product between two spacetime vectors
    '''

    if len(q_bold)!= len(p_bold):
        raise ValueError('Spacetime Vector lengths do not match! ')

    L = len(q_bold)

    if L == 4:
        q = q_bold[0]
        Q = q_bold[1:]

        p = p_bold[0]
        P = p_bold[1:]

        return p*q - np.dot(Q,P)
    
    if L == 6:
        
        return np.dot(Re(q_bold), Re(p_bold)) - np.dot(Im(q_bold), Im(p_bold) )


# ================ Functions to Prepare Spacetime Objects for the Scipy Optimizer  ===========================

def random_vec():
    '''
    Generates a random spacetime vector
    '''
    b = 1
    a = -1
    real_part = (b-a)*np.random.random((4)) + a
    #imag_part = 2*np.random.random((4)) - 1

    st_vec = real_part #+ 1j*imag_part
    return st_vec


def unpack(data):
    '''
    Unpacks a list of spacetime vectors into a 1D array
    '''
    return np.concatenate(data).flatten()

def repack(data):
    '''
    Does the inverse of 'unpack'
    '''
    return np.split(data, [4,4][0])

# def complex_split(data):
#     '''
#     Splits a complex list into a list of the real parts in the first half of the list and the 
#     imaginary parts in the last half of the list
#     '''
#     real_data = np.real(data)
#     imag_data = np.imag(data)
#     split_list = np.hstack((real_data, imag_data))
#     return split_list

# def complex_join(data):
#     '''
#     Performs the inverse of 'complex_split'
#     '''
#     Middle = len(data)//2
#     real_data = data[:Middle]
#     imag_data = data[Middle:]
#     joined_list = real_data + 1j*imag_data
#     return joined_list

# ============================== Data Visualization =====================================


def printvec(vector):
    """
    Formats a complex vector as a LaTeX column vector.

    Parameters:
        vector (numpy array): A 1D numpy array of complex numbers.

    Returns:
        str: LaTeX string representing the column vector.
    """
    # Format each entry in LaTeX syntax
    formatted_entries = [f"{v.real:.3f} + {v.imag:.3f}i" if v.imag >= 0 else f"{v.real:.3f} - {-v.imag:.3f}i"
                         for v in vector]
    
    # Join entries to form the column vector in LaTeX
    latex_vector = r"\begin{bmatrix} " + r" \\ ".join(formatted_entries) + r" \end{bmatrix}"
    return Math(latex_vector)



def printvec2(vector):
    """
    Formats a complex vector as a LaTeX column vector.

    Parameters:
        vector (numpy array): A 1D numpy array of complex numbers.

    Returns:
        str: LaTeX string representing the column vector.
    """
    vector2_real = Re(vector)
    vector2_imag = Im(vector)

    three_vector = vector2_real + 1j*vector2_imag

    # Format each entry in LaTeX syntax
    formatted_entries = [f"{v.real:.3f} + {v.imag:.3f}i" if v.imag >= 0 else f"{v.real:.3f} - {-v.imag:.3f}i"
                         for v in three_vector]
    
    # Join entries to form the column vector in LaTeX
    latex_vector = r"\begin{bmatrix} " + r" \\ ".join(formatted_entries) + r" \end{bmatrix}"
    return Math(latex_vector)


