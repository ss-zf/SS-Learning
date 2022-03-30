import numpy as np
import matplotlib.pyplot as plt 
###############################################################################
############################## Dubins Approach ################################
###############################################################################
## (LSL,RSR,RSL,LSR) - clockwise and counter clockwise approach
## polynomical curve for N order
## accordoing to the litrature, minimum distance will be covered at 60 degree at circle 2
###############################################################################
###############################################################################
###############################################################################

####################### RSR Clockwise Approach ################################
def RSR_CW(r,TP,alpha):
    
    TP = [TP[0],-1*TP[1]]
    x,y     = TP[0],TP[1]
    theta   = alpha + (np.pi/2) 
    ########################################
    c       = x - (r*np.sin(theta))
    d       = y + (r*np.cos(theta))
    
    temp0  = np.square(c)
    temp1  = np.square(d-r)
    LS     = np.sqrt(temp0 + temp1)                          # Equation 16
    ########################################
    
    temp2  = np.arctan2((d-r),c)
    
    phi_1  = temp2 % (2*np.pi)                              # Equation 17
    ########################################
    temp3   = theta - phi_1
    
    phi_2   = temp3 % (2*np.pi)    
    ########################################
    A         = round(phi_1,3),round(LS,3),round(phi_2,3)
    Len_Curve = round(LS + (r * (phi_1 + phi_2)),3)
    
    ##########################################################################
    ############################### Plotting #################################
    ########################################################################## 
    X,Y = [0],[0]
    for i1 in np.arange(0,A[0],0.0001):
        x_temp = r*np.cos(i1+(np.pi*1.5))
        y_temp = r*np.sin(i1+(np.pi*1.5)) + r
        X.append(x_temp) 
        Y.append(y_temp) 
        
    x_temp = A[1]*np.cos(A[0]) + X[-1]
    y_temp = A[1]*np.sin(A[0]) + Y[-1]
    X.append(x_temp) 
    Y.append(y_temp) 
    
    c_temp = A[1]* np.cos(A[0])
    d_temp = A[1]*np.sin(A[0]) + r
    for i1 in np.arange(0,A[2],0.0001):
        x_temp = r*np.cos(i1 +(np.pi*1.5)+ A[0]) 
        y_temp = r*np.sin(i1 + (np.pi*1.5) + A[0]) 
        X.append(x_temp  + c_temp) 
        Y.append(y_temp + d_temp)
    ########################################################################## 
    Y = -1 * np.array(Y)
    Pts  = [X,Y.tolist()]
    return [A, Len_Curve,Pts]
    
####################### RSR Counter Clockwise Approach ########################

def RSR_CCW(r,TP,alpha):
    
    TP = [TP[0],-1*TP[1]]
    ########################################################################
    
    (x,y)   = TP[0],TP[1]                                 #
    theta   = alpha - (np.pi/2) 
    ########################################
    temp1   = (x - (r*np.sin(theta)))**2
    temp2   = (y + (r*np.cos(theta)) - r)**2
    
    LS      = np.sqrt(temp1 + temp2)                     # Equation 1
    ########################################
    temp3   = np.arctan2(y + (r*np.cos(theta)) - r,x - (r*np.sin(theta)))
    
    phi_1   = temp3 % (2*np.pi)                          # Equation 2
    ########################################
    temp4   = theta - phi_1
    
    phi_2   = temp4 % (2*np.pi)                          # Equation 3
    ########################################
    A         = round(phi_1,8),round(LS,8),round(phi_2,8)
    Len_Curve = round(LS + (r * (phi_1 + phi_2)),3)
    
    ##########################################################################
    ######################## Points Generation ###############################
    ########################################################################## 
    X,Y = [0],[0]
    for i1 in np.arange(0,A[0],0.0001):
        x_temp = r*np.cos(i1+(np.pi*1.5))
        y_temp = r*np.sin(i1+(np.pi*1.5)) + r
        X.append(x_temp) 
        Y.append(y_temp) 
        
    x_temp = A[1]*np.cos(A[0]) + X[-1]
    y_temp = A[1]*np.sin(A[0]) + Y[-1]
    X.append(x_temp) 
    Y.append(y_temp) 
    
    c_temp = A[1]* np.cos(A[0])
    d_temp = A[1]*np.sin(A[0]) + r
    for i1 in np.arange(0,A[2],0.0001):
        x_temp = r*np.cos(i1 +(np.pi*1.5)+ A[0]) 
        y_temp = r*np.sin(i1 + (np.pi*1.5) + A[0]) 
        X.append(x_temp  + c_temp) 
        Y.append(y_temp + d_temp) 
    ########################################################################## 
    Y = -1 * np.array(Y)
    Pts  = [X,Y.tolist()]
    return [A, Len_Curve,Pts]

####################### LSL Clockwise Approach ################################

def LSL_CW(r,TP,alpha):
    """
    Parameters
    ----------
    r            : Minimum turn radius and radius of the target circle
    TP           : Target Point(x,y)
    alpha        : Target Heading Angle (Angular position of the final point on the target circle) ** in radians**

    Returns
    -------
    phi_1       :  Circle 1 arc angle 
    phi_2       :  Circle 2 arc angle
    LS          :  Straight line length

    """
    (x,y)   = TP[0],TP[1]                                 #
    theta   = alpha - (np.pi/2) 
    ########################################
    temp1   = (x - (r*np.sin(theta)))**2
    temp2   = (y + (r*np.cos(theta)) - r)**2
    
    LS      = np.sqrt(temp1 + temp2)                     # Equation 1
    ########################################
    temp3   = np.arctan2(y + (r*np.cos(theta)) - r,x - (r*np.sin(theta)))
    
    phi_1   = temp3 % (2*np.pi)                          # Equation 2
    ########################################
    temp4   = theta - phi_1
    
    phi_2   = temp4 % (2*np.pi)                          # Equation 3
    ########################################
    A         = round(phi_1,8),round(LS,8),round(phi_2,8)
    Len_Curve = round(LS + (r * (phi_1 + phi_2)),3)
     
    ##########################################################################
    ######################## Points Generation ###############################
    ########################################################################## 
    X,Y = [0],[0]
    for i1 in np.arange(0,A[0],0.0001):
        x_temp = r*np.cos(i1+(np.pi*1.5))
        y_temp = r*np.sin(i1+(np.pi*1.5)) + r
        X.append(x_temp) 
        Y.append(y_temp) 
        
    x_temp = A[1]*np.cos(A[0]) + X[-1]
    y_temp = A[1]*np.sin(A[0]) + Y[-1]
    X.append(x_temp) 
    Y.append(y_temp) 
    
    c_temp = A[1]* np.cos(A[0])
    d_temp = A[1]*np.sin(A[0]) + r
    for i1 in np.arange(0,A[2],0.0001):
        x_temp = r*np.cos(i1 +(np.pi*1.5)+ A[0]) 
        y_temp = r*np.sin(i1 + (np.pi*1.5) + A[0]) 
        X.append(x_temp  + c_temp) 
        Y.append(y_temp + d_temp) 
    ########################################################################## 
    Pts  = [X,Y]
    return [A, Len_Curve,Pts]

################# LSL Counter clockwise Approach ############################## 


def LSL_CCW(r,TP,alpha):
    x,y     = TP[0],TP[1]
    theta   = alpha + (np.pi/2) 
    ########################################
    c       = x - (r*np.sin(theta))
    d       = y + (r*np.cos(theta))
    
    temp0  = np.square(c)
    temp1  = np.square(d-r)
    LS     = np.sqrt(temp0 + temp1)                          # Equation 16
    ########################################
    
    temp2  = np.arctan2((d-r),c)
    
    phi_1  = temp2 % (2*np.pi)                              # Equation 17
    ########################################
    temp3   = theta - phi_1
    
    phi_2   = temp3 % (2*np.pi)    
    ########################################
    A         = round(phi_1,3),round(LS,3),round(phi_2,3)
    Len_Curve = round(LS + (r * (phi_1 + phi_2)),3)
    
    ##########################################################################
    ######################## Points Generation ###############################
    ########################################################################## 
    X,Y = [0],[0]
    for i1 in np.arange(0,A[0],0.0001):
        x_temp = r*np.cos(i1+(np.pi*1.5))
        y_temp = r*np.sin(i1+(np.pi*1.5)) + r
        X.append(x_temp) 
        Y.append(y_temp) 
        
    x_temp = A[1]*np.cos(A[0]) + X[-1]
    y_temp = A[1]*np.sin(A[0]) + Y[-1]
    X.append(x_temp) 
    Y.append(y_temp) 
    
    c_temp = A[1]* np.cos(A[0])
    d_temp = A[1]*np.sin(A[0]) + r
    for i1 in np.arange(0,A[2],0.0001):
        x_temp = r*np.cos(i1 +(np.pi*1.5)+ A[0]) 
        y_temp = r*np.sin(i1 + (np.pi*1.5) + A[0]) 
        X.append(x_temp  + c_temp) 
        Y.append(y_temp + d_temp)
    ########################################################################## 
    Pts  = [X,Y]
    return [A, Len_Curve,Pts]
    
###################### RSL Clockwise Approach ################################# 


def RSL_CW(r,TP,alpha):
    CCW = np.array([[0,-1],
                   [1,0]]) 

    CW = np.array([[0,1],
                   [-1,0]]) 
    
    P       = CCW.dot(np.array(TP))
    x,y     = P[0],P[1]
    theta   = alpha - (np.pi/2) 
    ########################################
    temp0   = np.square(x - (r*np.sin(theta)) - r) 
    temp1   = np.square(y + (r*np.cos(theta)))
    L_cc    = np.sqrt(temp0 + temp1)                    # Equation 7
    
    LS      = np.sqrt(np.square(L_cc) - (4*np.square(r)))       # Equation 8
    ########################################
    
    psi_1   = np.arctan2(y + (r*np.cos(theta)), x - (r*np.sin(theta)) - r) # Equation 11
    psi_2   = np.arctan2(2*r,LS)                                           # Equation 12
    
    phi_1   = (-psi_1 + psi_2 + (np.pi/2))%(2*np.pi)                       # Equation 9
    phi_2   = (theta + phi_1 - (np.pi/2)) % (2*np.pi)                      # Equation 10
    ########################################
    A         = round(phi_1,6),round(LS,6),round(phi_2,6)
    Len_Curve = round(LS + (r * (phi_1 + phi_2)),4)
    ##########################################################################
    ######################## Points Generation ###############################
    ########################################################################## 
    X,Y = [0],[0]
    for i1 in np.arange(0,A[0],0.0001):
        x_temp = -(r*np.cos(i1) - r)
        y_temp = r*np.sin(i1)
        X.append(x_temp) 
        Y.append(y_temp) 
    
    y_temp = A[1]*np.cos(A[0]) + Y[-1]
    x_temp = A[1]*np.sin(A[0]) + X[-1]
    X.append(x_temp) 
    Y.append(y_temp) 
    
    c_temp = 0
    d_temp = 0
    c_tol,d_tol = X[-1],Y[-1]
    for i1 in np.arange(0,A[2],0.0001):
        x_temp = r*np.cos(i1 +(np.pi)+ ((np.pi) - A[0])) 
        y_temp = r*np.sin(i1 + (np.pi) + ((np.pi) - A[0])) 
        if i1 == 0:
            c_temp = x_temp
            d_temp = y_temp
            
        X.append(x_temp - c_temp + c_tol) 
        Y.append(y_temp - d_temp + d_tol) 
    
    ########################################################################## 
    Pts   = np.array([X,Y]) 
    P_rtd = CW.dot(Pts)
    return [A, Len_Curve,P_rtd]


################# RSL Counter clockwise Approach ############################## 

def RSL_CCW(r,TP,alpha):
    CCW = np.array([[0,-1],
                   [1,0]]) 

    CW = np.array([[0,1],
                   [-1,0]]) 
    
    P       = CCW.dot(np.array(TP))
    x,y     = P[0],P[1]
    theta   = alpha + (np.pi/2) 
    ########################################
    c       = x - (r*np.sin(theta))
    d       = y + (r*np.cos(theta))
    
    temp0  = np.square(d)
    temp1  = np.square(c-r)
    L_cc   = np.sqrt(temp0 + temp1)                           # Equation 18
    LS     = np.sqrt((L_cc**2) - (4*r**2))                    # Equation 19
    ########################################
    
    temp0  = np.arctan2(2*r, LS)
    temp1  = np.arctan2(d, c-r) 
    
    phi_1  = (temp0 - temp1 +(np.pi/2)) % (2*np.pi)           # Equation 20
    phi_2  = (theta + phi_1 - (np.pi/2)) % (2*np.pi)          # Equation 21
    ########################################
    A         = round(phi_1,3),round(LS,3),round(phi_2,3)
    Len_Curve = round(LS + (r * (phi_1 + phi_2)),3)
    ##########################################################################
    ############################### Plotting #################################
    ########################################################################## 
    X,Y = [0],[0]
    for i1 in np.arange(0,A[0],0.0001):
        x_temp = -(r*np.cos(i1) - r)
        y_temp = r*np.sin(i1)
        X.append(x_temp) 
        Y.append(y_temp) 
    
    y_temp = A[1]*np.cos(A[0]) + Y[-1]
    x_temp = A[1]*np.sin(A[0]) + X[-1]
    X.append(x_temp) 
    Y.append(y_temp) 
    
    c_temp , d_temp = X[-1],Y[-1]
    
    for i1 in np.arange(0,A[2],0.0001):
        x_temp = r*np.cos(i1  + (np.pi*2) - A[0])  
        y_temp = r*np.sin(i1  + (np.pi*2) - A[0]) 
        if i1 == 0:
            x_tol = x_temp
            y_tol = y_temp
        X.append(x_temp  + c_temp - x_tol) 
        Y.append(y_temp  + d_temp - y_tol )
    ########################################################################## 
    Pts   = np.array([X,Y]) 
    P_rtd = CW.dot(Pts)
    return [A, Len_Curve,P_rtd]


##################### LSR Clockwise Approach ################################## 

def LSR_CW(r,TP,alpha):
    TP = [-1*TP[0],TP[1]]
    x,y     = TP[0],TP[1]
    theta   = alpha - (np.pi/2) 
    ########################################
    temp0   = np.square(x - (r*np.sin(theta)) - r) 
    temp1   = np.square(y + (r*np.cos(theta)))
    L_cc    = np.sqrt(temp0 + temp1)                    # Equation 7
    
    LS      = np.sqrt(np.square(L_cc) - (4*np.square(r)))       # Equation 8
    ########################################
    
    psi_1   = np.arctan2(y + (r*np.cos(theta)), x - (r*np.sin(theta)) - r) # Equation 11
    psi_2   = np.arctan2(2*r,LS)                                           # Equation 12
    
    phi_1   = (-psi_1 + psi_2 + (np.pi/2))%(2*np.pi)                       # Equation 9
    phi_2   = (theta + phi_1 - (np.pi/2)) % (2*np.pi)                      # Equation 10
    ########################################
    A         = round(phi_1,6),round(LS,6),round(phi_2,6)
    Len_Curve = round(LS + (r * (phi_1 + phi_2)),4)
    ##########################################################################
    ############################### Plotting #################################
    ########################################################################## 
    X,Y = [0],[0]
    for i1 in np.arange(0,A[0],0.0001):
        x_temp = -(r*np.cos(i1) - r)
        y_temp = r*np.sin(i1)
        X.append(x_temp) 
        Y.append(y_temp) 
    
    y_temp = A[1]*np.cos(A[0]) + Y[-1]
    x_temp = A[1]*np.sin(A[0]) + X[-1]
    X.append(x_temp) 
    Y.append(y_temp) 
    
    c_temp = 0
    d_temp = 0
    c_tol,d_tol = X[-1],Y[-1]
    for i1 in np.arange(0,A[2],0.0001):
        x_temp = r*np.cos(i1 +(np.pi)+ ((np.pi) - A[0])) 
        y_temp = r*np.sin(i1 + (np.pi) + ((np.pi) - A[0])) 
        if i1 == 0:
            c_temp = x_temp
            d_temp = y_temp
            
        X.append(x_temp - c_temp + c_tol) 
        Y.append(y_temp - d_temp + d_tol) 
    
    ########################################################################## 
    X = -1 * np.array(X)
    Pts  = [X.tolist(),Y]
    return [A, Len_Curve,Pts]


##################### LSR Counter clockwise Approach ##########################

def LSR_CCW(r,TP,alpha):
    TP = [-1*TP[0],TP[1]]
    x,y     = TP[0],TP[1]
    theta   = alpha + (np.pi/2) 
    ########################################
    c       = x - (r*np.sin(theta))
    d       = y + (r*np.cos(theta))
    
    temp0  = np.square(d)
    temp1  = np.square(c-r)
    L_cc   = np.sqrt(temp0 + temp1)                           # Equation 18
    LS     = np.sqrt((L_cc**2) - (4*r**2))                    # Equation 19
    ########################################
    
    temp0  = np.arctan2(2*r, LS)
    temp1  = np.arctan2(d, c-r) 
    
    phi_1  = (temp0 - temp1 +(np.pi/2)) % (2*np.pi)           # Equation 20
    phi_2  = (theta + phi_1 - (np.pi/2)) % (2*np.pi)          # Equation 21
    ########################################
    A         = round(phi_1,3),round(LS,3),round(phi_2,3)
    Len_Curve = round(LS + (r * (phi_1 + phi_2)),3)
    ##########################################################################
    ############################### Plotting #################################
    ########################################################################## 
    X,Y = [0],[0]
    for i1 in np.arange(0,A[0],0.0001):
        x_temp = -(r*np.cos(i1) - r)
        y_temp = r*np.sin(i1)
        X.append(x_temp) 
        Y.append(y_temp) 
    
    y_temp = A[1]*np.cos(A[0]) + Y[-1]
    x_temp = A[1]*np.sin(A[0]) + X[-1]
    X.append(x_temp) 
    Y.append(y_temp) 
    
    c_temp , d_temp = X[-1],Y[-1]
    
    for i1 in np.arange(0,A[2],0.0001):
        x_temp = r*np.cos(i1  + (np.pi*2) - A[0])  
        y_temp = r*np.sin(i1  + (np.pi*2) - A[0]) 
        if i1 == 0:
            x_tol = x_temp
            y_tol = y_temp
        X.append(x_temp  + c_temp - x_tol) 
        Y.append(y_temp  + d_temp - y_tol )
    ########################################################################## 
    X = -1 * np.array(X)
    Pts  = [X.tolist(),Y]
    return [A, Len_Curve,Pts]

###############################################################################
###############################################################################
###############################################################################
###############################################################################








 