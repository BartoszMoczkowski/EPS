import numpy as np
import sympy as sp
import scipy as sc
from scipy.integrate import dblquad
from scipy.optimize import fsolve

def sim(data):

    m = data["m"]
    l = data["l"]
    b = data["b"]
    B = data["B"]
    r = data["r"]
    h = data["h"]
    
    v = data["v"]
    R = data["R"]

    g = data["g"]
    f = data["f"]
    K = data["K"]
    mu = data["mu"]

    yaw_rate = v/(R+B/2)

    def jxyi(x2,y2,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        part1 = (Rp - B/2 + x2) * np.sin((l/2-so-y2)*yaw_rate/r/avi)
        part2 = l/2 - so
        part3 = y2 * np.cos((l/2 - so - y2)*yaw_rate/r/avi)

        jyi = part1 - part2 + part3

        part4 = (Rp - B/2 + x2) * (np.cos((l/2-so-y2)*yaw_rate/r/avi) - 1)
        part5 = y2 * np.sin((l/2 - so - y2)*yaw_rate/r/avi)

        jxi = part4 - part5

        return jxi,jyi
    def jxyo(x1,y1,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        part1 = (Rp + B/2 + x1) * np.sin((l/2-so-y1)*yaw_rate/r/avo)
        part2 = l/2 - so
        part3 = y1 * np.cos((l/2 - so - y1)*yaw_rate/r/avo)

        jyo = part1 - part2 + part3

        part4 = (Rp + B/2 + x1) * (np.cos((l/2-so-y1)*yaw_rate/r/avo) - 1)
        part5 = y1 * np.sin((l/2 - so - y1)*yaw_rate/r/avo)

        jxo = part4 - part5

        return jxo,jyo
    
    def Fxo_integrat(x,y,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        jxo,jyo = jxyo(x,y,avi,avo,so)
        jo = np.sqrt(jxo**2 + jyo **2)
        cos_r1 = (-y*yaw_rate)/np.sqrt(((Rp+B/2+x)*yaw_rate-r*avo)**2+(y*yaw_rate)**2)

        beta = np.arcsin(so/R)
        Wo = m*g/2+h*m*v**2/B/Rp*np.cos(beta)
        po = Wo/(b*l) - 12/(b*l**3)*h*m*v**2/2/Rp*np.sin(beta)*(y+so)

        return po*mu*(1-np.exp(-jo/K))*cos_r1
    
    def Fxi_integrat(x,y,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        jxi,jyi = jxyi(x,y,avi,avo,so)
        ji = np.sqrt(jxi**2 + jyi **2)
        cos_r2 = (-y*yaw_rate)/np.sqrt(((Rp-B/2+x)*yaw_rate-r*avi)**2+(y*yaw_rate)**2)
        beta = np.arcsin(so/R)
        Wi = m*g/2-h*m*v**2/B/Rp*np.cos(beta)
        pi = Wi/(b*l) - 12/(b*l**3)*h*m*v**2/2/Rp*np.sin(beta)*(y+so)   
        return pi*mu**(1-np.exp(-ji/K))*cos_r2
    
    def Fyo_integrat(x,y,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        jxo,jyo = jxyo(x,y,avi,avo,so)
        jo = np.sqrt(jxo**2 + jyo **2)
        sin_r1 = ((Rp + B/2 + x)*yaw_rate - r*avo)/np.sqrt(((Rp+B/2+x)*yaw_rate-r*avo)**2+(y*yaw_rate)**2)
        
        beta = np.arcsin(so/R)
        Wo = m*g/2+h*m*v**2/B/Rp*np.cos(beta)
        po = Wo/(b*l) - 12/(b*l**3)*h*m*v**2/2/Rp*np.sin(beta)*(y+so)

        return po*mu*(1-np.exp(-jo/K))*sin_r1

    def Fyi_integrat(x,y,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        jxi,jyi = jxyi(x,y,avi,avo,so)
        ji = np.sqrt(jxi**2 + jyi **2)
        sin_r2 = ((Rp - B/2 + x)*yaw_rate - r*avi)/np.sqrt(((Rp-B/2+x)*yaw_rate-r*avi)**2+(y*yaw_rate)**2)

        beta = np.arcsin(so/R)
        Wi = m*g/2-h*m*v**2/B/Rp*np.cos(beta)
        pi = Wi/(b*l) - 12/(b*l**3)*h*m*v**2/2/Rp*np.sin(beta)*(y+so)   

        return pi*mu*(1-np.exp(-ji/K))*sin_r2

    def Mlo_integrat(x,y,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        jxo,jyo = jxyo(x,y,avi,avo,so)
        jo = np.sqrt(jxo**2 + jyo **2)
        sin_r1 = ((Rp + B/2 + x)*yaw_rate - r*avo)/np.sqrt(((Rp+B/2+x)*yaw_rate-r*avo)**2+(y*yaw_rate)**2)
        beta = np.arcsin(so/R)
        Wo = m*g/2+h*m*v**2/B/Rp*np.cos(beta)
        po = Wo/(b*l) - 12/(b*l**3)*h*m*v**2/2/Rp*np.sin(beta)*(y+so)
        return (B/2 + x)*po*mu*(1-np.exp(-jo/K))*sin_r1

    def Mli_integrat(x,y,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        jxi,jyi = jxyi(x,y,avi,avo,so)
        ji = np.sqrt(jxi**2 + jyi **2)
        sin_r2 = ((Rp - B/2 + x)*yaw_rate - r*avi)/np.sqrt(((Rp-B/2+x)*yaw_rate-r*avi)**2+(y*yaw_rate)**2)
        beta = np.arcsin(so/R)
        Wi = m*g/2-h*m*v**2/B/Rp*np.cos(beta)
        pi = Wi/(b*l) - 12/(b*l**3)*h*m*v**2/2/Rp*np.sin(beta)*(y+so)   
        return (B/2 - x)*pi*mu*(1-np.exp(-ji/K))*sin_r2

    def Mto_integrat(x,y,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        jxo,jyo = jxyo(x,y,avi,avo,so)
        jo = np.sqrt(jxo**2 + jyo **2)

        cos_r1 = (-y*yaw_rate)/np.sqrt(((Rp+B/2+x)*yaw_rate-r*avo)**2+(y*yaw_rate)**2)
        beta = np.arcsin(so/R)
        Wo = m*g/2+h*m*v**2/B/Rp*np.cos(beta)
        po = Wo/(b*l) - 12/(b*l**3)*h*m*v**2/2/Rp*np.sin(beta)*(y+so)
        return y*po*mu*(1-np.exp(-jo/K))*cos_r1

    def Mti_integrat(x,y,avi,avo,so):
        Rp = np.sqrt(R**2 - so**2)
        jxi,jyi = jxyi(x,y,avi,avo,so)
        ji = np.sqrt(jxi**2 + jyi **2)

        cos_r2 = (-y*yaw_rate)/np.sqrt(((Rp-B/2+x)*yaw_rate-r*avi)**2+(y*yaw_rate)**2)
        beta = np.arcsin(so/R)
        Wi = m*g/2-h*m*v**2/B/Rp*np.cos(beta)
        pi = Wi/(b*l) - 12/(b*l**3)*h*m*v**2/2/Rp*np.sin(beta)*(y+so)   
        return y*pi*mu*(1-np.exp(-ji/K))*cos_r2
    
    def integ_calc(avi,avo,so):

        yb1 = -l/2 - so
        yb2 = l/2 - so

        xb1 = -b/2
        xb2 = b/2

        Fxi = -dblquad(lambda y,x : Fxi_integrat(x,y,avi,avo,so),xb1,xb2,yb1,yb2)[0]
        Fxo = -dblquad(lambda y,x : Fxo_integrat(x,y,avi,avo,so),xb1,xb2,yb1,yb2)[0]

        Fyi = -dblquad(lambda y,x : Fyi_integrat(x,y,avi,avo,so),xb1,xb2,yb1,yb2)[0]
        Fyo = -dblquad(lambda y,x : Fyo_integrat(x,y,avi,avo,so),xb1,xb2,yb1,yb2)[0]

        Mli = -dblquad(lambda y,x : Mli_integrat(x,y,avi,avo,so),xb1,xb2,yb1,yb2)[0]
        Mlo = -dblquad(lambda y,x : Mlo_integrat(x,y,avi,avo,so),xb1,xb2,yb1,yb2)[0]

        Mti = -dblquad(lambda y,x : Mti_integrat(x,y,avi,avo,so),xb1,xb2,yb1,yb2)[0]
        Mto = -dblquad(lambda y,x : Mto_integrat(x,y,avi,avo,so),xb1,xb2,yb1,yb2)[0]

        return Fxi,Fxo,Fyi,Fyo,Mli,Mlo,Mti,Mto

    def optimize(vec):

        avi = vec[0]
        avo = vec[1]
        so = vec[2]
        beta = np.arcsin(so/R)
        Fxi,Fxo,Fyi,Fyo,Mli,Mlo,Mti,Mto = integ_calc(avi,avo,so)
        d = (avo-avi)/(avi+avo)
        return np.array([
            Fxo + Fxi + m*v**2/R*np.cos(beta),
            Fyo+Fyi - m*v**2/R*np.sin(beta)-f*m*g,
            Mlo - Mli - B/2*(d*f*m*g) + m*v**2/R*np.cos(beta)*so-Mto-Mti

        ])
    def print_optimize(vec):

        avi = vec[0]
        avo = vec[1]
        so = vec[2]
        beta = np.arcsin(so/R)
        Fxi,Fxo,Fyi,Fyo,Mli,Mlo,Mti,Mto = integ_calc(avi,avo,so)

        return Fxi,Fxo,Fyi,Fyo,Mli,Mlo,Mti,Mto
    
    guesses = [(R-B/2)*yaw_rate/r,(R+B/2)*yaw_rate/r,l/4]

    s = fsolve(optimize,guesses,full_output=True)[0]
    forces = print_optimize(s)

    return [s,forces]


data = {
    "m" : 305,
    "l" : 1.3,
    "b" : 0.168,
    "B" : 0.754+0.168,
    "r" : 0.1,
    "h" : 0.298,
    "g" : 9.81,
    "v" : 1,
    "R" : 1,
    "f": 0.0263,
    "K" : 0.025,
    "mu" : 0.6,
}
for R in [0.5,1,2,5,10,30,100]:

    print(f"starting for r:{R}")
    data["R"] = R
    solution, forces = sim(data)

    Fxi,Fxo,Fyi,Fyo,Mli,Mlo,Mti,Mto = forces

    print(Fyi,Fyo)


