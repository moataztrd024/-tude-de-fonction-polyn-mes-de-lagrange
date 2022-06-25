#!/usr/bin/env python
# coding: utf-8

# In[29]:


#tp1 simulation numerique 
#ex1
#1er methode 
import numpy as np 
import matplotlib.pyplot as plt
#linspace(start, stop, num=)
x=np.linspace(0,5,1000)
def f(x):
    y=1/(1+x**2)
    return y
plt.plot(x,f(x),'c')
plt.show()


# In[31]:


#ex1 2eme M
from pylab import *#pylab permet d utiliser bib.numpy et bib.matplotlib 
#linspace(start, stop, num=)
x=linspace(0,5,1000)
def f(x):
    y=1/(1+x**2)
    return y
plot(x,f(x),'r')
show()


# In[32]:


#ex2
#1)
import numpy as np
import matplotlib.pyplot as plt 
def f(x) :
    return 1/(1+x*x)
 
x= np.linspace (0,5,400)
plt.plot (x,f(x),'c')
plt.show ()

#2)
def lagrange(f,a,b,n) :
    x= np.linspace(a,b,n+1)
    x= np.poly1d ([1,0])
    
    s=0
    for i in range (n+1):
        li = 1
        for j in range (n+1):
            if (i==j) :
                continue 
            else:
                li = li*(x-x(j))/(x(i)-x(j))
        s=s+li*f(x[i])
    return s

k = [lagrange (f,0,5,n) for n in [3,5,10]]
for n in range(3):
    print (k[n])
    
for n in range (3):
    plt.plot(x, np.polyval (k[n], x))


# In[19]:


#ex1 
#cos 1 er methode
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 30)
y = np.cos(x)
plt.plot(x, y)

plt.show() # affiche la figure a l'ecran


# In[20]:


#cos 2 methode
from pylab import *

x = linspace(0,5, 30)
y = cos(x)
plot(x, y)

show() # affiche la figure a l'ecran


# In[21]:


#sin 
#1er methode
from pylab import *

x = linspace(0,5, 30)
y = sin(x)
plot(x, y)

show() # affiche la figure a l'ecran


# In[22]:


#sin 2eme methode
import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 5, 30)
y = np.sin(x)
plt.plot(x, y)

plt.show() # affiche la figure a l'ecran


# In[34]:


#exp
from math import *
from pylab import *
x1=linspace(0,5,1000)
y1=exp(x1)
plot(x1,y1,'m')
show()


# In[ ]:




