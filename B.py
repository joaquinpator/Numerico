########################################################################################## Imports

from scipy.optimize import brentq
import timeit #Para calcular tiempo de corrida
import numpy as np #Manejo de arrays
import matplotlib.pylab as plt #Rutinas gráficas

########################################################################################## Funciones f1, f2, f3 y derivadas

def f1(x): 
    return x**2-2
def df1(x): 
    return 2*x
def ddf1(x): 
    return 2

def f2(x): 
    return x**5 - 6.6 * x**4 +5.12 * x**3 + 21.312 * x**2 - 38.016 * x + 17.28
def df2(x): 
    return 5 * x**4 - 26.4 * x**3 + 15.36 * x**2 + 42.624 * x - 38.016
def ddf2(x): 
    return 20 * x**3 - 79.2 * x**2 + 30.72 * x + 42.624

def f3(x): 
    return  (x - 1.5) * np.exp(-4 * (x-1.5)**2)
def df3(x): 
    return (- 8 * x +12.0) * (x - 1.5) * np.exp(-4 * (x-1.5)**2) + np.exp(-4 * (x-1.5)**2)
def ddf3(x): 
    return (-24 * x + ( x - 1.5 ) * (8 * x - 12.0 )**2 + 36.0) * np.exp(-4 * (x - 1.5)**2)


########################################################################################## Funcion busqueda de raices por bisección

def bisec(funcion, a, b, tolerancia, max_iteraciones):

    x = a+(b-a)/2    #mejor que (a+b)/2 segun Burden
    delta = (b-a)/2
    
    print('{0:^4} {1:^17} {2:^17} {3:^17}'.format('i', 'x', 'a', 'b'))
    print('{0:4} {1: .14f} {2: .14f} {3: .14f}'.format(0, x, a, b))
    
    for i in range(max_iteraciones):
        if funcion(a) * funcion(x) > 0:
            a = x
        else:
            b = x
        x_anterior = x
        x = a+(b-a)/2 # vamor medio
        delta = np.abs(x - x_anterior)
        
        print('{0:4} {1: .14f} {2: .14f} {3: .14f}'.format(i+1, x, a, b))
        
        if delta <= tolerancia: #Hubo convergencia
            print('Hubo convergencia en %s iteraciones' % str(i+1))
            return x, delta, i+1
    
    #Si todavia no salio es que no hubo convergencia:
    raise ValueError('No hubo convergencia')
    return x, delta, i+1

########################################################################################## Parámetros pedidos


#Intervalo para buscar raiz
a = 0.0
b = 2.0

#Parametros para el algoritmo
tolerancia1 = 1e-5
tolerancia2 = 1e-13
max_iteraciones = 100


########################################################################################## Grafica de las funciones

extension_graficos = '.png'

# GRAFICA DE f1(x)
xx = np.linspace(a, b, 256+1)
yy = f1(xx)
nombre_funcion = 'f1'
plt.figure(figsize=(10,7))
plt.plot(xx, yy, lw=2)
#plt.legend(loc=best)
plt.xlabel('x')
plt.ylabel(nombre_funcion +'(x)')
plt.title('Funcion '+ nombre_funcion)
plt.grid(True)
plt.savefig(nombre_funcion + extension_graficos)
plt.show()

# GRAFICA DE f2(x)
xx = np.linspace(a, b, 256+1)
yy = f2(xx)
nombre_funcion = 'f2'
plt.figure(figsize=(10,7))
plt.plot(xx, yy, lw=2)
#plt.legend(loc=best)
plt.xlabel('x')
plt.ylabel(nombre_funcion +'(x)')
plt.title('Funcion '+ nombre_funcion)
plt.grid(True)
plt.savefig(nombre_funcion + extension_graficos)
plt.show()

# GRAFICA DE f3(x)
xx = np.linspace(a, b, 256+1)
yy = f3(xx)
nombre_funcion = 'f3'
plt.figure(figsize=(10,7))
plt.plot(xx, yy, lw=2)
#plt.legend(loc=best)
plt.xlabel('x')
plt.ylabel(nombre_funcion +'(x)')
plt.title('Funcion '+ nombre_funcion)
plt.grid(True)
plt.savefig(nombre_funcion + extension_graficos)
plt.show()


########################################################################################## Impresión de resultados


print('\t\t\t----------------')
print('\t\t\tMetodo biseccion')
print('\t\t\t----------------')

print('')

print('\t\t\tFuncion f1, tolerancia = '+str(tolerancia1))
raiz_aprox, delta, n_iter = bisec(f1, a, b, tolerancia1, max_iteraciones)
print('Raiz = ' +str(raiz_aprox))
print('Delta = ' +str(delta))
print('Numero de iteraciones: ' +str(n_iter))

print('') ### Cambio de tolerancia

print('\t\t\tFuncion f1, tolerancia = '+str(tolerancia2))
raiz_aprox, delta, n_iter = bisec(f1, a, b, tolerancia2, max_iteraciones)
print('Raiz = ' +str(raiz_aprox))
print('Delta = ' +str(delta))
print('Numero de iteraciones: ' +str(n_iter))

print('')
############# Cambio de funcion
print('')

print('\t\t\tFuncion f2, tolerancia = '+str(tolerancia1))
raiz_aprox, delta, n_iter = bisec(f2, a, b, tolerancia1, max_iteraciones)
print('Raiz = ' +str(raiz_aprox))
print('Delta = ' +str(delta))
print('Numero de iteraciones: ' +str(n_iter))

print('') ### Cambio de tolerancia

print('\t\t\tFuncion f2, tolerancia = '+str(tolerancia2))
raiz_aprox, delta, n_iter = bisec(f2, a, b, tolerancia2, max_iteraciones)
print('Raiz = ' +str(raiz_aprox))
print('Delta = ' +str(delta))
print('Numero de iteraciones: ' +str(n_iter))

print('')
############# Cambio de funcion
print('')

print('\t\t\tFuncion f3, tolerancia = '+str(tolerancia1))
raiz_aprox, delta, n_iter = bisec(f3, a, b, tolerancia1, max_iteraciones)
print('Raiz = ' +str(raiz_aprox))
print('Delta = ' +str(delta))
print('Numero de iteraciones: ' +str(n_iter))

print('') ### Cambio de tolerancia

print('\t\t\tFuncion f3, tolerancia = '+str(tolerancia2))
raiz_aprox, delta, n_iter = bisec(f3, a, b, tolerancia2, max_iteraciones)
print('Raiz = ' +str(raiz_aprox))
print('Delta = ' +str(delta))
print('Numero de iteraciones: ' +str(n_iter))

print('')
