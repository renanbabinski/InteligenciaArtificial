import numpy as np

#Calculo da sigmóide
def sigmoid(x):
    return 1/(1+np.exp(-x))

#Derivada da função sigmoide
def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))

learnrate = 0.5  #Taxa de aprendizado
x = np.array([1, 2, 3, 4])      #Entradas da rede
y = np.array(0.5)   #Valor esperado 
b = 0.5             #Bias, valor de ativação do neurônio

# Pesos iniciais
w = np.array([0.5, -0.5, 0.3, 0.1])

### Calcule um gradiente descendente para cada peso

# Produto escalar das entradas e pesos da rede
h = np.dot(x, w)+b

# Calculo das saidas da rede neural
nn_output = sigmoid(h)

# Calculo do erro da rede
error = y - nn_output

# Calcule o termo de erro

error_term = error * sigmoid_prime(h)

# Alterações nos pesos
del_w = learnrate * error_term * x

print('Saída da Rede Neural:')
print(nn_output)
print('Tamanho do erro:')
print(error)
print('Alterações nos pesos:')
print(del_w)